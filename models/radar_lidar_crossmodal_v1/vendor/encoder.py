import torch
import torch.nn as nn
import numpy as np
import config

# Handle spconv import gracefully
try:
    import spconv.pytorch as spconv
except ImportError:
    print("⚠️ WARNING: spconv not found. CUDA operations will fail.")
    spconv = None


class SparseResBlock(spconv.SparseModule):
    """
    Standard Residual Block for Sparse Convolutions.
    Consists of two SubM convolutions with a skip connection.    
    """
    def __init__(self, in_channels, out_channels, stride=1, indice_key=None):
        super().__init__()
        
        if indice_key is None:
            indice_key = f"res_{in_channels}_{out_channels}"

        # Shortcut
        self.shortcut = spconv.SparseSequential()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = spconv.SparseSequential(
                spconv.SparseConv3d(
                    in_channels, out_channels, 3, stride=stride, 
                    padding=1, bias=False, indice_key=f"{indice_key}_skip"
                ),
                nn.BatchNorm1d(out_channels)
            )
            
        # Conv 1
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, 3, padding=1, 
            bias=False, indice_key=indice_key
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # Conv 2
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, 3, padding=1, 
            bias=False, indice_key=indice_key
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        
        # Somma residua
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out


class PFE_Engine(nn.Module):
    def __init__(self, num_input_features=4, latent_dim=config.LATENT_DIM):
        
        super().__init__()
        
        # --- 1. Parametri Geometrici ---
        self.voxel_size = torch.tensor(config.VOXEL_SIZE).cuda()
        self.pc_range = torch.tensor(config.POINT_CLOUD_RANGE).cuda()
        self.grid_size_zyx = config.GRID_SIZE[::-1]
        self.grid_size_tensor = torch.tensor(config.GRID_SIZE).cuda()

        # --- 2. Input MLP OTTIMIZZATO ---
        # 4 -> 16 -> 32 (Più leggero di 4->64)
        self.input_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # --- 3. Sparse Backbone MICRO ---
        
        self.backbone = spconv.SparseSequential(
            # ============================================================
            # SCALA 1/1 (Full Resolution)
            # Canali: 32 (Input) -> 16
            # ============================================================
            spconv.SubMConv3d(32, 16, 3, padding=1, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(16), 
            nn.ReLU(),
            SparseResBlock(16, 16, indice_key="res_scale1a"),
            # Rimosso secondo blocco qui per risparmiare, ne basta 1 a res piena
            
            # ============================================================
            # SCALA 1/2
            # Canali: 16 -> 32
            # ============================================================
            spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False, indice_key="down1"), 
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            SparseResBlock(32, 32, indice_key="res_scale2a"),
            
            # ============================================================
            # SCALA 1/4 - CRITICAL ZONE
            # Canali: 32 -> 64 (Era 192 -> Qui risparmiamo 3.5M parametri)
            # Manteniamo i 2 ResBlock per la capacità semantica
            # ============================================================
            spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False, indice_key="down2"),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            SparseResBlock(64, 64, indice_key="res_scale4a"),
            SparseResBlock(64, 64, indice_key="res_scale4b"),

            # ============================================================
            # SCALA 1/8 - Compressione Latente
            # 64 -> 64 -> latent_dim
            # ============================================================
            
            # Step 1: Downsample 
            spconv.SparseConv3d(64, latent_dim, 3, stride=2, padding=1, bias=False, indice_key="down3"),
            nn.BatchNorm1d(latent_dim), 
            nn.ReLU(),
            
            # Step 2: Refinement latente
            SparseResBlock(latent_dim, latent_dim, indice_key="res_scale8a"),
            
            # Step 3: Final 1x1 
            spconv.SubMConv3d(latent_dim, latent_dim, 1, bias=False, indice_key="final_compress"),
            nn.BatchNorm1d(latent_dim), 
            nn.ReLU(),
            
            # Step 4: Final ResBlock
            SparseResBlock(latent_dim, latent_dim, indice_key="res_latent")
        )

    def _voxelize_batch(self, points_list):
        """Voxelization - INVARIATO"""
        batch_voxels = []
        batch_coords = []

        for b_idx, points in enumerate(points_list):
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).float().cuda()
            else:
                points = points.cuda()

            # A. Filtro Range
            mask = (points[:, 0] >= self.pc_range[0]) & (points[:, 0] < self.pc_range[3]) & \
                   (points[:, 1] >= self.pc_range[1]) & (points[:, 1] < self.pc_range[4]) & \
                   (points[:, 2] >= self.pc_range[2]) & (points[:, 2] < self.pc_range[5])
            points = points[mask]
            
            if len(points) == 0: 
                continue

            # B. Calcolo Indici Voxel
            coords = ((points[:, :3] - self.pc_range[:3]) / self.voxel_size).long()
            
            # C. Check Limiti Griglia
            mask_grid = (coords[:, 0] >= 0) & (coords[:, 0] < self.grid_size_tensor[0]) & \
                        (coords[:, 1] >= 0) & (coords[:, 1] < self.grid_size_tensor[1]) & \
                        (coords[:, 2] >= 0) & (coords[:, 2] < self.grid_size_tensor[2])
            points = points[mask_grid]
            coords = coords[mask_grid]
            if len(points) == 0: 
                continue

            # D. Riordino [X, Y, Z] -> [Z, Y, X]
            coords = coords[:, [2, 1, 0]]

            # E. Mean Voxelization + Density
            unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
            n_voxels = unique_coords.shape[0]
            
            dim_feat = 3
            
            voxel_features = torch.zeros((n_voxels, dim_feat), device=points.device)
            voxel_counts = torch.zeros((n_voxels, 1), device=points.device)
            
            voxel_features.index_add_(0, inverse_indices, points[:, :3])
            voxel_counts.index_add_(0, inverse_indices, torch.ones((points.shape[0], 1), device=points.device))
            
            mean_features = voxel_features / voxel_counts.clamp(min=1.0)
            density_feature = torch.log(voxel_counts.clamp(min=1.0))
            final_features = torch.cat([mean_features, density_feature], dim=1)

            batch_idx_col = torch.full((n_voxels, 1), b_idx, device=points.device)
            coords_with_batch = torch.cat([batch_idx_col, unique_coords], dim=1)

            batch_voxels.append(final_features)
            batch_coords.append(coords_with_batch)

        if not batch_voxels:
            return None, None
            
        return torch.cat(batch_voxels, dim=0), torch.cat(batch_coords, dim=0)

    def forward(self, points_list):
        """Forward pass - INVARIATO"""
        batch_size = len(points_list)
        
        features, coords = self._voxelize_batch(points_list)
        
        if features is None or features.shape[0] == 0:
            return torch.zeros(
                batch_size, config.LATENT_DIM, 
                self.grid_size_zyx[0]//8, self.grid_size_zyx[1]//8, self.grid_size_zyx[2]//8
            ).cuda()

        features = self.input_mlp(features)

        input_sp = spconv.SparseConvTensor(
            features, 
            coords.int(), 
            self.grid_size_zyx, 
            batch_size
        )

        try:
            encoded_sp = self.backbone(input_sp)
        except RuntimeError as e:
            print(f"⚠️ spconv runtime error: {e}. Returning zeros.")
            return torch.zeros(
                batch_size, config.LATENT_DIM, 
                self.grid_size_zyx[0]//8, self.grid_size_zyx[1]//8, self.grid_size_zyx[2]//8
            ).cuda()

        return encoded_sp.dense()