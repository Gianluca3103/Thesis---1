"""Shared online geometry for aligned radar and LiDAR training samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math

import torch


@dataclass(frozen=True)
class HorizontalFlipConfig:
    enabled: bool = True
    probability: float = 0.5


@dataclass(frozen=True)
class TranslationAugmentationConfig:
    enabled: bool = True
    max_x_m: float = 0.5
    max_y_m: float = 0.5


@dataclass(frozen=True)
class YawAugmentationConfig:
    enabled: bool = True
    max_degrees: float = 5.0


@dataclass(frozen=True)
class ScaleAugmentationConfig:
    enabled: bool = True
    min: float = 0.95
    max: float = 1.05


@dataclass(frozen=True)
class GeometricAugmentationConfig:
    enabled: bool = False
    horizontal_flip: HorizontalFlipConfig = field(default_factory=HorizontalFlipConfig)
    translation: TranslationAugmentationConfig = field(default_factory=TranslationAugmentationConfig)
    yaw: YawAugmentationConfig = field(default_factory=YawAugmentationConfig)
    scale: ScaleAugmentationConfig = field(default_factory=ScaleAugmentationConfig)

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("augmentation.enabled must be boolean")
        if not isinstance(self.horizontal_flip.enabled, bool):
            raise ValueError("augmentation.horizontal_flip.enabled must be boolean")
        numeric_values = (
            self.horizontal_flip.probability,
            self.translation.max_x_m,
            self.translation.max_y_m,
            self.yaw.max_degrees,
            self.scale.min,
            self.scale.max,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("augmentation values must be finite")
        if not 0.0 <= self.horizontal_flip.probability <= 1.0:
            raise ValueError("horizontal-flip probability must be in [0,1]")
        if not isinstance(self.translation.enabled, bool):
            raise ValueError("augmentation.translation.enabled must be boolean")
        if self.translation.max_x_m < 0.0 or self.translation.max_y_m < 0.0:
            raise ValueError("translation limits must be non-negative")
        if not isinstance(self.yaw.enabled, bool):
            raise ValueError("augmentation.yaw.enabled must be boolean")
        if self.yaw.max_degrees < 0.0:
            raise ValueError("maximum yaw must be non-negative")
        if not isinstance(self.scale.enabled, bool):
            raise ValueError("augmentation.scale.enabled must be boolean")
        if self.scale.min <= 0.0 or self.scale.max < self.scale.min:
            raise ValueError("scale bounds must satisfy 0 < min <= max")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "GeometricAugmentationConfig":
        payload = {} if payload is None else dict(payload)
        allowed = {field.name for field in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("Unknown augmentation settings: " + ", ".join(sorted(unknown)))

        def nested(name, config_type):
            value = payload.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"augmentation.{name} must be an object")
            valid = {field.name for field in fields(config_type)}
            extra = set(value) - valid
            if extra:
                raise ValueError(f"Unknown augmentation.{name} settings: " + ", ".join(sorted(extra)))
            return config_type(**value)

        config = cls(
            enabled=payload.get("enabled", False),
            horizontal_flip=nested("horizontal_flip", HorizontalFlipConfig),
            translation=nested("translation", TranslationAugmentationConfig),
            yaw=nested("yaw", YawAugmentationConfig),
            scale=nested("scale", ScaleAugmentationConfig),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class GeometricTransform:
    flip_y: bool = False
    scale: float = 1.0
    yaw_radians: float = 0.0
    translation_x_m: float = 0.0
    translation_y_m: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            not self.flip_y
            and self.scale == 1.0
            and self.yaw_radians == 0.0
            and self.translation_x_m == 0.0
            and self.translation_y_m == 0.0
        )

    def to_dict(self) -> dict:
        return {
            "flip_y": self.flip_y,
            "scale": self.scale,
            "yaw_degrees": math.degrees(self.yaw_radians),
            "translation_x_m": self.translation_x_m,
            "translation_y_m": self.translation_y_m,
        }


def _uniform(low: float, high: float, generator=None) -> float:
    if low == high:
        return float(low)
    value = torch.rand((), generator=generator).item()
    return float(low + (high - low) * value)


def sample_transform(config: GeometricAugmentationConfig, *, generator=None) -> GeometricTransform:
    """Use the same parameter distribution as the BEV reconstruction pipeline."""
    if not config.enabled:
        return GeometricTransform()
    flip = config.horizontal_flip.enabled and (
        torch.rand((), generator=generator).item() < config.horizontal_flip.probability
    )
    translation = config.translation
    yaw = config.yaw
    scale = config.scale
    return GeometricTransform(
        flip_y=flip,
        scale=_uniform(scale.min, scale.max, generator) if scale.enabled else 1.0,
        yaw_radians=(math.radians(_uniform(-yaw.max_degrees, yaw.max_degrees, generator))
                     if yaw.enabled else 0.0),
        translation_x_m=(_uniform(-translation.max_x_m, translation.max_x_m, generator)
                         if translation.enabled else 0.0),
        translation_y_m=(_uniform(-translation.max_y_m, translation.max_y_m, generator)
                         if translation.enabled else 0.0),
    )


def forward_xy_matrix(transform: GeometricTransform, dtype, device) -> torch.Tensor:
    cosine = math.cos(transform.yaw_radians)
    sine = math.sin(transform.yaw_radians)
    flip = -1.0 if transform.flip_y else 1.0
    return torch.tensor(
        [[transform.scale * cosine, -transform.scale * sine * flip],
         [transform.scale * sine, transform.scale * cosine * flip]],
        dtype=dtype, device=device,
    )


def transform_points_xy(points: torch.Tensor, transform: GeometricTransform) -> torch.Tensor:
    """Transform XYZ positions in the ground plane; preserve Z and sensor channels."""
    if transform.is_identity:
        return points
    transformed = points.clone()
    matrix = forward_xy_matrix(transform, transformed.dtype, transformed.device)
    translation = transformed.new_tensor([transform.translation_x_m, transform.translation_y_m])
    transformed[:, :2] = transformed[:, :2] @ matrix.T + translation
    return transformed
