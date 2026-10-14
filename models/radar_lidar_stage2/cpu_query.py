"""CPU spatial-query concurrency for Stage-II candidate sites."""

from __future__ import annotations

import os


def query_workers(count: int) -> int:
    """Use one worker for small queries; allow a per-process override for large ones."""
    if count < 4096:
        return 1
    requested = os.environ.get("STAGE2_KDTREE_WORKERS")
    workers = min(4, os.cpu_count() or 1) if requested is None else int(requested)
    if workers < 1:
        raise ValueError("STAGE2_KDTREE_WORKERS must be a positive integer")
    return workers
