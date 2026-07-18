"""OpenCV SIFT geometric baseline for pairwise-benchmark-v2."""

from typing import Any

from .config import SiftGeometricConfig

__all__ = ["SiftGeometricAdapter", "SiftGeometricConfig"]


def __getattr__(name: str) -> Any:
    """Preserve the public adapter import without eagerly loading the detector."""

    if name == "SiftGeometricAdapter":
        from .adapter import SiftGeometricAdapter

        return SiftGeometricAdapter
    raise AttributeError(name)
