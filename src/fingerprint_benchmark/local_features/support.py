"""Physical, PPI-aware support assignment shared by all detectors."""

from __future__ import annotations

import math

import numpy as np


def support_size_pixels(
    ppi: float,
    *,
    reference_ppi: float,
    support_size_reference_px: float,
) -> float:
    """Convert one reference-pixel diameter to native pixels at image PPI."""

    values = (float(ppi), float(reference_ppi), float(support_size_reference_px))
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("PPI and support-size values must be finite and positive.")
    return float(support_size_reference_px) * float(ppi) / float(reference_ppi)


def assign_support_sizes(
    count: int,
    ppi: float,
    *,
    reference_ppi: float,
    support_size_reference_px: float,
) -> np.ndarray:
    """Return the same physical support policy for every ranked location."""

    if count < 0:
        raise ValueError("Feature count must be non-negative.")
    size = support_size_pixels(
        ppi,
        reference_ppi=reference_ppi,
        support_size_reference_px=support_size_reference_px,
    )
    return np.full(int(count), size, dtype=np.float32)


__all__ = ["assign_support_sizes", "support_size_pixels"]
