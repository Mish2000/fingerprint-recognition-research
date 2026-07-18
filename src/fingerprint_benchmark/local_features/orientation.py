"""Common deterministic dominant-gradient orientation assignment."""

from __future__ import annotations

import math

import cv2
import numpy as np


ORIENTATION_POLICY = "sift_dominant_gradient"


def assign_orientations(
    image: np.ndarray,
    points: np.ndarray,
    support_sizes: np.ndarray,
    *,
    policy: str = ORIENTATION_POLICY,
    bins: int = 36,
) -> tuple[np.ndarray, dict[str, object]]:
    """Assign one common angle per location without using detector-native angles."""

    if policy not in (ORIENTATION_POLICY, "sift_dominant_gradient_v1"):
        raise ValueError(f"Unsupported orientation policy: {policy!r}.")
    source = np.asarray(image)
    locations = np.asarray(points, dtype=np.float32)
    sizes = np.asarray(support_sizes, dtype=np.float32)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("Orientation assignment requires a non-empty grayscale image.")
    if locations.ndim != 2 or locations.shape[1:] != (2,):
        raise ValueError("points must have shape (N, 2).")
    if sizes.shape != (locations.shape[0],) or np.any(sizes <= 0.0):
        raise ValueError("support_sizes must contain one positive value per point.")
    if bins != 36:
        raise ValueError("detector_only_v1 uses exactly 36 orientation bins.")

    float_image = np.ascontiguousarray(source, dtype=np.float32)
    gradient_x = cv2.Sobel(
        float_image, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT
    )
    gradient_y = cv2.Sobel(
        float_image, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT
    )
    angles = np.asarray(
        [
            _dominant_angle(gradient_x, gradient_y, float(x), float(y), float(size), bins)
            for (x, y), size in zip(locations, sizes, strict=True)
        ],
        dtype=np.float32,
    )
    return angles, {
        "orientation_policy": ORIENTATION_POLICY,
        "orientation_count": int(angles.size),
        "orientation_bins": int(bins),
        "gradient_operator": "opencv_sobel_3x3",
        "gradient_border": "BORDER_REFLECT",
        "detector_native_angles_used": False,
    }


def _dominant_angle(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    x: float,
    y: float,
    support_size: float,
    bins: int,
) -> float:
    height, width = gradient_x.shape
    sigma = max(float(support_size) / 2.0, np.finfo(np.float32).eps)
    radius = max(1, int(math.floor(3.0 * 1.5 * sigma + 0.5)))
    left = max(0, int(math.ceil(x - radius)))
    right = min(width - 1, int(math.floor(x + radius)))
    top = max(0, int(math.ceil(y - radius)))
    bottom = min(height - 1, int(math.floor(y + radius)))
    if left > right or top > bottom:
        return 0.0

    yy, xx = np.mgrid[top : bottom + 1, left : right + 1]
    dx = xx.astype(np.float64) - x
    dy = yy.astype(np.float64) - y
    inside = dx * dx + dy * dy <= radius * radius
    local_x = np.asarray(gradient_x[top : bottom + 1, left : right + 1], dtype=np.float64)
    local_y = np.asarray(gradient_y[top : bottom + 1, left : right + 1], dtype=np.float64)
    magnitude = np.hypot(local_x, local_y)
    valid = inside & np.isfinite(magnitude) & (magnitude > 0.0)
    if not np.any(valid):
        return 0.0
    weight_sigma = 1.5 * sigma
    weights = np.exp(-(dx * dx + dy * dy) / (2.0 * weight_sigma * weight_sigma))
    degrees = np.mod(np.degrees(np.arctan2(local_y, local_x)), 360.0)
    indexes = np.floor(degrees * bins / 360.0).astype(np.int32) % bins
    histogram = np.bincount(
        indexes[valid],
        weights=(weights * magnitude)[valid],
        minlength=bins,
    )
    peak = int(np.argmax(histogram))
    return float((peak * 360.0 / bins) % 360.0)


__all__ = ["ORIENTATION_POLICY", "assign_orientations"]
