"""Grayscale image preparation and deterministic non-learned valid regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SiftGeometricConfig


@dataclass(frozen=True)
class ImagePreparation:
    image: np.ndarray
    mask: np.ndarray | None
    metadata: dict[str, object]


def prepare_image(gray: np.ndarray, ppi: float, config: SiftGeometricConfig) -> ImagePreparation:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"Expected uint8 grayscale image, got shape={gray.shape}, dtype={gray.dtype}.")
    original_height, original_width = gray.shape
    if config.image_policy == "native":
        image = np.ascontiguousarray(gray)
        scale = 1.0
        padding = {"left": 0, "top": 0, "right": 0, "bottom": 0}
        enhancement = "none"
    elif config.image_policy == "reference_reproduction":
        clahe = cv2.createCLAHE(
            clipLimit=float(config.reference_clahe_clip),
            tileGridSize=(int(config.reference_clahe_grid_x), int(config.reference_clahe_grid_y)),
        )
        enhanced = clahe.apply(gray)
        image, scale, padding = resize_pad_to_square(enhanced, int(config.reference_target_size))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhancement = "clahe_then_resize_pad_then_minmax"
    else:
        raise ValueError(f"Unsupported image policy: {config.image_policy!r}.")

    mask = None
    mask_metadata: dict[str, object] = {
        "mask_mode": config.mask_mode,
        "mask_coverage_ratio": 1.0,
        "mask_status": "disabled",
    }
    if config.mask_mode == "valid_region":
        mask, mask_metadata = valid_region_mask(image, ppi, config)
    return ImagePreparation(
        image=image,
        mask=mask,
        metadata={
            "image_policy": config.image_policy,
            "enhancement": enhancement,
            "original_width": int(original_width),
            "original_height": int(original_height),
            "prepared_width": int(image.shape[1]),
            "prepared_height": int(image.shape[0]),
            "resize_scale": float(scale),
            "padding": padding,
            **mask_metadata,
        },
    )


def resize_pad_to_square(image: np.ndarray, target: int) -> tuple[np.ndarray, float, dict[str, int]]:
    height, width = image.shape
    scale = float(target) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((target, target), dtype=np.uint8)
    top = (target - new_height) // 2
    left = (target - new_width) // 2
    output[top : top + new_height, left : left + new_width] = resized
    return output, scale, {
        "left": int(left),
        "top": int(top),
        "right": int(target - new_width - left),
        "bottom": int(target - new_height - top),
    }


def valid_region_mask(
    image: np.ndarray,
    ppi: float,
    config: SiftGeometricConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    scale = float(ppi) / float(config.reference_ppi)
    close_size = _odd(max(3, int(round(config.valid_region_close_kernel_at_reference_ppi * scale))))
    erosion = max(0, int(round(config.valid_region_erode_at_reference_ppi * scale)))
    binary = (image > int(config.valid_region_black_threshold)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if components <= 1:
        return np.zeros_like(image), _mask_meta(0.0, "no_connected_component", close_size, erosion)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8) * 255
    if erosion > 0:
        erode_size = _odd(2 * erosion + 1)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        mask = cv2.erode(mask, erode_kernel, iterations=1)
    coverage = float(np.mean(mask > 0))
    status = "ok" if coverage >= float(config.valid_region_min_coverage) else "coverage_too_small"
    if status != "ok":
        mask = np.zeros_like(image)
    return mask, _mask_meta(coverage, status, close_size, erosion)


def _mask_meta(coverage: float, status: str, close_size: int, erosion: int) -> dict[str, object]:
    return {
        "mask_mode": "valid_region",
        "mask_coverage_ratio": float(coverage),
        "mask_status": status,
        "mask_close_kernel": int(close_size),
        "mask_erosion_radius": int(erosion),
    }


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1
