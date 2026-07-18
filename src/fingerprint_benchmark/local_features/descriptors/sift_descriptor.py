"""SIFT descriptor computation at externally supplied canonical keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .rootsift import process_descriptors


@dataclass(frozen=True)
class SiftDescriptorResult:
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray
    diagnostics: dict[str, object]


def compute_sift_descriptors(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint] | tuple[cv2.KeyPoint, ...],
    *,
    descriptor: str,
) -> SiftDescriptorResult:
    """Compute a common descriptor without invoking the SIFT detector."""

    if descriptor not in ("sift", "standard", "rootsift"):
        raise ValueError(f"Unsupported descriptor: {descriptor!r}.")
    source = np.asarray(image)
    if source.ndim != 2 or source.dtype != np.uint8:
        raise ValueError("SIFT descriptor computation requires a uint8 grayscale image.")
    requested = list(keypoints)
    if not requested:
        return SiftDescriptorResult(
            keypoints=(),
            descriptors=np.empty((0, 128), dtype=np.float32),
            diagnostics={"requested_keypoints": 0, "computed_descriptors": 0},
        )
    sift = cv2.SIFT_create(nfeatures=max(1, len(requested)))
    computed, raw = sift.compute(source, requested)
    computed = computed or []
    if raw is None or not computed:
        descriptors = np.empty((0, 128), dtype=np.float32)
        computed = []
    else:
        mode = "rootsift" if descriptor == "rootsift" else "standard"
        descriptors = process_descriptors(raw, mode)
    if len(computed) != int(descriptors.shape[0]):
        raise ValueError("Computed keypoint and descriptor counts differ.")
    return SiftDescriptorResult(
        keypoints=tuple(computed),
        descriptors=descriptors,
        diagnostics={
            "descriptor": descriptor,
            "descriptor_engine": "cv2.SIFT.compute_at_supplied_keypoints",
            "sift_detector_invoked": False,
            "requested_keypoints": len(requested),
            "computed_descriptors": int(descriptors.shape[0]),
        },
    )


__all__ = ["SiftDescriptorResult", "compute_sift_descriptors"]
