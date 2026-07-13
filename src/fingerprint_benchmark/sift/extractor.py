"""OpenCV SIFT extraction and opaque representation construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.contract import MethodExecutionError

from .config import SiftGeometricConfig
from .descriptors import process_descriptors
from .preprocessing import prepare_image


@dataclass(frozen=True)
class SiftRepresentation:
    points: np.ndarray
    sizes: np.ndarray
    angles: np.ndarray
    responses: np.ndarray
    octaves: np.ndarray
    class_ids: np.ndarray
    descriptors: np.ndarray
    width: int
    height: int
    ppi: float
    metadata: dict[str, Any]

    @property
    def keypoint_count(self) -> int:
        return int(self.points.shape[0])


def extract_representation(
    image_path: Path,
    image_metadata: Mapping[str, Any],
    config: SiftGeometricConfig,
) -> tuple[SiftRepresentation, dict[str, Any], float]:
    started = perf_counter_ns()
    load_started = perf_counter_ns()
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image_load_ms = _elapsed(load_started)
    if gray is None:
        raise MethodExecutionError(
            "image_read_failure",
            f"OpenCV could not read grayscale image: {image_path}",
            method_internal_ms=_elapsed(started),
            diagnostics={"image_load_ms": image_load_ms, "path": str(image_path)},
        )
    try:
        ppi = float(image_metadata["ppi"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MethodExecutionError(
            "missing_or_invalid_ppi",
            "SIFT preparation requires finite positive manifest PPI.",
            method_internal_ms=_elapsed(started),
            diagnostics={"image_load_ms": image_load_ms},
        ) from exc
    if not np.isfinite(ppi) or ppi <= 0:
        raise MethodExecutionError(
            "missing_or_invalid_ppi",
            "SIFT preparation requires finite positive manifest PPI.",
            method_internal_ms=_elapsed(started),
            diagnostics={"image_load_ms": image_load_ms, "ppi": ppi},
        )

    prep_started = perf_counter_ns()
    try:
        prepared = prepare_image(gray, ppi, config)
    except (ValueError, cv2.error) as exc:
        raise MethodExecutionError(
            "image_preparation_failure",
            str(exc),
            method_internal_ms=_elapsed(started),
            diagnostics={"image_load_ms": image_load_ms},
        ) from exc
    image_preparation_ms = _elapsed(prep_started)

    sift = cv2.SIFT_create(
        nfeatures=int(config.nfeatures),
        nOctaveLayers=int(config.n_octave_layers),
        contrastThreshold=float(config.contrast_threshold),
        edgeThreshold=float(config.edge_threshold),
        sigma=float(config.sigma),
    )
    extraction_started = perf_counter_ns()
    raw_count: int | None = None
    if prepared.mask is None:
        keypoints, raw_descriptors = sift.detectAndCompute(prepared.image, None)
    else:
        raw_keypoints = sift.detect(prepared.image, None)
        raw_count = len(raw_keypoints or [])
        filtered = [kp for kp in (raw_keypoints or []) if _inside_mask(kp, prepared.mask)]
        keypoints, raw_descriptors = sift.compute(prepared.image, filtered)
    sift_extraction_ms = _elapsed(extraction_started)
    keypoints = keypoints or []
    if raw_descriptors is None or not keypoints:
        raise MethodExecutionError(
            "missing_descriptors",
            "OpenCV SIFT produced no descriptors.",
            method_internal_ms=_elapsed(started),
            diagnostics={
                "image_load_ms": image_load_ms,
                "image_preparation_ms": image_preparation_ms,
                "sift_extraction_ms": sift_extraction_ms,
                "keypoint_count": len(keypoints),
                **prepared.metadata,
            },
        )
    if len(raw_descriptors) < int(config.minimum_descriptors):
        raise MethodExecutionError(
            "too_few_descriptors",
            f"SIFT produced {len(raw_descriptors)} descriptors; at least {config.minimum_descriptors} are required.",
            method_internal_ms=_elapsed(started),
            diagnostics={"keypoint_count": len(keypoints), **prepared.metadata},
        )

    descriptor_started = perf_counter_ns()
    try:
        descriptors = process_descriptors(raw_descriptors, config.descriptor_mode)
    except ValueError as exc:
        raise MethodExecutionError(
            "invalid_descriptors",
            str(exc),
            method_internal_ms=_elapsed(started),
            diagnostics={"keypoint_count": len(keypoints), **prepared.metadata},
        ) from exc
    descriptor_processing_ms = _elapsed(descriptor_started)
    if descriptors.shape[0] != len(keypoints):
        raise MethodExecutionError(
            "descriptor_keypoint_mismatch",
            "SIFT keypoint and descriptor counts differ.",
            method_internal_ms=_elapsed(started),
            diagnostics={"keypoint_count": len(keypoints), "descriptor_count": len(descriptors)},
        )

    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    representation = SiftRepresentation(
        points=points,
        sizes=np.asarray([kp.size for kp in keypoints], dtype=np.float32),
        angles=np.asarray([kp.angle for kp in keypoints], dtype=np.float32),
        responses=np.asarray([kp.response for kp in keypoints], dtype=np.float32),
        octaves=np.asarray([kp.octave for kp in keypoints], dtype=np.int32),
        class_ids=np.asarray([kp.class_id for kp in keypoints], dtype=np.int32),
        descriptors=descriptors,
        width=int(prepared.image.shape[1]),
        height=int(prepared.image.shape[0]),
        ppi=ppi,
        metadata={**prepared.metadata, "source_path": str(image_path)},
    )
    diagnostics = {
        **prepared.metadata,
        "ppi": ppi,
        "keypoint_count_before_mask": raw_count,
        "keypoint_count": representation.keypoint_count,
        "descriptor_count": int(descriptors.shape[0]),
        "descriptor_dimensions": int(descriptors.shape[1]),
        "descriptor_dtype": str(descriptors.dtype),
        "descriptor_processing_mode": config.descriptor_mode,
        "image_load_ms": image_load_ms,
        "image_preparation_ms": image_preparation_ms,
        "sift_extraction_ms": sift_extraction_ms,
        "descriptor_processing_ms": descriptor_processing_ms,
    }
    return representation, diagnostics, _elapsed(started)


def _inside_mask(keypoint: cv2.KeyPoint, mask: np.ndarray) -> bool:
    x = min(max(int(round(keypoint.pt[0])), 0), mask.shape[1] - 1)
    y = min(max(int(round(keypoint.pt[1])), 0), mask.shape[0] - 1)
    return bool(mask[y, x] > 0)


def _elapsed(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0
