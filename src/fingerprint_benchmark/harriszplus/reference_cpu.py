"""Readable float32 CPU reference for the clean-room HarrisZ+ detector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np

from .config import HarrisZPlusConfig
from .kernels import (
    central_difference_numpy,
    gaussian_blur_numpy,
    sample_zscore_numpy,
    validate_grayscale_float32,
)
from .selection import (
    greedy_distance_suppression,
    iterative_uniform_selection_with_diagnostics,
    passes_eigen_axis_ratio,
    rank_candidates,
    refine_subpixel_numpy,
    remove_scale_01_duplicates,
    scale_suppression_distance,
    strict_local_maxima_numpy,
)
from .types import DetectedKeypoint, DetectorResult, SelectionCandidate, make_scale_spec


@dataclass(frozen=True, slots=True)
class _WorkingCandidate:
    x: float
    y: float
    response: float
    scale_index: int
    source_index: int
    integer_x: int
    integer_y: int


def compute_harrisz_scale_cpu(
    working_image: np.ndarray,
    scale_index: int,
    config: HarrisZPlusConfig | None = None,
    *,
    base_gradients: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute one dense HarrisZ scale and its strict candidate masks.

    ``working_image`` must already have the correct resolution: indexes 0 and
    1 receive the caller-supplied 2x Lanczos array.  No resize is hidden in the
    detector core.
    """

    active_config = config or HarrisZPlusConfig(backend="reference_cpu")
    active_config._validate_scale_index(scale_index)
    source = validate_grayscale_float32(working_image, name="working_image")
    if base_gradients is None:
        base_gradient_x, base_gradient_y = central_difference_numpy(source)
    else:
        base_gradient_x, base_gradient_y = base_gradients
        if base_gradient_x.shape != source.shape or base_gradient_y.shape != source.shape:
            raise ValueError("base_gradients must match working_image shape.")
        if base_gradient_x.dtype != np.float32 or base_gradient_y.dtype != np.float32:
            raise TypeError("base_gradients must be float32.")

    differentiation_sigma = active_config.working_sigma(scale_index)
    integration_sigma = active_config.working_integration_sigma(scale_index)
    truncate = active_config.gaussian_truncate

    scale_gradient_x = gaussian_blur_numpy(
        base_gradient_x,
        differentiation_sigma,
        truncate=truncate,
    )
    scale_gradient_y = gaussian_blur_numpy(
        base_gradient_y,
        differentiation_sigma,
        truncate=truncate,
    )
    magnitude_squared = scale_gradient_x * scale_gradient_x + scale_gradient_y * scale_gradient_y
    magnitude = np.sqrt(magnitude_squared, dtype=np.float32)
    magnitude_mean = np.mean(magnitude, dtype=np.float32)
    raw_edge_threshold = np.float32(
        active_config.raw_edge_mean_multiplier * float(magnitude_mean)
    )
    raw_edge_mask = magnitude > raw_edge_threshold
    edge_mask = gaussian_blur_numpy(
        raw_edge_mask.astype(np.float32),
        differentiation_sigma,
        truncate=truncate,
    )

    enhanced_x = scale_gradient_x * edge_mask
    enhanced_y = scale_gradient_y * edge_mask
    autocorrelation_xx = gaussian_blur_numpy(
        enhanced_x * enhanced_x,
        integration_sigma,
        truncate=truncate,
    )
    autocorrelation_xy = gaussian_blur_numpy(
        enhanced_x * enhanced_y,
        integration_sigma,
        truncate=truncate,
    )
    autocorrelation_yy = gaussian_blur_numpy(
        enhanced_y * enhanced_y,
        integration_sigma,
        truncate=truncate,
    )
    determinant = autocorrelation_xx * autocorrelation_yy - autocorrelation_xy * autocorrelation_xy
    trace = autocorrelation_xx + autocorrelation_yy
    trace_squared = trace * trace
    response = sample_zscore_numpy(determinant) - sample_zscore_numpy(trace_squared)
    response = response.astype(np.float32, copy=False)
    nonfinite_response_count = int(
        response.size - np.count_nonzero(np.isfinite(response))
    )
    if nonfinite_response_count:
        raise FloatingPointError(
            f"HarrisZ+ scale {scale_index} produced {nonfinite_response_count} "
            "non-finite dense responses."
        )

    response_positive_mask = response > active_config.response_threshold
    candidate_mask = response_positive_mask & (edge_mask > active_config.edge_mask_threshold)
    local_maxima_mask = strict_local_maxima_numpy(
        response,
        candidate_mask,
        tie_atol=active_config.local_maximum_tie_atol,
    )
    return {
        "scale_index": scale_index,
        "working_image_scale": active_config.working_image_scale(scale_index),
        "differentiation_sigma": differentiation_sigma,
        "integration_sigma": integration_sigma,
        "base_gradient_x": base_gradient_x,
        "base_gradient_y": base_gradient_y,
        "scale_gradient_x": scale_gradient_x,
        "scale_gradient_y": scale_gradient_y,
        "gradient_magnitude": magnitude,
        "gradient_magnitude_mean": float(magnitude_mean),
        "raw_edge_threshold": float(raw_edge_threshold),
        "raw_edge_mask": raw_edge_mask,
        "edge_mask": edge_mask,
        "autocorrelation_xx": autocorrelation_xx,
        "autocorrelation_xy": autocorrelation_xy,
        "autocorrelation_yy": autocorrelation_yy,
        "determinant": determinant,
        "trace_squared": trace_squared,
        "response": response,
        "nonfinite_response_count": nonfinite_response_count,
        "response_positive_mask": response_positive_mask,
        "candidate_mask": candidate_mask,
        "local_maxima_mask": local_maxima_mask,
    }


def _validate_detector_inputs(
    image: np.ndarray,
    doubled_image: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    source = validate_grayscale_float32(image, name="image")
    if doubled_image is None:
        raise ValueError(
            "doubled_image is required: the extractor must supply one deterministic "
            "OpenCV INTER_LANCZOS4 2x array for both detector backends."
        )
    doubled = validate_grayscale_float32(doubled_image, name="doubled_image")
    expected_shape = (source.shape[0] * 2, source.shape[1] * 2)
    if doubled.shape != expected_shape:
        raise ValueError(
            f"doubled_image must have shape {expected_shape}; got {doubled.shape}."
        )
    return source, doubled


def _working_candidates(
    maps: Mapping[str, Any],
    scale_index: int,
    source_offset: int,
) -> list[_WorkingCandidate]:
    locations_y, locations_x = np.nonzero(maps["local_maxima_mask"])
    response = maps["response"]
    return [
        _WorkingCandidate(
            x=float(x),
            y=float(y),
            response=float(response[y, x]),
            scale_index=scale_index,
            source_index=source_offset + offset,
            integer_x=int(x),
            integer_y=int(y),
        )
        for offset, (y, x) in enumerate(zip(locations_y.tolist(), locations_x.tolist()))
    ]


def _refine_and_filter_cpu(
    candidates: tuple[_WorkingCandidate, ...],
    maps: Mapping[str, Any],
    config: HarrisZPlusConfig,
    scale_index: int,
) -> tuple[SelectionCandidate, ...]:
    response = maps["response"]
    autocorrelation_xx = maps["autocorrelation_xx"]
    autocorrelation_xy = maps["autocorrelation_xy"]
    autocorrelation_yy = maps["autocorrelation_yy"]
    coordinate_divisor = config.working_image_scale(scale_index)
    output: list[SelectionCandidate] = []
    for candidate in candidates:
        x = candidate.integer_x
        y = candidate.integer_y
        if not passes_eigen_axis_ratio(
            autocorrelation_xx[y, x],
            autocorrelation_xy[y, x],
            autocorrelation_yy[y, x],
            threshold=config.eigen_axis_ratio_threshold,
        ):
            continue
        refined_x, refined_y = refine_subpixel_numpy(response, x, y)
        output.append(
            SelectionCandidate(
                x=refined_x / coordinate_divisor,
                y=refined_y / coordinate_divisor,
                response=candidate.response,
                scale_index=scale_index,
                source_index=candidate.source_index,
            )
        )
    return tuple(output)


def _to_detected_keypoint(
    candidate: SelectionCandidate,
    config: HarrisZPlusConfig,
) -> DetectedKeypoint:
    scale_index = candidate.scale_index
    return DetectedKeypoint(
        x=float(candidate.x),
        y=float(candidate.y),
        response=float(candidate.response),
        scale_index=scale_index,
        sigma=config.output_sigma(scale_index),
        integration_sigma=config.output_integration_sigma(scale_index),
        effective_support_diameter=config.effective_gaussian_support_diameter(scale_index),
        size=config.keypoint_size(scale_index),
        source_index=candidate.source_index,
    )


def detect_harriszplus_cpu(
    image: np.ndarray,
    config: HarrisZPlusConfig | None = None,
    *,
    doubled_image: np.ndarray | None = None,
    return_response_maps: bool = False,
) -> DetectorResult:
    """Detect HarrisZ+ keypoints with the readable CPU reference backend."""

    active_config = config or HarrisZPlusConfig(backend="reference_cpu")
    total_started = perf_counter()
    validation_started = perf_counter()
    source, doubled = _validate_detector_inputs(image, doubled_image)
    cv2.setNumThreads(active_config.opencv_threads)
    cv2.setUseOptimized(active_config.opencv_optimized)
    validation_seconds = perf_counter() - validation_started

    gradient_started = perf_counter()
    gradients_by_image_scale = {
        1.0: central_difference_numpy(source),
        2.0: central_difference_numpy(doubled),
    }
    gradient_seconds = perf_counter() - gradient_started

    scale_diagnostics: dict[str, Any] = {}
    response_maps: dict[int, np.ndarray] | None = {} if return_response_maps else None
    selected_across_scales: list[SelectionCandidate] = []
    source_offset = 0
    dense_response_seconds = 0.0
    local_maximum_seconds = 0.0
    scale_suppression_seconds = 0.0
    refinement_seconds = 0.0
    counts = {
        "dense_pixels": 0,
        "candidates_before_mask": 0,
        "candidates_after_mask": 0,
        "candidates_after_local_maxima": 0,
        "candidates_after_scale_suppression": 0,
        "candidates_after_eigen_ratio": 0,
    }
    invalid_or_nan_outputs = 0

    for scale_index in active_config.scale_indices:
        working_scale = active_config.working_image_scale(scale_index)
        working_image = doubled if working_scale == 2.0 else source
        dense_started = perf_counter()
        maps = compute_harrisz_scale_cpu(
            working_image,
            scale_index,
            active_config,
            base_gradients=gradients_by_image_scale[working_scale],
        )
        dense_elapsed = perf_counter() - dense_started
        dense_response_seconds += dense_elapsed

        # Local-max computation is part of compute_harrisz_scale_cpu.  Record
        # it with the dense stage while preserving a stable timing key.
        local_started = perf_counter()
        candidates = _working_candidates(maps, scale_index, source_offset)
        local_elapsed = perf_counter() - local_started
        local_maximum_seconds += local_elapsed
        source_offset += len(candidates)

        suppression_started = perf_counter()
        suppression_distance = scale_suppression_distance(active_config.working_sigma(scale_index))
        scale_selected, _ = greedy_distance_suppression(candidates, suppression_distance)
        scale_suppression_seconds += perf_counter() - suppression_started

        refinement_started = perf_counter()
        refined = _refine_and_filter_cpu(scale_selected, maps, active_config, scale_index)
        refinement_seconds += perf_counter() - refinement_started
        selected_across_scales.extend(refined)
        invalid_or_nan_outputs += int(maps["nonfinite_response_count"])

        scale_counts = {
            "dense_pixels": int(maps["response"].size),
            "candidates_before_mask": int(np.count_nonzero(maps["response_positive_mask"])),
            "candidates_after_mask": int(np.count_nonzero(maps["candidate_mask"])),
            "candidates_after_local_maxima": len(candidates),
            "candidates_after_scale_suppression": len(scale_selected),
            "candidates_after_eigen_ratio": len(refined),
        }
        for key in counts:
            counts[key] += scale_counts[key]
        scale_diagnostics[str(scale_index)] = {
            "scale": make_scale_spec(active_config, scale_index).as_dict(),
            "suppression_distance_working_px": suppression_distance,
            "counts": scale_counts,
            "nonfinite_response_count": int(maps["nonfinite_response_count"]),
            "timing_seconds": dense_elapsed + local_elapsed,
        }
        if response_maps is not None:
            response_maps[scale_index] = np.array(maps["response"], dtype=np.float32, copy=True)

    duplicate_started = perf_counter()
    deduplicated = remove_scale_01_duplicates(
        selected_across_scales,
        distance=active_config.duplicate_distance,
    )
    duplicate_seconds = perf_counter() - duplicate_started

    uniform_started = perf_counter()
    uniform, uniform_diagnostics = iterative_uniform_selection_with_diagnostics(
        deduplicated,
        source.shape[0],
        source.shape[1],
        maximum_keypoints=active_config.max_keypoints,
    )
    final_ranked = rank_candidates(uniform)
    keypoints = tuple(
        _to_detected_keypoint(candidate, active_config) for candidate in final_ranked
    )
    uniform_seconds = perf_counter() - uniform_started

    total_seconds = perf_counter() - total_started
    timings = {
        "input_validation_seconds": validation_seconds,
        "base_gradients_seconds": gradient_seconds,
        "dense_response_seconds": dense_response_seconds,
        "local_candidate_materialization_seconds": local_maximum_seconds,
        "scale_suppression_seconds": scale_suppression_seconds,
        "subpixel_and_eigen_seconds": refinement_seconds,
        "duplicate_removal_seconds": duplicate_seconds,
        "uniform_selection_seconds": uniform_seconds,
        "total_seconds": total_seconds,
    }
    diagnostics: dict[str, Any] = {
        "backend": "reference_cpu",
        "input_shape": [int(source.shape[0]), int(source.shape[1])],
        "doubled_shape": [int(doubled.shape[0]), int(doubled.shape[1])],
        "input_dtype": str(source.dtype),
        "scale_indices": list(active_config.scale_indices),
        "scales": scale_diagnostics,
        **counts,
        "candidates_after_duplicate_removal": len(deduplicated),
        "candidates_after_uniform_selection": len(uniform),
        "final_keypoint_count": len(keypoints),
        "cap_truncated_count": uniform_diagnostics["cap_truncated_count"],
        "uniform_selection": uniform_diagnostics,
        "invalid_or_nan_outputs": invalid_or_nan_outputs,
    }
    return DetectorResult(
        backend="reference_cpu",
        keypoints=keypoints,
        diagnostics=diagnostics,
        timings=timings,
        response_maps=response_maps,
    )


class HarrisZPlusReferenceCPU:
    """State-light convenience wrapper with a detector-style interface."""

    def __init__(self, config: HarrisZPlusConfig | None = None) -> None:
        self.config = config or HarrisZPlusConfig(backend="reference_cpu")

    def detect(
        self,
        image: np.ndarray,
        *,
        doubled_image: np.ndarray | None = None,
        return_response_maps: bool = False,
    ) -> DetectorResult:
        return detect_harriszplus_cpu(
            image,
            self.config,
            doubled_image=doubled_image,
            return_response_maps=return_response_maps,
        )


HarrisZPlusReferenceDetector = HarrisZPlusReferenceCPU
detect_cpu = detect_harriszplus_cpu
detect = detect_harriszplus_cpu


__all__ = [
    "HarrisZPlusReferenceCPU",
    "HarrisZPlusReferenceDetector",
    "compute_harrisz_scale_cpu",
    "detect_harriszplus_cpu",
    "detect_cpu",
    "detect",
]
