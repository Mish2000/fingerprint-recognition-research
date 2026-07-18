"""PPI-aware HarrisZ+ v4 detector with frozen v3 response semantics."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .cuda_detector import (
    _materialize_candidates,
    _refine_and_filter_torch,
    _require_torch,
    _resolve_device,
    _synchronize,
    configure_torch_determinism,
)
from .kernels import (
    central_difference_numpy,
    central_difference_torch,
    sample_zscore_numpy,
    sample_zscore_torch,
    symmetric_pad2d_torch,
    validate_grayscale_float32,
)
from .ppi_aware_v4 import PpiAwareRuntimeConfig
from .reference_cpu import (
    _refine_and_filter_cpu,
    _validate_detector_inputs,
    _working_candidates,
)
from .selection import (
    greedy_distance_suppression,
    iterative_uniform_selection_with_diagnostics,
    rank_candidates,
    remove_scale_01_duplicates,
    strict_local_maxima_numpy,
    strict_local_maxima_torch,
)
from .types import DetectorResult, SelectionCandidate, make_scale_spec


def gaussian_kernel1d_with_radius(sigma: float, radius: int) -> np.ndarray:
    """Return the frozen sampled Gaussian using an explicit PPI-scaled radius."""

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if int(radius) != radius or radius < 1:
        raise ValueError("radius must be a positive integer.")
    coordinates = np.arange(-int(radius), int(radius) + 1, dtype=np.float32)
    sigma32 = np.float32(sigma)
    kernel = np.exp(
        np.float32(-0.5) * (coordinates / sigma32) ** np.float32(2.0)
    ).astype(np.float32, copy=False)
    total = np.sum(kernel, dtype=np.float32)
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Gaussian kernel normalization failed.")
    kernel = (kernel / total).astype(np.float32, copy=False)
    kernel[int(radius)] = np.float32(
        kernel[int(radius)]
        + (np.float32(1.0) - np.sum(kernel, dtype=np.float32))
    )
    return np.ascontiguousarray(kernel)


def gaussian_blur_numpy_with_radius(
    image: np.ndarray,
    sigma: float,
    radius: int,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("Gaussian blur expects a two-dimensional array.")
    kernel = gaussian_kernel1d_with_radius(sigma, radius)
    return np.ascontiguousarray(
        cv2.sepFilter2D(
            source,
            cv2.CV_32F,
            kernel,
            kernel,
            borderType=cv2.BORDER_REFLECT,
        ),
        dtype=np.float32,
    )


def gaussian_blur_torch_with_radius(
    image: Any,
    sigma: float,
    radius: int,
) -> Any:
    torch = _require_torch()
    import torch.nn.functional as functional

    if image.ndim != 2 or image.dtype != torch.float32:
        raise ValueError("Gaussian blur expects a 2-D float32 tensor.")
    kernel = torch.as_tensor(
        gaussian_kernel1d_with_radius(sigma, radius),
        dtype=torch.float32,
        device=image.device,
    )
    horizontal = symmetric_pad2d_torch(image, 0, int(radius))
    horizontal = functional.conv2d(
        horizontal[None, None],
        kernel.reshape(1, 1, 1, -1),
    )[0, 0]
    vertical = symmetric_pad2d_torch(horizontal, int(radius), 0)
    return functional.conv2d(
        vertical[None, None],
        kernel.reshape(1, 1, -1, 1),
    )[0, 0]


def compute_harrisz_scale_v4_cpu(
    working_image: np.ndarray,
    scale_index: int,
    config: PpiAwareRuntimeConfig,
    *,
    base_gradients: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute the v3 HarrisZ response with PPI-scaled Gaussian parameters."""

    config._validate_scale_index(scale_index)
    source = validate_grayscale_float32(working_image, name="working_image")
    gradient_x, gradient_y = (
        central_difference_numpy(source)
        if base_gradients is None
        else base_gradients
    )
    differentiation_sigma = config.working_sigma(scale_index)
    integration_sigma = config.working_integration_sigma(scale_index)
    differentiation_radius = config.kernel_radius(scale_index, "differentiation")
    integration_radius = config.kernel_radius(scale_index, "integration")
    blur_d = lambda image: gaussian_blur_numpy_with_radius(
        image, differentiation_sigma, differentiation_radius
    )
    blur_i = lambda image: gaussian_blur_numpy_with_radius(
        image, integration_sigma, integration_radius
    )
    scale_gradient_x = blur_d(gradient_x)
    scale_gradient_y = blur_d(gradient_y)
    magnitude = np.sqrt(
        scale_gradient_x * scale_gradient_x + scale_gradient_y * scale_gradient_y,
        dtype=np.float32,
    )
    magnitude_mean = np.mean(magnitude, dtype=np.float32)
    raw_edge_threshold = np.float32(
        config.raw_edge_mean_multiplier * float(magnitude_mean)
    )
    raw_edge_mask = magnitude > raw_edge_threshold
    edge_mask = blur_d(raw_edge_mask.astype(np.float32))
    enhanced_x = scale_gradient_x * edge_mask
    enhanced_y = scale_gradient_y * edge_mask
    autocorrelation_xx = blur_i(enhanced_x * enhanced_x)
    autocorrelation_xy = blur_i(enhanced_x * enhanced_y)
    autocorrelation_yy = blur_i(enhanced_y * enhanced_y)
    determinant = (
        autocorrelation_xx * autocorrelation_yy
        - autocorrelation_xy * autocorrelation_xy
    )
    trace = autocorrelation_xx + autocorrelation_yy
    trace_squared = trace * trace
    response = (
        sample_zscore_numpy(determinant) - sample_zscore_numpy(trace_squared)
    ).astype(np.float32, copy=False)
    nonfinite = int(response.size - np.count_nonzero(np.isfinite(response)))
    if nonfinite:
        raise FloatingPointError(
            f"HarrisZ+ v4 scale {scale_index} produced {nonfinite} non-finite responses."
        )
    response_positive_mask = response > config.response_threshold
    candidate_mask = response_positive_mask & (
        edge_mask > config.edge_mask_threshold
    )
    local_maxima_mask = strict_local_maxima_numpy(
        response,
        candidate_mask,
        tie_atol=config.local_maximum_tie_atol,
    )
    return {
        "scale_index": scale_index,
        "working_image_scale": config.working_image_scale(scale_index),
        "differentiation_sigma": differentiation_sigma,
        "integration_sigma": integration_sigma,
        "differentiation_kernel_radius": differentiation_radius,
        "integration_kernel_radius": integration_radius,
        "base_gradient_x": gradient_x,
        "base_gradient_y": gradient_y,
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
        "nonfinite_response_count": nonfinite,
        "response_positive_mask": response_positive_mask,
        "candidate_mask": candidate_mask,
        "local_maxima_mask": local_maxima_mask,
    }


def compute_harrisz_scale_v4_torch(
    working_image: Any,
    scale_index: int,
    config: PpiAwareRuntimeConfig,
    *,
    base_gradients: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Torch counterpart of :func:`compute_harrisz_scale_v4_cpu`."""

    torch = _require_torch()
    config._validate_scale_index(scale_index)
    if working_image.ndim != 2 or working_image.dtype != torch.float32:
        raise ValueError("working_image must be a 2-D float32 tensor.")
    gradient_x, gradient_y = (
        central_difference_torch(working_image)
        if base_gradients is None
        else base_gradients
    )
    differentiation_sigma = config.working_sigma(scale_index)
    integration_sigma = config.working_integration_sigma(scale_index)
    differentiation_radius = config.kernel_radius(scale_index, "differentiation")
    integration_radius = config.kernel_radius(scale_index, "integration")
    blur_d = lambda image: gaussian_blur_torch_with_radius(
        image, differentiation_sigma, differentiation_radius
    )
    blur_i = lambda image: gaussian_blur_torch_with_radius(
        image, integration_sigma, integration_radius
    )
    scale_gradient_x = blur_d(gradient_x)
    scale_gradient_y = blur_d(gradient_y)
    magnitude = torch.sqrt(
        scale_gradient_x * scale_gradient_x + scale_gradient_y * scale_gradient_y
    )
    magnitude_mean = torch.mean(magnitude)
    raw_edge_threshold = magnitude_mean * config.raw_edge_mean_multiplier
    raw_edge_mask = magnitude > raw_edge_threshold
    edge_mask = blur_d(raw_edge_mask.to(torch.float32))
    enhanced_x = scale_gradient_x * edge_mask
    enhanced_y = scale_gradient_y * edge_mask
    autocorrelation_xx = blur_i(enhanced_x * enhanced_x)
    autocorrelation_xy = blur_i(enhanced_x * enhanced_y)
    autocorrelation_yy = blur_i(enhanced_y * enhanced_y)
    determinant = (
        autocorrelation_xx * autocorrelation_yy
        - autocorrelation_xy * autocorrelation_xy
    )
    trace = autocorrelation_xx + autocorrelation_yy
    trace_squared = trace * trace
    response = sample_zscore_torch(determinant) - sample_zscore_torch(trace_squared)
    nonfinite = int(torch.count_nonzero(~torch.isfinite(response)).item())
    if nonfinite:
        raise FloatingPointError(
            f"HarrisZ+ v4 scale {scale_index} produced {nonfinite} non-finite responses."
        )
    response_positive_mask = response > config.response_threshold
    candidate_mask = response_positive_mask & (
        edge_mask > config.edge_mask_threshold
    )
    local_maxima_mask = strict_local_maxima_torch(
        response,
        candidate_mask,
        tie_atol=config.local_maximum_tie_atol,
    )
    return {
        "scale_index": scale_index,
        "working_image_scale": config.working_image_scale(scale_index),
        "differentiation_sigma": differentiation_sigma,
        "integration_sigma": integration_sigma,
        "differentiation_kernel_radius": differentiation_radius,
        "integration_kernel_radius": integration_radius,
        "base_gradient_x": gradient_x,
        "base_gradient_y": gradient_y,
        "scale_gradient_x": scale_gradient_x,
        "scale_gradient_y": scale_gradient_y,
        "gradient_magnitude": magnitude,
        "gradient_magnitude_mean": magnitude_mean,
        "raw_edge_threshold": raw_edge_threshold,
        "raw_edge_mask": raw_edge_mask,
        "edge_mask": edge_mask,
        "autocorrelation_xx": autocorrelation_xx,
        "autocorrelation_xy": autocorrelation_xy,
        "autocorrelation_yy": autocorrelation_yy,
        "determinant": determinant,
        "trace_squared": trace_squared,
        "response": response,
        "nonfinite_response_count": nonfinite,
        "response_positive_mask": response_positive_mask,
        "candidate_mask": candidate_mask,
        "local_maxima_mask": local_maxima_mask,
    }


def _inside_descriptor_safe_border(
    candidate: SelectionCandidate,
    *,
    width: int,
    height: int,
    margin: float,
) -> bool:
    return (
        candidate.x >= margin
        and candidate.y >= margin
        and candidate.x <= float(width - 1) - margin
        and candidate.y <= float(height - 1) - margin
    )


def _finish_detection(
    *,
    source: np.ndarray,
    config: PpiAwareRuntimeConfig,
    selected_across_scales: list[SelectionCandidate],
) -> tuple[tuple[Any, ...], tuple[SelectionCandidate, ...], dict[str, Any]]:
    deduplicated = remove_scale_01_duplicates(
        selected_across_scales,
        distance=config.duplicate_distance,
    )
    uniform, uniform_diagnostics = iterative_uniform_selection_with_diagnostics(
        deduplicated,
        source.shape[0],
        source.shape[1],
        maximum_keypoints=config.max_keypoints,
    )
    final_ranked = rank_candidates(uniform)
    keypoints = tuple(
        _to_detected_keypoint(candidate, config) for candidate in final_ranked
    )
    return keypoints, deduplicated, uniform_diagnostics


def _to_detected_keypoint(
    candidate: SelectionCandidate,
    config: PpiAwareRuntimeConfig,
) -> Any:
    from .types import DetectedKeypoint

    index = candidate.scale_index
    return DetectedKeypoint(
        x=float(candidate.x),
        y=float(candidate.y),
        response=float(candidate.response),
        scale_index=index,
        sigma=config.output_sigma(index),
        integration_sigma=config.output_integration_sigma(index),
        effective_support_diameter=config.effective_gaussian_support_diameter(index),
        size=config.keypoint_size(index),
        source_index=candidate.source_index,
    )


def _base_counts() -> dict[str, int]:
    return {
        "dense_pixels": 0,
        "candidates_before_mask": 0,
        "candidates_after_mask": 0,
        "candidates_after_local_maxima": 0,
        "candidates_after_scale_suppression": 0,
        "candidates_after_eigen_ratio": 0,
        "candidates_after_border_exclusion": 0,
    }


def _scale_counts(
    maps: Mapping[str, Any],
    candidates: list[Any],
    suppressed: tuple[Any, ...],
    refined: tuple[SelectionCandidate, ...],
    border_safe: tuple[SelectionCandidate, ...],
    *,
    torch_mode: bool,
) -> dict[str, int]:
    if torch_mode:
        dense_pixels = int(maps["response"].numel())
        before = int(maps["response_positive_mask"].sum().item())
        after = int(maps["candidate_mask"].sum().item())
    else:
        dense_pixels = int(maps["response"].size)
        before = int(np.count_nonzero(maps["response_positive_mask"]))
        after = int(np.count_nonzero(maps["candidate_mask"]))
    return {
        "dense_pixels": dense_pixels,
        "candidates_before_mask": before,
        "candidates_after_mask": after,
        "candidates_after_local_maxima": len(candidates),
        "candidates_after_scale_suppression": len(suppressed),
        "candidates_after_eigen_ratio": len(refined),
        "candidates_after_border_exclusion": len(border_safe),
    }


def _common_diagnostics(
    *,
    backend: str,
    source: np.ndarray,
    doubled: np.ndarray,
    config: PpiAwareRuntimeConfig,
    scale_diagnostics: dict[str, Any],
    counts: dict[str, int],
    keypoints: tuple[Any, ...],
    deduplicated: tuple[SelectionCandidate, ...],
    uniform_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": backend,
        "method_version": config.operational.method_version,
        "manifest_ppi": config.manifest_ppi,
        "reference_ppi": config.reference_ppi,
        "spatial_scale": config.spatial_scale,
        "spatial_scale_formula": "manifest_ppi / 1000.0",
        "input_shape": [int(source.shape[0]), int(source.shape[1])],
        "doubled_shape": [int(doubled.shape[0]), int(doubled.shape[1])],
        "input_dtype": str(source.dtype),
        "native_image_policy": "no_hidden_resize",
        "scale_indices": list(config.scale_indices),
        "scales": scale_diagnostics,
        **counts,
        "duplicate_radius_native_px": config.duplicate_distance,
        "candidates_after_duplicate_removal": len(deduplicated),
        "candidates_after_uniform_selection": len(keypoints),
        "final_keypoint_count": len(keypoints),
        "cap_truncated_count": uniform_diagnostics["cap_truncated_count"],
        "uniform_selection": uniform_diagnostics,
        "uniform_q_policy": "dimension_derived_not_independently_ppi_scaled",
        "invalid_or_nan_outputs": 0,
    }


def detect_harriszplus_v4_cpu(
    image: np.ndarray,
    config: PpiAwareRuntimeConfig,
    *,
    doubled_image: np.ndarray | None,
    return_response_maps: bool = False,
) -> DetectorResult:
    """Run PPI-aware v4 using the readable CPU backend."""

    total_started = perf_counter()
    source, doubled = _validate_detector_inputs(image, doubled_image)
    cv2.setNumThreads(config.opencv_threads)
    cv2.setUseOptimized(config.opencv_optimized)
    gradients = {
        1.0: central_difference_numpy(source),
        2.0: central_difference_numpy(doubled),
    }
    selected: list[SelectionCandidate] = []
    scales: dict[str, Any] = {}
    response_maps: dict[int, np.ndarray] | None = {} if return_response_maps else None
    counts = _base_counts()
    source_offset = 0
    dense_seconds = suppression_seconds = refinement_seconds = 0.0
    for index in config.scale_indices:
        working_scale = config.working_image_scale(index)
        working = doubled if working_scale == 2.0 else source
        started = perf_counter()
        maps = compute_harrisz_scale_v4_cpu(
            working,
            index,
            config,
            base_gradients=gradients[working_scale],
        )
        dense_seconds += perf_counter() - started
        candidates = _working_candidates(maps, index, source_offset)
        source_offset += len(candidates)
        started = perf_counter()
        suppression_distance = config.scale_suppression_distance_working(index)
        suppressed, _ = greedy_distance_suppression(
            candidates, suppression_distance
        )
        suppression_seconds += perf_counter() - started
        started = perf_counter()
        refined = _refine_and_filter_cpu(suppressed, maps, config, index)
        margin = config.border_margin_native(index)
        border_safe = tuple(
            candidate
            for candidate in refined
            if _inside_descriptor_safe_border(
                candidate,
                width=source.shape[1],
                height=source.shape[0],
                margin=margin,
            )
        )
        refinement_seconds += perf_counter() - started
        selected.extend(border_safe)
        row_counts = _scale_counts(
            maps, candidates, suppressed, refined, border_safe, torch_mode=False
        )
        for key, value in row_counts.items():
            counts[key] += value
        scales[str(index)] = {
            "scale": make_scale_spec(config, index).as_dict(),
            "differentiation_kernel_radius_working_px": maps[
                "differentiation_kernel_radius"
            ],
            "integration_kernel_radius_working_px": maps[
                "integration_kernel_radius"
            ],
            "suppression_distance_working_px": suppression_distance,
            "border_margin_native_px": margin,
            "counts": row_counts,
            "nonfinite_response_count": 0,
        }
        if response_maps is not None:
            response_maps[index] = np.array(
                maps["response"], dtype=np.float32, copy=True
            )
    finish_started = perf_counter()
    keypoints, deduplicated, uniform_diagnostics = _finish_detection(
        source=source,
        config=config,
        selected_across_scales=selected,
    )
    selection_seconds = perf_counter() - finish_started
    timings = {
        "dense_response_seconds": dense_seconds,
        "scale_suppression_seconds": suppression_seconds,
        "subpixel_eigen_and_border_seconds": refinement_seconds,
        "duplicate_and_uniform_selection_seconds": selection_seconds,
        "selection_cpu_ms": 1000.0
        * (suppression_seconds + refinement_seconds + selection_seconds),
        "total_seconds": perf_counter() - total_started,
    }
    diagnostics = _common_diagnostics(
        backend="reference_cpu",
        source=source,
        doubled=doubled,
        config=config,
        scale_diagnostics=scales,
        counts=counts,
        keypoints=keypoints,
        deduplicated=deduplicated,
        uniform_diagnostics=uniform_diagnostics,
    )
    return DetectorResult(
        backend="reference_cpu",
        keypoints=keypoints,
        diagnostics=diagnostics,
        timings=timings,
        response_maps=response_maps,
    )


def detect_harriszplus_v4_cuda(
    image: np.ndarray,
    config: PpiAwareRuntimeConfig,
    *,
    doubled_image: np.ndarray | None,
    device: str | Any | None = None,
    return_response_maps: bool = False,
) -> DetectorResult:
    """Run PPI-aware v4 using deterministic float32 CUDA operations."""

    torch = _require_torch()
    total_started = perf_counter()
    source, doubled = _validate_detector_inputs(image, doubled_image)
    resolved = _resolve_device(config, device)
    deterministic_policy = configure_torch_determinism(config.rng_seed)
    with torch.no_grad(), torch.autocast(
        device_type=resolved.type, enabled=False
    ):
        source_tensor = torch.from_numpy(source).to(
            device=resolved, dtype=torch.float32
        )
        doubled_tensor = torch.from_numpy(doubled).to(
            device=resolved, dtype=torch.float32
        )
        _synchronize(resolved)
        gradients = {
            1.0: central_difference_torch(source_tensor),
            2.0: central_difference_torch(doubled_tensor),
        }
        selected: list[SelectionCandidate] = []
        scales: dict[str, Any] = {}
        response_maps: dict[int, np.ndarray] | None = (
            {} if return_response_maps else None
        )
        counts = _base_counts()
        source_offset = 0
        dense_seconds = transfer_seconds = 0.0
        suppression_seconds = refinement_seconds = 0.0
        for index in config.scale_indices:
            working_scale = config.working_image_scale(index)
            working = (
                doubled_tensor if working_scale == 2.0 else source_tensor
            )
            _synchronize(resolved)
            started = perf_counter()
            maps = compute_harrisz_scale_v4_torch(
                working,
                index,
                config,
                base_gradients=gradients[working_scale],
            )
            _synchronize(resolved)
            dense_seconds += perf_counter() - started
            started = perf_counter()
            candidates = _materialize_candidates(maps, index, source_offset)
            _synchronize(resolved)
            transfer_seconds += perf_counter() - started
            source_offset += len(candidates)
            started = perf_counter()
            suppression_distance = config.scale_suppression_distance_working(
                index
            )
            suppressed, _ = greedy_distance_suppression(
                candidates, suppression_distance
            )
            suppression_seconds += perf_counter() - started
            _synchronize(resolved)
            started = perf_counter()
            refined = _refine_and_filter_torch(
                suppressed, maps, config, index
            )
            _synchronize(resolved)
            margin = config.border_margin_native(index)
            border_safe = tuple(
                candidate
                for candidate in refined
                if _inside_descriptor_safe_border(
                    candidate,
                    width=source.shape[1],
                    height=source.shape[0],
                    margin=margin,
                )
            )
            refinement_seconds += perf_counter() - started
            selected.extend(border_safe)
            row_counts = _scale_counts(
                maps,
                candidates,
                suppressed,
                refined,
                border_safe,
                torch_mode=True,
            )
            for key, value in row_counts.items():
                counts[key] += value
            scales[str(index)] = {
                "scale": make_scale_spec(config, index).as_dict(),
                "differentiation_kernel_radius_working_px": maps[
                    "differentiation_kernel_radius"
                ],
                "integration_kernel_radius_working_px": maps[
                    "integration_kernel_radius"
                ],
                "suppression_distance_working_px": suppression_distance,
                "border_margin_native_px": margin,
                "counts": row_counts,
                "nonfinite_response_count": 0,
            }
            if response_maps is not None:
                response_maps[index] = (
                    maps["response"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
        finish_started = perf_counter()
        keypoints, deduplicated, uniform_diagnostics = _finish_detection(
            source=source,
            config=config,
            selected_across_scales=selected,
        )
        selection_seconds = perf_counter() - finish_started
    _synchronize(resolved)
    diagnostics = _common_diagnostics(
        backend="cuda",
        source=source,
        doubled=doubled,
        config=config,
        scale_diagnostics=scales,
        counts=counts,
        keypoints=keypoints,
        deduplicated=deduplicated,
        uniform_diagnostics=uniform_diagnostics,
    )
    diagnostics.update(
        {
            "device": str(resolved),
            "device_name": (
                torch.cuda.get_device_name(resolved)
                if resolved.type == "cuda"
                else "torch_cpu"
            ),
            "deterministic_policy": deterministic_policy,
        }
    )
    timings = {
        "dense_response_seconds": dense_seconds,
        "candidate_device_to_host_seconds": transfer_seconds,
        "scale_suppression_seconds": suppression_seconds,
        "subpixel_eigen_and_border_seconds": refinement_seconds,
        "duplicate_and_uniform_selection_seconds": selection_seconds,
        "detector_gpu_kernel_ms": 1000.0
        * (dense_seconds + refinement_seconds),
        "candidate_transfer_ms": 1000.0 * transfer_seconds,
        "selection_cpu_ms": 1000.0
        * (suppression_seconds + selection_seconds),
        "total_seconds": perf_counter() - total_started,
    }
    return DetectorResult(
        backend="cuda",
        keypoints=keypoints,
        diagnostics=diagnostics,
        timings=timings,
        response_maps=response_maps,
    )


__all__ = [
    "compute_harrisz_scale_v4_cpu",
    "compute_harrisz_scale_v4_torch",
    "detect_harriszplus_v4_cpu",
    "detect_harriszplus_v4_cuda",
    "gaussian_blur_numpy_with_radius",
    "gaussian_blur_torch_with_radius",
    "gaussian_kernel1d_with_radius",
]
