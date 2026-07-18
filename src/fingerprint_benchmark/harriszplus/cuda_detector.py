"""Deterministic float32 PyTorch/CUDA HarrisZ+ detector backend."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .config import HarrisZPlusConfig
from .kernels import (
    central_difference_torch,
    gaussian_blur_torch,
    sample_zscore_torch,
    validate_grayscale_float32,
)
from .selection import (
    greedy_distance_suppression,
    iterative_uniform_selection_with_diagnostics,
    rank_candidates,
    refine_subpixel_torch,
    remove_scale_01_duplicates,
    scale_suppression_distance,
    strict_local_maxima_torch,
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


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CPU-only installation path.
        raise RuntimeError("The HarrisZ+ CUDA backend requires PyTorch.") from exc
    return torch


def configure_torch_determinism(seed: int = 0) -> dict[str, Any]:
    """Apply and report the frozen deterministic float32 execution policy."""

    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    return {
        "rng_seed": int(seed),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "dtype": "float32",
        "autocast": False,
    }


def _resolve_device(config: HarrisZPlusConfig, device: str | Any | None) -> Any:
    torch = _require_torch()
    requested = device if device is not None else config.device
    resolved = torch.device(requested if requested is not None else "cuda")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for HarrisZ+ but torch.cuda.is_available() is false.")
    return resolved


def _synchronize(device: Any) -> None:
    torch = _require_torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_harrisz_scale_torch(
    working_image: Any,
    scale_index: int,
    config: HarrisZPlusConfig | None = None,
    *,
    base_gradients: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Compute one dense HarrisZ scale entirely with float32 torch tensors."""

    torch = _require_torch()
    active_config = config or HarrisZPlusConfig(backend="cuda")
    active_config._validate_scale_index(scale_index)
    if working_image.ndim != 2 or working_image.dtype != torch.float32:
        raise ValueError("working_image must be a 2D float32 torch tensor.")
    if base_gradients is None:
        base_gradient_x, base_gradient_y = central_difference_torch(working_image)
    else:
        base_gradient_x, base_gradient_y = base_gradients
        if base_gradient_x.shape != working_image.shape or base_gradient_y.shape != working_image.shape:
            raise ValueError("base_gradients must match working_image shape.")
        if base_gradient_x.dtype != torch.float32 or base_gradient_y.dtype != torch.float32:
            raise ValueError("base_gradients must be float32 tensors.")

    differentiation_sigma = active_config.working_sigma(scale_index)
    integration_sigma = active_config.working_integration_sigma(scale_index)
    truncate = active_config.gaussian_truncate
    scale_gradient_x = gaussian_blur_torch(
        base_gradient_x,
        differentiation_sigma,
        truncate=truncate,
    )
    scale_gradient_y = gaussian_blur_torch(
        base_gradient_y,
        differentiation_sigma,
        truncate=truncate,
    )
    magnitude = torch.sqrt(scale_gradient_x * scale_gradient_x + scale_gradient_y * scale_gradient_y)
    magnitude_mean = torch.mean(magnitude)
    raw_edge_threshold = magnitude_mean * active_config.raw_edge_mean_multiplier
    raw_edge_mask = magnitude > raw_edge_threshold
    edge_mask = gaussian_blur_torch(
        raw_edge_mask.to(torch.float32),
        differentiation_sigma,
        truncate=truncate,
    )

    enhanced_x = scale_gradient_x * edge_mask
    enhanced_y = scale_gradient_y * edge_mask
    autocorrelation_xx = gaussian_blur_torch(
        enhanced_x * enhanced_x,
        integration_sigma,
        truncate=truncate,
    )
    autocorrelation_xy = gaussian_blur_torch(
        enhanced_x * enhanced_y,
        integration_sigma,
        truncate=truncate,
    )
    autocorrelation_yy = gaussian_blur_torch(
        enhanced_y * enhanced_y,
        integration_sigma,
        truncate=truncate,
    )
    determinant = autocorrelation_xx * autocorrelation_yy - autocorrelation_xy * autocorrelation_xy
    trace = autocorrelation_xx + autocorrelation_yy
    trace_squared = trace * trace
    response = sample_zscore_torch(determinant) - sample_zscore_torch(trace_squared)
    nonfinite_response_count = int(torch.count_nonzero(~torch.isfinite(response)).item())
    if nonfinite_response_count:
        raise FloatingPointError(
            f"HarrisZ+ scale {scale_index} produced {nonfinite_response_count} "
            "non-finite dense responses."
        )
    response_positive_mask = response > active_config.response_threshold
    candidate_mask = response_positive_mask & (edge_mask > active_config.edge_mask_threshold)
    local_maxima_mask = strict_local_maxima_torch(
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
            "doubled_image is required: the extractor must supply the shared deterministic "
            "OpenCV INTER_LANCZOS4 2x array."
        )
    doubled = validate_grayscale_float32(doubled_image, name="doubled_image")
    expected_shape = (source.shape[0] * 2, source.shape[1] * 2)
    if doubled.shape != expected_shape:
        raise ValueError(
            f"doubled_image must have shape {expected_shape}; got {doubled.shape}."
        )
    return source, doubled


def _materialize_candidates(
    maps: Mapping[str, Any],
    scale_index: int,
    source_offset: int,
) -> list[_WorkingCandidate]:
    locations = maps["local_maxima_mask"].nonzero(as_tuple=False)
    if locations.numel() == 0:
        return []
    responses = maps["response"][locations[:, 0], locations[:, 1]]
    locations_cpu = locations.detach().cpu().numpy()
    responses_cpu = responses.detach().cpu().numpy()
    return [
        _WorkingCandidate(
            x=float(x),
            y=float(y),
            response=float(response),
            scale_index=scale_index,
            source_index=source_offset + offset,
            integer_x=int(x),
            integer_y=int(y),
        )
        for offset, ((y, x), response) in enumerate(zip(locations_cpu, responses_cpu))
    ]


def _refine_and_filter_torch(
    candidates: tuple[_WorkingCandidate, ...],
    maps: Mapping[str, Any],
    config: HarrisZPlusConfig,
    scale_index: int,
) -> tuple[SelectionCandidate, ...]:
    torch = _require_torch()
    if not candidates:
        return ()
    device = maps["response"].device
    x = torch.tensor([candidate.integer_x for candidate in candidates], dtype=torch.int64, device=device)
    y = torch.tensor([candidate.integer_y for candidate in candidates], dtype=torch.int64, device=device)
    a_xx = maps["autocorrelation_xx"][y, x]
    a_xy = maps["autocorrelation_xy"][y, x]
    a_yy = maps["autocorrelation_yy"][y, x]
    trace = a_xx + a_yy
    discriminant_squared = (a_xx - a_yy) * (a_xx - a_yy) + 4.0 * a_xy * a_xy
    discriminant = torch.sqrt(torch.clamp(discriminant_squared, min=0.0))
    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)
    safe_maximum = torch.where(lambda_max > 0.0, lambda_max, torch.ones_like(lambda_max))
    axis_ratio = torch.sqrt(torch.clamp(lambda_min / safe_maximum, min=0.0))
    valid = (
        torch.isfinite(axis_ratio)
        & torch.isfinite(lambda_max)
        & torch.isfinite(lambda_min)
        & (lambda_max > 0.0)
        & (lambda_min > 0.0)
        & (axis_ratio > config.eigen_axis_ratio_threshold)
    )
    refined_x, refined_y = refine_subpixel_torch(maps["response"], x, y)
    valid_cpu = valid.detach().cpu().numpy().astype(bool, copy=False)
    refined_x_cpu = refined_x.detach().cpu().numpy()
    refined_y_cpu = refined_y.detach().cpu().numpy()
    coordinate_divisor = config.working_image_scale(scale_index)
    return tuple(
        SelectionCandidate(
            x=float(refined_x_cpu[offset]) / coordinate_divisor,
            y=float(refined_y_cpu[offset]) / coordinate_divisor,
            response=candidate.response,
            scale_index=scale_index,
            source_index=candidate.source_index,
        )
        for offset, candidate in enumerate(candidates)
        if valid_cpu[offset]
    )


def _to_detected_keypoint(candidate: SelectionCandidate, config: HarrisZPlusConfig) -> DetectedKeypoint:
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


def detect_harriszplus_cuda(
    image: np.ndarray,
    config: HarrisZPlusConfig | None = None,
    *,
    doubled_image: np.ndarray | None = None,
    device: str | Any | None = None,
    return_response_maps: bool = False,
) -> DetectorResult:
    """Detect HarrisZ+ keypoints using deterministic dense torch operations."""

    torch = _require_torch()
    active_config = config or HarrisZPlusConfig(backend="cuda")
    total_started = perf_counter()
    validation_started = perf_counter()
    source, doubled = _validate_detector_inputs(image, doubled_image)
    resolved_device = _resolve_device(active_config, device)
    deterministic_policy = configure_torch_determinism(active_config.rng_seed)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    validation_seconds = perf_counter() - validation_started

    # Explicitly disable autocast even if the caller entered an autocast scope.
    with torch.no_grad(), torch.autocast(device_type=resolved_device.type, enabled=False):
        transfer_started = perf_counter()
        source_tensor = torch.from_numpy(source).to(device=resolved_device, dtype=torch.float32)
        doubled_tensor = torch.from_numpy(doubled).to(device=resolved_device, dtype=torch.float32)
        _synchronize(resolved_device)
        input_transfer_seconds = perf_counter() - transfer_started

        detector_gpu_kernel_ms = 0.0
        gradient_event_start = None
        gradient_event_end = None
        if resolved_device.type == "cuda":
            gradient_event_start = torch.cuda.Event(enable_timing=True)
            gradient_event_end = torch.cuda.Event(enable_timing=True)
            gradient_event_start.record()
        gradient_started = perf_counter()
        gradients_by_image_scale = {
            1.0: central_difference_torch(source_tensor),
            2.0: central_difference_torch(doubled_tensor),
        }
        if gradient_event_end is not None:
            gradient_event_end.record()
        _synchronize(resolved_device)
        gradient_seconds = perf_counter() - gradient_started
        if gradient_event_start is not None and gradient_event_end is not None:
            detector_gpu_kernel_ms += float(gradient_event_start.elapsed_time(gradient_event_end))
        else:
            detector_gpu_kernel_ms += 1000.0 * gradient_seconds

        scale_diagnostics: dict[str, Any] = {}
        response_maps: dict[int, np.ndarray] | None = {} if return_response_maps else None
        selected_across_scales: list[SelectionCandidate] = []
        source_offset = 0
        dense_response_seconds = 0.0
        candidate_transfer_seconds = 0.0
        scale_suppression_seconds = 0.0
        refinement_seconds = 0.0
        response_map_transfer_seconds = 0.0
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
            working_image = doubled_tensor if working_scale == 2.0 else source_tensor
            _synchronize(resolved_device)
            dense_event_start = None
            dense_event_end = None
            if resolved_device.type == "cuda":
                dense_event_start = torch.cuda.Event(enable_timing=True)
                dense_event_end = torch.cuda.Event(enable_timing=True)
                dense_event_start.record()
            dense_started = perf_counter()
            maps = compute_harrisz_scale_torch(
                working_image,
                scale_index,
                active_config,
                base_gradients=gradients_by_image_scale[working_scale],
            )
            if dense_event_end is not None:
                dense_event_end.record()
            _synchronize(resolved_device)
            dense_elapsed = perf_counter() - dense_started
            dense_response_seconds += dense_elapsed
            if dense_event_start is not None and dense_event_end is not None:
                detector_gpu_kernel_ms += float(dense_event_start.elapsed_time(dense_event_end))
            else:
                detector_gpu_kernel_ms += 1000.0 * dense_elapsed

            transfer_started = perf_counter()
            candidates = _materialize_candidates(maps, scale_index, source_offset)
            _synchronize(resolved_device)
            candidate_transfer_seconds += perf_counter() - transfer_started
            source_offset += len(candidates)

            suppression_started = perf_counter()
            suppression_distance = scale_suppression_distance(active_config.working_sigma(scale_index))
            scale_selected, _ = greedy_distance_suppression(candidates, suppression_distance)
            scale_suppression_seconds += perf_counter() - suppression_started

            refinement_event_start = None
            refinement_event_end = None
            if resolved_device.type == "cuda":
                refinement_event_start = torch.cuda.Event(enable_timing=True)
                refinement_event_end = torch.cuda.Event(enable_timing=True)
                refinement_event_start.record()
            refinement_started = perf_counter()
            refined = _refine_and_filter_torch(scale_selected, maps, active_config, scale_index)
            if refinement_event_end is not None:
                refinement_event_end.record()
            _synchronize(resolved_device)
            refinement_elapsed = perf_counter() - refinement_started
            refinement_seconds += refinement_elapsed
            if refinement_event_start is not None and refinement_event_end is not None:
                detector_gpu_kernel_ms += float(
                    refinement_event_start.elapsed_time(refinement_event_end)
                )
            else:
                detector_gpu_kernel_ms += 1000.0 * refinement_elapsed
            selected_across_scales.extend(refined)
            invalid_or_nan_outputs += int(maps["nonfinite_response_count"])

            scale_counts = {
                "dense_pixels": int(maps["response"].numel()),
                "candidates_before_mask": int(maps["response_positive_mask"].sum().item()),
                "candidates_after_mask": int(maps["candidate_mask"].sum().item()),
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
                "timing_seconds": dense_elapsed,
            }
            if response_maps is not None:
                map_transfer_started = perf_counter()
                response_maps[scale_index] = maps["response"].detach().cpu().numpy().astype(
                    np.float32,
                    copy=True,
                )
                _synchronize(resolved_device)
                response_map_transfer_seconds += perf_counter() - map_transfer_started

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
    _synchronize(resolved_device)
    total_seconds = perf_counter() - total_started

    peak_vram_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(resolved_device)) if resolved_device.type == "cuda" else 0
    )
    peak_vram_reserved_bytes = (
        int(torch.cuda.max_memory_reserved(resolved_device)) if resolved_device.type == "cuda" else 0
    )
    selection_cpu_ms = 1000.0 * (
        scale_suppression_seconds + duplicate_seconds + uniform_seconds
    )
    timings = {
        "input_validation_and_setup_seconds": validation_seconds,
        "input_host_to_device_seconds": input_transfer_seconds,
        "base_gradients_seconds": gradient_seconds,
        "dense_response_seconds": dense_response_seconds,
        "candidate_device_to_host_seconds": candidate_transfer_seconds,
        "scale_suppression_seconds": scale_suppression_seconds,
        "subpixel_and_eigen_seconds": refinement_seconds,
        "response_map_device_to_host_seconds": response_map_transfer_seconds,
        "duplicate_removal_seconds": duplicate_seconds,
        "uniform_selection_seconds": uniform_seconds,
        "detector_gpu_kernel_ms": detector_gpu_kernel_ms,
        "candidate_transfer_ms": 1000.0 * candidate_transfer_seconds,
        "selection_cpu_ms": selection_cpu_ms,
        "total_seconds": total_seconds,
        "total_ms": 1000.0 * total_seconds,
    }
    diagnostics: dict[str, Any] = {
        "backend": "cuda",
        "device": str(resolved_device),
        "device_name": (
            torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else "torch_cpu"
        ),
        "input_shape": [int(source.shape[0]), int(source.shape[1])],
        "doubled_shape": [int(doubled.shape[0]), int(doubled.shape[1])],
        "input_dtype": str(source.dtype),
        "scale_indices": list(active_config.scale_indices),
        "scales": scale_diagnostics,
        "deterministic_policy": deterministic_policy,
        "peak_vram_bytes": peak_vram_allocated_bytes,
        "peak_vram_allocated_bytes": peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": peak_vram_reserved_bytes,
        **counts,
        "candidates_after_duplicate_removal": len(deduplicated),
        "candidates_after_uniform_selection": len(uniform),
        "final_keypoint_count": len(keypoints),
        "cap_truncated_count": uniform_diagnostics["cap_truncated_count"],
        "uniform_selection": uniform_diagnostics,
        "invalid_or_nan_outputs": invalid_or_nan_outputs,
        "detector_gpu_kernel_timing_source": (
            "summed_cuda_event_elapsed_time"
            if resolved_device.type == "cuda"
            else "torch_cpu_synchronized_wall_time_fallback"
        ),
    }
    return DetectorResult(
        backend="cuda",
        keypoints=keypoints,
        diagnostics=diagnostics,
        timings=timings,
        response_maps=response_maps,
    )


class HarrisZPlusCUDADetector:
    def __init__(
        self,
        config: HarrisZPlusConfig | None = None,
        *,
        device: str | Any | None = None,
    ) -> None:
        self.config = config or HarrisZPlusConfig(backend="cuda")
        self.device = device

    def detect(
        self,
        image: np.ndarray,
        *,
        doubled_image: np.ndarray | None = None,
        return_response_maps: bool = False,
    ) -> DetectorResult:
        return detect_harriszplus_cuda(
            image,
            self.config,
            doubled_image=doubled_image,
            device=self.device,
            return_response_maps=return_response_maps,
        )


HarrisZPlusCudaDetector = HarrisZPlusCUDADetector
detect_cuda = detect_harriszplus_cuda
detect = detect_harriszplus_cuda


__all__ = [
    "HarrisZPlusCUDADetector",
    "HarrisZPlusCudaDetector",
    "configure_torch_determinism",
    "compute_harrisz_scale_torch",
    "detect_harriszplus_cuda",
    "detect_cuda",
    "detect",
]
