"""Native-resolution PPI-aware HarrisZ+ v4 and unchanged RootSIFT extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.contract import MethodExecutionError
from fingerprint_benchmark.sift.descriptors import rootsift
from fingerprint_benchmark.sift.extractor import SiftRepresentation

from .detector_v4 import (
    detect_harriszplus_v4_cpu,
    detect_harriszplus_v4_cuda,
)
from .extractor import (
    _backend_kind,
    _elapsed,
    _empty_prepare_timings,
    _enforce_determinism,
    _fail,
    _json_native,
    _manifest_ppi,
    _merge_detector_timings,
    _scale_mapping_records,
    _validate_adapter_config,
    _validate_detector_keypoints,
)
from .orientation_v4 import assign_orientations_v4_with_diagnostics
from .ppi_aware_v4 import (
    REPRESENTATION_VERSION,
    PpiAwareHarrisZPlusConfig,
)


def representation_sha256_v4(
    representation: SiftRepresentation,
    *,
    spatial_scale: float,
) -> str:
    """Hash deterministic representation content and the v4 spatial scale."""

    digest = hashlib.sha256()
    digest.update((REPRESENTATION_VERSION + "\0").encode("ascii"))
    arrays = (
        ("points", "<f4"),
        ("sizes", "<f4"),
        ("angles", "<f4"),
        ("responses", "<f4"),
        ("octaves", "<i4"),
        ("class_ids", "<i4"),
        ("descriptors", "<f4"),
    )
    for name, dtype in arrays:
        value = np.ascontiguousarray(
            getattr(representation, name), dtype=np.dtype(dtype)
        )
        digest.update(name.encode("ascii") + b"\0")
        digest.update(
            json.dumps(value.shape, separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    stable = {
        "width": int(representation.width),
        "height": int(representation.height),
        "manifest_ppi": float(representation.ppi),
        "reference_ppi": 1000.0,
        "spatial_scale": float(spatial_scale),
        "harriszplus_scale_indices": representation.metadata.get(
            "harriszplus_scale_indices", []
        ),
        "harriszplus_source_indices": representation.metadata.get(
            "harriszplus_source_indices", []
        ),
        "scale_mapping_records": representation.metadata.get(
            "scale_mapping_records", []
        ),
        "border_exclusion": representation.metadata.get("border_exclusion"),
        "opencv_octave_policy": representation.metadata.get(
            "opencv_octave_policy"
        ),
        "opencv_class_id_policy": representation.metadata.get(
            "opencv_class_id_policy"
        ),
    }
    digest.update(
        json.dumps(
            stable,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _cuda_measurement_start(config: object) -> tuple[Any, Any, dict[str, int]]:
    """Start a clean, physically bounded CUDA peak-memory measurement."""

    try:
        import torch
    except ImportError as exc:
        raise MethodExecutionError(
            "missing_torch_cuda_dependency",
            "The CUDA backend requires the pinned PyTorch dependency.",
        ) from exc
    if not torch.cuda.is_available():
        raise MethodExecutionError(
            "cuda_unavailable",
            "The configured HarrisZ+ v4 CUDA backend has no CUDA device.",
        )
    device = torch.device(getattr(config, "device", None) or "cuda")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_bytes, allocator_total_bytes = torch.cuda.mem_get_info(device)
    physical_total_bytes = int(
        torch.cuda.get_device_properties(device).total_memory
    )
    return torch, device, {
        "vram_free_before_bytes": int(free_bytes),
        "vram_allocator_total_before_bytes": int(allocator_total_bytes),
        "vram_physical_total_bytes": physical_total_bytes,
        "vram_allocated_before_bytes": int(
            torch.cuda.memory_allocated(device)
        ),
        "vram_reserved_before_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _cuda_measurement_finish(
    runtime: tuple[Any, Any, dict[str, int]]
) -> dict[str, Any]:
    torch, device, baseline = runtime
    torch.cuda.synchronize(device)
    measurement: dict[str, Any] = {
        **baseline,
        "peak_vram_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_vram_reserved": int(torch.cuda.max_memory_reserved(device)),
        "vram_allocated_after_bytes": int(
            torch.cuda.memory_allocated(device)
        ),
        "vram_reserved_after_bytes": int(torch.cuda.memory_reserved(device)),
        "vram_measurement_policy": (
            "synchronize_empty_cache_synchronize_reset_peak_then_measure;"
            "allocated_and_reserved_reported_separately_never_summed"
        ),
    }
    total = int(measurement["vram_physical_total_bytes"])
    measurement["peak_vram_allocated_within_physical"] = (
        int(measurement["peak_vram_allocated"]) <= total
    )
    measurement["peak_vram_reserved_within_physical"] = (
        int(measurement["peak_vram_reserved"]) <= total
    )
    measurement["vram_measurement_valid"] = bool(
        measurement["peak_vram_allocated_within_physical"]
        and measurement["peak_vram_reserved_within_physical"]
    )
    if not measurement["vram_measurement_valid"]:
        raise MethodExecutionError(
            "invalid_vram_measurement",
            "CUDA allocator peaks exceed physical device memory.",
            diagnostics=measurement,
        )
    for field in (
        "vram_free_before_bytes",
        "vram_allocator_total_before_bytes",
        "vram_physical_total_bytes",
        "vram_allocated_before_bytes",
        "vram_reserved_before_bytes",
        "peak_vram_allocated",
        "peak_vram_reserved",
        "vram_allocated_after_bytes",
        "vram_reserved_after_bytes",
    ):
        measurement[field.replace("_bytes", "") + "_mib"] = (
            float(measurement[field]) / (1024.0 * 1024.0)
        )
    return measurement


def extract_representation_v4(
    image_path: Path,
    image_metadata: Mapping[str, Any],
    config: PpiAwareHarrisZPlusConfig,
) -> tuple[SiftRepresentation, dict[str, Any], float]:
    """Prepare one v4 representation; only spatial detector interpretation changes."""

    started = perf_counter_ns()
    timings = _empty_prepare_timings()
    load_started = perf_counter_ns()
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    timings["image_load_ms"] = _elapsed(load_started)
    if gray is None or gray.ndim != 2 or gray.size == 0:
        _fail(
            "image_read_failure",
            f"OpenCV could not read a native grayscale image: {image_path}",
            started,
            timings,
            path=str(image_path),
        )
    ppi = _manifest_ppi(image_metadata, started, timings)
    runtime = config.runtime(ppi)
    _validate_adapter_config(runtime)
    _enforce_determinism(runtime)

    lanczos_started = perf_counter_ns()
    try:
        doubled_u8 = cv2.resize(
            gray,
            (int(gray.shape[1]) * 2, int(gray.shape[0]) * 2),
            interpolation=cv2.INTER_LANCZOS4,
        )
    except (ValueError, cv2.error) as exc:
        timings["lanczos_ms"] = _elapsed(lanczos_started)
        _fail(
            "lanczos_doubling_failure",
            str(exc),
            started,
            timings,
            source_shape=list(gray.shape),
        )
    timings["lanczos_ms"] = _elapsed(lanczos_started)
    image_float = np.ascontiguousarray(gray, dtype=np.float32)
    doubled_float = np.ascontiguousarray(doubled_u8, dtype=np.float32)
    if doubled_float.shape != (gray.shape[0] * 2, gray.shape[1] * 2):
        _fail(
            "invalid_lanczos_shape",
            "INTER_LANCZOS4 did not produce the exact two-times image shape.",
            started,
            timings,
        )

    backend_kind = _backend_kind(str(runtime.backend))
    cuda_runtime = (
        _cuda_measurement_start(runtime) if backend_kind == "cuda" else None
    )
    detector_started = perf_counter_ns()
    try:
        if backend_kind == "cuda":
            detector_result = detect_harriszplus_v4_cuda(
                image_float,
                runtime,
                doubled_image=doubled_float,
                device=runtime.device,
                return_response_maps=False,
            )
            assert cuda_runtime is not None
            cuda_runtime[0].cuda.synchronize(cuda_runtime[1])
        else:
            detector_result = detect_harriszplus_v4_cpu(
                image_float,
                runtime,
                doubled_image=doubled_float,
                return_response_maps=False,
            )
    except MethodExecutionError:
        raise
    except Exception as exc:
        _fail(
            "detector_failure",
            str(exc),
            started,
            timings,
            backend=backend_kind,
            exception_type=type(exc).__name__,
        )
    detector_wall_ms = _elapsed(detector_started)
    timings[
        "detector_gpu_wall_ms"
        if backend_kind == "cuda"
        else "detector_cpu_wall_ms"
    ] = detector_wall_ms
    detector_timings = _json_native(
        dict(getattr(detector_result, "timings", {}))
    )
    _merge_detector_timings(
        timings, detector_timings, backend=backend_kind
    )
    if cuda_runtime is not None:
        timings.update(_cuda_measurement_finish(cuda_runtime))

    detected = tuple(getattr(detector_result, "keypoints", ()))
    if len(detected) > int(runtime.max_keypoints):
        _fail(
            "keypoint_cap_violation",
            "HarrisZ+ v4 exceeded the frozen 3000-keypoint cap.",
            started,
            timings,
            detector_keypoint_count=len(detected),
        )
    if not detected:
        _fail(
            "missing_descriptors",
            "HarrisZ+ v4 produced no descriptor-safe keypoints.",
            started,
            timings,
        )
    _validate_detector_keypoints(detected, started, timings)

    orientation_started = perf_counter_ns()
    try:
        angles, orientation_diagnostics = (
            assign_orientations_v4_with_diagnostics(
                image_float, detected, runtime
            )
        )
    except (ValueError, cv2.error) as exc:
        timings["orientation_cpu_ms"] = _elapsed(orientation_started)
        _fail(
            "orientation_assignment_failure",
            str(exc),
            started,
            timings,
        )
    timings["orientation_cpu_ms"] = _elapsed(orientation_started)
    supplied_keypoints = [
        cv2.KeyPoint(
            float(item.x),
            float(item.y),
            float(item.size),
            float(angle),
            float(item.response),
            0,
            int(item.scale_index),
        )
        for item, angle in zip(detected, angles, strict=True)
    ]

    descriptor_started = perf_counter_ns()
    sift = cv2.SIFT_create(
        nfeatures=int(runtime.max_keypoints),
        nOctaveLayers=int(runtime.sift_n_octave_layers),
        contrastThreshold=float(runtime.sift_contrast_threshold),
        edgeThreshold=float(runtime.sift_edge_threshold),
        sigma=float(runtime.sift_sigma),
    )
    try:
        computed_keypoints, raw_descriptors = sift.compute(
            gray, supplied_keypoints
        )
    except cv2.error as exc:
        timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)
        _fail(
            "sift_descriptor_failure",
            str(exc),
            started,
            timings,
        )
    computed_keypoints = list(computed_keypoints or [])
    if raw_descriptors is None:
        _fail(
            "missing_descriptors",
            "OpenCV SIFT.compute produced no supplied-keypoint descriptors.",
            started,
            timings,
        )
    raw = np.asarray(raw_descriptors)
    if (
        raw.ndim != 2
        or raw.shape[1] != 128
        or raw.shape[0] != len(computed_keypoints)
        or len(computed_keypoints) != len(supplied_keypoints)
    ):
        _fail(
            "descriptor_keypoint_mismatch",
            "OpenCV supplied-keypoint output count or shape changed.",
            started,
            timings,
            raw_shape=list(raw.shape),
            supplied_count=len(supplied_keypoints),
            computed_count=len(computed_keypoints),
        )
    finite_rows = np.isfinite(raw).all(axis=1)
    nonfinite_rows = int(np.count_nonzero(~finite_rows))
    retained_opencv = [
        keypoint
        for keypoint, keep in zip(
            computed_keypoints, finite_rows, strict=True
        )
        if bool(keep)
    ]
    retained_detector = [
        keypoint
        for keypoint, keep in zip(detected, finite_rows, strict=True)
        if bool(keep)
    ]
    finite_raw = np.ascontiguousarray(raw[finite_rows], dtype=np.float32)
    if finite_raw.shape[0] < int(runtime.minimum_descriptors):
        _fail(
            "too_few_descriptors",
            "Too few finite RootSIFT descriptors remain.",
            started,
            timings,
            finite_descriptor_count=int(finite_raw.shape[0]),
        )
    try:
        descriptors = rootsift(finite_raw)
    except ValueError as exc:
        _fail(
            "invalid_descriptors",
            str(exc),
            started,
            timings,
        )
    timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)

    scale_records = _scale_mapping_records(detected)
    representation = SiftRepresentation(
        points=np.asarray(
            [keypoint.pt for keypoint in retained_opencv],
            dtype=np.float32,
        ),
        sizes=np.asarray(
            [keypoint.size for keypoint in retained_opencv],
            dtype=np.float32,
        ),
        angles=np.mod(
            np.asarray(
                [keypoint.angle for keypoint in retained_opencv],
                dtype=np.float32,
            ),
            np.float32(360.0),
        ),
        responses=np.asarray(
            [keypoint.response for keypoint in retained_opencv],
            dtype=np.float32,
        ),
        octaves=np.asarray(
            [keypoint.octave for keypoint in retained_opencv],
            dtype=np.int32,
        ),
        class_ids=np.asarray(
            [keypoint.class_id for keypoint in retained_opencv],
            dtype=np.int32,
        ),
        descriptors=descriptors,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
        ppi=ppi,
        metadata={
            "source_path": str(image_path),
            "method_version": config.method_version,
            "representation_version": REPRESENTATION_VERSION,
            "image_policy": "native_grayscale_no_enhancement_no_hidden_resize",
            "manifest_ppi": ppi,
            "reference_ppi": config.reference_ppi,
            "spatial_scale": runtime.spatial_scale,
            "spatial_scale_formula": "manifest_ppi / 1000.0",
            "backend": backend_kind,
            "lanczos_policy": (
                "one_exact_2x_resize_for_scale_indices_0_and_1"
            ),
            "border_exclusion": (
                "descriptor_safe_per_scale_before_uniform_cap"
            ),
            "scale_mapping_records": scale_records,
            "harriszplus_scale_indices": [
                int(keypoint.scale_index)
                for keypoint in retained_detector
            ],
            "harriszplus_source_indices": [
                int(keypoint.source_index)
                for keypoint in retained_detector
            ],
            "opencv_octave_policy": "zero_for_all_native_supplied_keypoints",
            "opencv_class_id_policy": "harriszplus_scale_index",
            "descriptor_mode": "rootsift_existing_implementation_unchanged",
            "prepare_total_ms": None,
        },
    )
    representation_hash = representation_sha256_v4(
        representation, spatial_scale=runtime.spatial_scale
    )
    detector_diagnostics = _json_native(
        dict(getattr(detector_result, "diagnostics", {}))
    )
    diagnostics = {
        "method_version": config.method_version,
        "representation_version": REPRESENTATION_VERSION,
        "image_policy": "native_grayscale_no_downsampling_no_enhancement",
        "no_hidden_resize": True,
        "native_width": int(gray.shape[1]),
        "native_height": int(gray.shape[0]),
        "manifest_ppi": ppi,
        "reference_ppi": config.reference_ppi,
        "spatial_scale": runtime.spatial_scale,
        "spatial_scale_formula": "manifest_ppi / 1000.0",
        "detector_backend": backend_kind,
        "detector_keypoint_count": len(detected),
        "max_keypoints": int(runtime.max_keypoints),
        "descriptor_count": int(descriptors.shape[0]),
        "dropped_descriptor_count": nonfinite_rows,
        "descriptor_dimension": int(descriptors.shape[1]),
        "descriptor_dtype": str(descriptors.dtype),
        "descriptor_mode": "rootsift",
        "rootsift_function": (
            "fingerprint_benchmark.sift.descriptors.rootsift"
        ),
        "scale_mapping_records": scale_records,
        "representation_sha256": representation_hash,
        **detector_diagnostics,
        **orientation_diagnostics,
        **timings,
    }
    total_ms = _elapsed(started)
    diagnostics["prepare_total_ms"] = total_ms
    representation.metadata["prepare_total_ms"] = total_ms
    return representation, diagnostics, total_ms


__all__ = [
    "extract_representation_v4",
    "representation_sha256_v4",
]
