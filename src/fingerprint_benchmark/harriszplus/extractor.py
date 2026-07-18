"""Native-resolution HarrisZ+ detection and supplied-keypoint RootSIFT extraction."""

from __future__ import annotations

from pathlib import Path
import random
from time import perf_counter_ns
from typing import Any, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.contract import MethodExecutionError
from fingerprint_benchmark.sift.descriptors import rootsift
from fingerprint_benchmark.sift.extractor import SiftRepresentation

from .config import HarrisZPlusConfig
from .orientation import assign_orientations_with_diagnostics
from .provenance import representation_sha256
from .reference_cpu import detect_harriszplus_cpu


def extract_representation(
    image_path: Path,
    image_metadata: Mapping[str, Any],
    config: HarrisZPlusConfig,
) -> tuple[SiftRepresentation, dict[str, Any], float]:
    """Detect HarrisZ+ points and compute unchanged OpenCV-SIFT/RootSIFT descriptors."""

    started = perf_counter_ns()
    timings = _empty_prepare_timings()
    load_started = perf_counter_ns()
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    timings["image_load_ms"] = _elapsed(load_started)
    if gray is None:
        _fail(
            "image_read_failure",
            f"OpenCV could not read native grayscale image: {image_path}",
            started,
            timings,
            path=str(image_path),
        )
    assert gray is not None
    if gray.ndim != 2 or gray.size == 0:
        _fail(
            "invalid_grayscale_image",
            "HarrisZ+ requires a non-empty native grayscale image.",
            started,
            timings,
            path=str(image_path),
            shape=list(gray.shape),
        )
    ppi = _manifest_ppi(image_metadata, started, timings)
    _validate_adapter_config(config)
    _enforce_determinism(config)

    lanczos_started = perf_counter_ns()
    try:
        # Deliberately resize uint8 source pixels first, matching image-resize semantics,
        # and convert the two detector inputs to float32 only after interpolation.
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
            source_shape=list(gray.shape),
            doubled_shape=list(doubled_float.shape),
        )

    backend = str(config.backend)
    backend_kind = _backend_kind(backend)
    torch_runtime = _cuda_timing_start(config) if backend_kind == "cuda" else None
    detector_started = perf_counter_ns()
    try:
        if backend_kind == "cuda":
            from .cuda_detector import detect_harriszplus_cuda

            detector_result = detect_harriszplus_cuda(
                image_float,
                config,
                device=config.device,
                doubled_image=doubled_float,
                return_response_maps=False,
            )
        elif backend_kind == "reference_cpu":
            detector_result = detect_harriszplus_cpu(
                image_float,
                config,
                doubled_image=doubled_float,
                return_response_maps=False,
            )
        else:
            raise ValueError(f"Unsupported HarrisZ+ backend: {backend!r}.")
        if torch_runtime is not None:
            _cuda_synchronize(torch_runtime)
    except MethodExecutionError:
        raise
    except Exception as exc:
        if torch_runtime is not None:
            _cuda_synchronize(torch_runtime, suppress_errors=True)
        timings[
            "detector_gpu_wall_ms" if backend_kind == "cuda" else "detector_cpu_wall_ms"
        ] = (
            _elapsed(detector_started)
        )
        _fail(
            "detector_failure",
            str(exc),
            started,
            timings,
            backend=backend,
            exception_type=type(exc).__name__,
        )
    detector_wall_ms = _elapsed(detector_started)
    if backend_kind == "cuda":
        timings["detector_gpu_wall_ms"] = detector_wall_ms
    else:
        timings["detector_cpu_wall_ms"] = detector_wall_ms
    detector_timings = _json_native(dict(getattr(detector_result, "timings", {})))
    if not isinstance(detector_timings, dict):
        detector_timings = {}
    _merge_detector_timings(timings, detector_timings, backend=backend_kind)
    if torch_runtime is not None:
        timings.update(_cuda_peak_memory(torch_runtime))

    detected = tuple(getattr(detector_result, "keypoints", ()))
    if len(detected) > int(config.max_keypoints):
        _fail(
            "keypoint_cap_violation",
            f"Detector returned {len(detected)} keypoints above the frozen cap {config.max_keypoints}.",
            started,
            timings,
            detector_keypoint_count=len(detected),
            max_keypoints=int(config.max_keypoints),
        )
    if not detected:
        _fail(
            "missing_descriptors",
            "HarrisZ+ detector produced no keypoints for RootSIFT extraction.",
            started,
            timings,
            detector_keypoint_count=0,
        )
    _validate_detector_keypoints(detected, started, timings)

    orientation_started = perf_counter_ns()
    try:
        angles, orientation_diagnostics = assign_orientations_with_diagnostics(
            image_float,
            detected,
            config,
        )
    except (ValueError, cv2.error) as exc:
        timings["orientation_cpu_ms"] = _elapsed(orientation_started)
        _fail(
            "orientation_assignment_failure",
            str(exc),
            started,
            timings,
            requested_keypoints=len(detected),
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
    if any(keypoint.angle < 0.0 for keypoint in supplied_keypoints):
        _fail(
            "invalid_supplied_orientation",
            "Every supplied OpenCV keypoint must have an explicit angle in [0, 360).",
            started,
            timings,
        )

    descriptor_started = perf_counter_ns()
    descriptor_compute_started = perf_counter_ns()
    sift = cv2.SIFT_create(
        nfeatures=int(config.max_keypoints),
        nOctaveLayers=int(config.sift_n_octave_layers),
        contrastThreshold=float(config.sift_contrast_threshold),
        edgeThreshold=float(config.sift_edge_threshold),
        sigma=float(config.sift_sigma),
    )
    try:
        computed_keypoints, raw_descriptors = sift.compute(gray, supplied_keypoints)
    except cv2.error as exc:
        timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)
        _fail(
            "sift_descriptor_failure",
            str(exc),
            started,
            timings,
            requested_keypoints=len(supplied_keypoints),
        )
    descriptor_compute_ms = _elapsed(descriptor_compute_started)
    computed_keypoints = list(computed_keypoints or [])
    if raw_descriptors is None:
        timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)
        _fail(
            "missing_descriptors",
            "OpenCV SIFT.compute produced no descriptors at supplied HarrisZ+ keypoints.",
            started,
            timings,
            requested_keypoints=len(supplied_keypoints),
            computed_keypoints=len(computed_keypoints),
        )
    assert raw_descriptors is not None
    raw = np.asarray(raw_descriptors)
    if (
        raw.ndim != 2
        or raw.shape[1] != 128
        or raw.shape[0] != len(computed_keypoints)
        or len(computed_keypoints) != len(supplied_keypoints)
    ):
        timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)
        _fail(
            "descriptor_keypoint_mismatch",
            "OpenCV supplied-keypoint SIFT output has inconsistent shape or keypoint count.",
            started,
            timings,
            descriptor_shape=list(raw.shape),
            computed_keypoints=len(computed_keypoints),
            requested_keypoints=len(supplied_keypoints),
        )
    finite_rows = np.isfinite(raw).all(axis=1)
    non_finite_value_count = int(np.count_nonzero(~np.isfinite(raw)))
    non_finite_row_count = int(np.count_nonzero(~finite_rows))
    retained_keypoints = [
        keypoint for keypoint, keep in zip(computed_keypoints, finite_rows, strict=True) if bool(keep)
    ]
    retained_detector_keypoints = [
        keypoint for keypoint, keep in zip(detected, finite_rows, strict=True) if bool(keep)
    ]
    finite_raw = np.ascontiguousarray(raw[finite_rows], dtype=np.float32)
    if finite_raw.shape[0] < int(config.minimum_descriptors):
        timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)
        _fail(
            "too_few_descriptors",
            (
                f"Only {finite_raw.shape[0]} finite descriptors remain; at least "
                f"{config.minimum_descriptors} are required."
            ),
            started,
            timings,
            requested_keypoints=len(supplied_keypoints),
            computed_keypoints=len(computed_keypoints),
            dropped_descriptor_count=non_finite_row_count,
            non_finite_descriptor_value_count=non_finite_value_count,
        )
    rootsift_started = perf_counter_ns()
    try:
        # This is the existing implementation, imported directly and left unchanged.
        descriptors = rootsift(finite_raw)
    except ValueError as exc:
        timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)
        _fail(
            "invalid_descriptors",
            str(exc),
            started,
            timings,
            finite_descriptor_count=int(finite_raw.shape[0]),
        )
    rootsift_ms = _elapsed(rootsift_started)
    timings["descriptor_cpu_ms"] = _elapsed(descriptor_started)

    points = np.asarray([keypoint.pt for keypoint in retained_keypoints], dtype=np.float32)
    sizes = np.asarray([keypoint.size for keypoint in retained_keypoints], dtype=np.float32)
    result_angles = np.mod(
        np.asarray([keypoint.angle for keypoint in retained_keypoints], dtype=np.float32),
        np.float32(360.0),
    )
    responses = np.asarray(
        [keypoint.response for keypoint in retained_keypoints], dtype=np.float32
    )
    octaves = np.asarray([keypoint.octave for keypoint in retained_keypoints], dtype=np.int32)
    class_ids = np.asarray(
        [keypoint.class_id for keypoint in retained_keypoints], dtype=np.int32
    )
    scale_records = _scale_mapping_records(detected)
    detector_diagnostics = _json_native(dict(getattr(detector_result, "diagnostics", {})))
    if not isinstance(detector_diagnostics, dict):
        detector_diagnostics = {}
    representation = SiftRepresentation(
        points=points,
        sizes=sizes,
        angles=result_angles,
        responses=responses,
        octaves=octaves,
        class_ids=class_ids,
        descriptors=descriptors,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
        ppi=ppi,
        metadata={
            "source_path": str(image_path),
            "image_policy": "native_grayscale_no_enhancement",
            "manifest_ppi": ppi,
            "backend": backend,
            "lanczos_policy": "one_exact_2x_resize_for_scale_indices_0_and_1",
            "lanczos_interpolation": "cv2.INTER_LANCZOS4",
            "coordinate_policy": "detector_reports_i0_i1_back_in_native_pixel_coordinates",
            "keypoint_size_mapping": "opencv_size_equals_2_times_output_integration_sigma",
            "effective_gaussian_support_recorded_separately": True,
            "scale_mapping_records": scale_records,
            "harriszplus_scale_indices": [
                int(keypoint.scale_index) for keypoint in retained_detector_keypoints
            ],
            "harriszplus_source_indices": [
                int(keypoint.source_index) for keypoint in retained_detector_keypoints
            ],
            "opencv_octave_policy": "zero_for_all_native_supplied_keypoints",
            "opencv_class_id_policy": "harriszplus_scale_index",
            "descriptor_mode": "rootsift_existing_implementation_unchanged",
            "prepare_total_ms": None,
        },
    )
    representation_hash = representation_sha256(representation)
    diagnostics: dict[str, Any] = {
        "image_policy": "native_grayscale_no_downsampling_no_enhancement",
        "native_width": int(gray.shape[1]),
        "native_height": int(gray.shape[0]),
        "ppi": ppi,
        "ppi_source": "manifest",
        "lanczos_interpolation": "cv2.INTER_LANCZOS4",
        "lanczos_scale_factor": 2,
        "lanczos_call_count": 1,
        "lanczos_scale_indices": [0, 1],
        "detector_backend": backend,
        "detector_keypoint_count": len(detected),
        "max_keypoints": int(config.max_keypoints),
        "requested_keypoints": len(supplied_keypoints),
        "computed_keypoints": len(computed_keypoints),
        "descriptor_count": int(descriptors.shape[0]),
        "dropped_descriptor_count": non_finite_row_count,
        "non_finite_descriptor_row_count": non_finite_row_count,
        "non_finite_descriptor_value_count": non_finite_value_count,
        "descriptor_dimension": int(descriptors.shape[1]),
        "descriptor_dtype": str(descriptors.dtype),
        "descriptor_mode": "rootsift",
        "rootsift_function": "fingerprint_benchmark.sift.descriptors.rootsift",
        "sift_descriptor_compute_ms": descriptor_compute_ms,
        "rootsift_ms": rootsift_ms,
        "keypoint_size_mapping": "size = 2 * output_integration_sigma",
        "effective_gaussian_support_recorded_separately": True,
        "scale_mapping_records": scale_records,
        "harriszplus_scale_indices": [
            int(keypoint.scale_index) for keypoint in retained_detector_keypoints
        ],
        "harriszplus_source_indices": [
            int(keypoint.source_index) for keypoint in retained_detector_keypoints
        ],
        "opencv_octave_policy": "zero_for_all_native_supplied_keypoints",
        "opencv_class_id_policy": "harriszplus_scale_index",
        "representation_sha256": representation_hash,
        **detector_diagnostics,
        **orientation_diagnostics,
        **timings,
    }
    prepare_total_ms = _elapsed(started)
    timings["prepare_total_ms"] = prepare_total_ms
    diagnostics["prepare_total_ms"] = prepare_total_ms
    representation.metadata["prepare_total_ms"] = prepare_total_ms
    return representation, diagnostics, prepare_total_ms


def _manifest_ppi(
    image_metadata: Mapping[str, Any],
    started: int,
    timings: dict[str, Any],
) -> float:
    try:
        ppi = float(image_metadata["ppi"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MethodExecutionError(
            "missing_or_invalid_ppi",
            "HarrisZ+ preparation requires finite positive manifest PPI.",
            method_internal_ms=_elapsed(started),
            diagnostics={**timings, "ppi_source": "manifest"},
        ) from exc
    if not np.isfinite(ppi) or ppi <= 0.0:
        _fail(
            "missing_or_invalid_ppi",
            "HarrisZ+ preparation requires finite positive manifest PPI.",
            started,
            timings,
            ppi=ppi,
            ppi_source="manifest",
        )
    return ppi


def _validate_adapter_config(config: HarrisZPlusConfig) -> None:
    if str(config.lanczos_interpolation) != "INTER_LANCZOS4":
        raise ValueError("The frozen adapter requires OpenCV INTER_LANCZOS4.")
    if tuple(int(value) for value in config.doubled_scale_indices) != (0, 1):
        raise ValueError("Only HarrisZ+ scale indices 0 and 1 use the exact doubled image.")
    if str(config.descriptor_mode) != "rootsift":
        raise ValueError("The HarrisZ+ method requires unchanged RootSIFT descriptors.")
    if int(config.minimum_descriptors) < 2:
        raise ValueError("At least two descriptors are required for mutual KNN matching.")
    if str(config.keypoint_size_formula) != "2_times_output_integration_sigma":
        raise ValueError("OpenCV keypoint size must equal twice output integration sigma.")


def _validate_detector_keypoints(
    keypoints: tuple[object, ...],
    started: int,
    timings: dict[str, Any],
) -> None:
    for index, item in enumerate(keypoints):
        values = (
            float(getattr(item, "x")),
            float(getattr(item, "y")),
            float(getattr(item, "response")),
            float(getattr(item, "sigma")),
            _integration_sigma(item),
            float(getattr(item, "effective_support_diameter")),
            float(getattr(item, "size")),
        )
        if not all(np.isfinite(value) for value in values):
            _fail(
                "non_finite_detector_keypoint",
                "Detector keypoint contains a non-finite coordinate, response, or scale.",
                started,
                timings,
                keypoint_index=index,
            )
        if values[3] <= 0.0 or values[4] <= 0.0 or values[5] <= 0.0 or values[6] <= 0.0:
            _fail(
                "invalid_detector_keypoint_scale",
                "Detector sigma, effective support, and OpenCV size must be positive.",
                started,
                timings,
                keypoint_index=index,
            )
        if not np.isclose(values[6], 2.0 * values[4], rtol=0.0, atol=1e-6):
            _fail(
                "keypoint_size_mapping_violation",
                "OpenCV keypoint size must equal twice the output integration sigma.",
                started,
                timings,
                keypoint_index=index,
                output_integration_sigma=values[4],
                opencv_keypoint_size=values[6],
            )
        angle = getattr(item, "angle", None)
        if angle is not None and float(angle) == -1.0:
            _fail(
                "detector_orientation_leak",
                "Detector keypoints must receive adapter orientation, never OpenCV angle=-1.",
                started,
                timings,
                keypoint_index=index,
            )


def _scale_mapping_records(keypoints: tuple[object, ...]) -> list[dict[str, Any]]:
    unique: dict[tuple[int, float, float, float, float], dict[str, Any]] = {}
    for item in keypoints:
        scale_index = int(getattr(item, "scale_index"))
        sigma = float(getattr(item, "sigma"))
        integration_sigma = _integration_sigma(item)
        support = float(getattr(item, "effective_support_diameter"))
        size = float(getattr(item, "size"))
        key = (scale_index, sigma, integration_sigma, support, size)
        unique[key] = {
            "scale_index": scale_index,
            "harriszplus_differentiation_sigma": sigma,
            "output_integration_sigma": integration_sigma,
            "effective_gaussian_support_diameter": support,
            "opencv_keypoint_size": size,
            "mapping_formula": "opencv_size = 2 * output_integration_sigma",
        }
    return [unique[key] for key in sorted(unique)]


def _integration_sigma(keypoint: object) -> float:
    if hasattr(keypoint, "output_integration_sigma"):
        return float(getattr(keypoint, "output_integration_sigma"))
    return float(getattr(keypoint, "integration_sigma"))


def _enforce_determinism(config: HarrisZPlusConfig) -> None:
    seed = int(config.rng_seed)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    if _backend_kind(str(config.backend)) != "cuda":
        return
    try:
        import torch
    except ImportError as exc:
        raise MethodExecutionError(
            "missing_torch_cuda_dependency",
            "The CUDA backend requires the pinned PyTorch optional dependency.",
        ) from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda_timing_start(config: HarrisZPlusConfig) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise MethodExecutionError(
            "missing_torch_cuda_dependency",
            "The CUDA backend requires the pinned PyTorch optional dependency.",
        ) from exc
    if not torch.cuda.is_available():
        raise MethodExecutionError(
            "cuda_unavailable",
            "The configured HarrisZ+ CUDA backend has no available CUDA device.",
        )
    device = torch.device(config.device or "cuda")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return torch, device


def _backend_kind(value: str) -> str:
    aliases = {
        "cuda": "cuda",
        "harriszplus_cuda": "cuda",
        "reference_cpu": "reference_cpu",
        "harriszplus_reference_cpu": "reference_cpu",
    }
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported HarrisZ+ backend: {value!r}.") from exc


def _cuda_synchronize(runtime: tuple[Any, Any], *, suppress_errors: bool = False) -> None:
    torch, device = runtime
    try:
        torch.cuda.synchronize(device)
    except Exception:
        if not suppress_errors:
            raise


def _cuda_peak_memory(runtime: tuple[Any, Any]) -> dict[str, int]:
    torch, device = runtime
    return {
        "peak_vram_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_vram_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _merge_detector_timings(
    timings: dict[str, Any],
    detector_timings: dict[str, Any],
    *,
    backend: str,
) -> None:
    timings["detector_reported_timings"] = detector_timings
    aliases = {
        "detector_gpu_kernel_ms": (
            "detector_gpu_kernel_ms",
            "gpu_kernel_ms",
            "kernel_ms",
        ),
        "candidate_transfer_ms": ("candidate_transfer_ms", "transfer_ms"),
        "selection_cpu_ms": ("selection_cpu_ms", "selection_ms"),
    }
    for destination, candidates in aliases.items():
        for candidate in candidates:
            if candidate in detector_timings and detector_timings[candidate] is not None:
                timings[destination] = float(detector_timings[candidate])
                break
    if timings["candidate_transfer_ms"] is None:
        candidate_transfer_seconds = detector_timings.get(
            "candidate_device_to_host_seconds"
        )
        if candidate_transfer_seconds is not None:
            timings["candidate_transfer_ms"] = 1000.0 * float(
                candidate_transfer_seconds
            )
    if backend == "cuda" and timings["detector_gpu_kernel_ms"] is None:
        # Compatibility fallback for third-party detector shims that do not
        # expose the core detector's summed CUDA-event timing.
        synchronized_compute_fields = (
            "base_gradients_seconds",
            "dense_response_seconds",
            "subpixel_and_eigen_seconds",
        )
        compute_seconds = [
            float(detector_timings[field])
            for field in synchronized_compute_fields
            if detector_timings.get(field) is not None
        ]
        if compute_seconds:
            timings["detector_gpu_kernel_ms"] = 1000.0 * sum(compute_seconds)
    if timings["selection_cpu_ms"] is None:
        selection_second_fields = (
            (
                "scale_suppression_seconds",
                "duplicate_removal_seconds",
                "uniform_selection_seconds",
            )
            if backend == "cuda"
            else (
                "local_candidate_materialization_seconds",
                "scale_suppression_seconds",
                "subpixel_and_eigen_seconds",
                "duplicate_removal_seconds",
                "uniform_selection_seconds",
            )
        )
        present = [
            float(detector_timings[field])
            for field in selection_second_fields
            if detector_timings.get(field) is not None
        ]
        if present:
            timings["selection_cpu_ms"] = 1000.0 * sum(present)
    if backend != "cuda":
        timings["detector_gpu_wall_ms"] = None
        timings["detector_gpu_kernel_ms"] = None
        timings["candidate_transfer_ms"] = 0.0


def _empty_prepare_timings() -> dict[str, Any]:
    return {
        "image_load_ms": None,
        "lanczos_ms": None,
        "detector_gpu_wall_ms": None,
        "detector_gpu_kernel_ms": None,
        "detector_cpu_wall_ms": None,
        "candidate_transfer_ms": None,
        "selection_cpu_ms": None,
        "orientation_cpu_ms": None,
        "descriptor_cpu_ms": None,
        "prepare_total_ms": None,
        "peak_vram_allocated": 0,
        "peak_vram_reserved": 0,
    }


def _fail(
    code: str,
    message: str,
    started: int,
    timings: dict[str, Any],
    **diagnostics: Any,
) -> None:
    elapsed = _elapsed(started)
    timings["prepare_total_ms"] = elapsed
    raise MethodExecutionError(
        code,
        message,
        method_internal_ms=elapsed,
        diagnostics={**_json_native(timings), **_json_native(diagnostics)},
    )


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_native(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _elapsed(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0
