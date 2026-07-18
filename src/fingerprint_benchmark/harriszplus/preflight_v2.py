"""Pre-declared HarrisZ+ CPU/CUDA semantic validation contract v2.

This module is validation/provenance code only.  It imports the unchanged v1
candidate detector, descriptor, matcher, and geometry implementation and
changes no score-producing parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from ..contract import (
    COMPARISON_FAILURE,
    OK,
    PREPARE_A_FAILURE,
    PREPARE_B_FAILURE,
    MethodExecutionError,
)
from ..hashing import canonical_json_bytes, file_sha256, stable_config_hash, stable_hash
from ..manifest import PairRecord, read_pair_manifest
from .adapter import HarrisZPlusGeometricAdapter
from .config import HarrisZPlusConfig
from .cuda_detector import detect_harriszplus_cuda
from .preflight import (
    DATASETS,
    EXPECTED_SELECTION_SHA256,
    RESPONSE_ATOL,
    RESPONSE_MAX_ABSOLUTE_DELTA,
    RESPONSE_RTOL,
    MINIMUM_RESPONSE_PIXEL_COVERAGE,
    SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE,
    _canonical_validation_keypoint_index_order,
    _configure_and_describe_cuda,
    _effective_runner_config,
    _keypoint_record,
    _keypoints,
    _ordered_response_maps,
    _spearman_rank_correlation,
    _without_timing_fields,
    detector_result_sha256,
    load_and_verify_selection,
    representation_sha256,
    synthetic_suite,
    validate_authoritative_manifest,
)
from .provenance import implementation_source_hashes
from .reference_cpu import detect_harriszplus_cpu


METHOD_NAME = "harriszplus_rootsift_geometric"
METHOD_VERSION = "harriszplus-rootsift-geometric-v2"
PARENT_FAILED_CANDIDATE = "harriszplus-rootsift-geometric-v1"
PREFLIGHT_CONTRACT = "engineering-preflight-v2"
PREFLIGHT_SCHEMA_VERSION = "harriszplus-engineering-preflight-v2"
DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
METHOD_RESULTS_RELATIVE = Path("results/harriszplus_rootsift_geometric_v2")
SOURCE_PILOT_RELATIVE = Path("results/pilots/sourceafis_joint_500_v1")
CONTRACT_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_contract_v2.json"
)
PASS_RELATIVE = METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_pass.json"
FAILURE_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_failure.json"
)

EXPECTED_CONTRACT_SHA256 = (
    "3dfa83653375763eedc9b6df40168d1d7b686d727613af54390f53c16540db9a"
)
EXPECTED_V1_FAILURE_SHA256 = (
    "9b822a9a2bc0e67e8b0bf3d9658b55d84865bb6c64f3f82a5900debebbb8cd42"
)
EXPECTED_CANDIDATE_CONFIG_SHA256 = (
    "521c23e27e5e3b0336aeeb755c8fa5baf32f510fecb9e6d2244c2fc204ba87fa"
)
THRESHOLD = 4
MAX_KEYPOINTS = 3000
SPATIAL_TOLERANCE_ORIGINAL_PX = 0.5
RELATIVE_SCALE_TOLERANCE = 0.01
MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION = 0.995
MINIMUM_RESPONSE_RANK_SPEARMAN = 0.9999
COUNT_RELATIVE_TOLERANCE = 0.005
COUNT_MINIMUM_ABSOLUTE_TOLERANCE = 2
MAXIMUM_RAW_SCORE_DELTA = 1
MINIMUM_EXACT_RAW_SCORES = 9
VALID_STATUSES = (OK, PREPARE_A_FAILURE, PREPARE_B_FAILURE, COMPARISON_FAILURE)

# Independent tripwire captured before v2 validation code was added.  These are
# the v1 score-producing source identities and must remain byte-exact.
EXPECTED_V1_ALGORITHM_SOURCE_SHA256 = {
    "adapter.py": "125629842f012ce1b7b13b9c2ea4628d0a086cb4e7c6a7073f5d832bef6fa130",
    "config.py": "8e543e0ae61e97fe8e11b546f564139271bc90fef0987cc38ea42092e48032ab",
    "cuda_detector.py": "fa28d89a5fdb7f503824a6ce05dcf534793e4c1d5ed8f0c6f2b17766de65c52c",
    "extractor.py": "513e86d9e4374b115ebb90811a79a8c10a1755897701beb08c7ea5bffd21af34",
    "kernels.py": "e4fd03697f3fe07041d4e685d1b749c4601f82318039b1febade8b42ca9781a9",
    "orientation.py": "93eb4bf385b3277862331d30ae8fae1ea89fce87e1ac7edbb91ca7cc74aa8ff6",
    "provenance.py": "98e5716358eab0a887b1190f5dc6a6b476d1cc7a28ca7fc5081a3e264abfc41c",
    "reference_cpu.py": "01533331da30c6aa58bd79369e2ce9a30c9c31a11ddcab3ccd9ceb158d84c983",
    "selection.py": "8a7c89f3b6a25d41d08d90b2cd0a7e17488d7d74ae92103d342b4d194c9fc166",
    "types.py": "14c50df0db4840c1c3b625c485e4161b82c772a5543e343d6fba38c411ba6c26",
}


class HarrisZPlusPreflightV2Error(ValueError):
    """Raised when v2 evidence cannot be produced or immutably published."""


def allowed_absolute_delta(first: int, second: int) -> int:
    """Return the frozen hybrid count tolerance."""

    first = int(first)
    second = int(second)
    if first < 0 or second < 0:
        raise ValueError("Candidate counts must be non-negative.")
    return max(
        COUNT_MINIMUM_ABSOLUTE_TOLERANCE,
        math.ceil(COUNT_RELATIVE_TOLERANCE * max(first, second)),
    )


def count_equivalence(first: int, second: int) -> dict[str, Any]:
    """Evaluate the v2 count gate and retain the legacy ratio as diagnostics."""

    cpu_count = int(first)
    cuda_count = int(second)
    maximum = max(cpu_count, cuda_count)
    ratio = 1.0 if maximum == 0 else min(cpu_count, cuda_count) / maximum
    zero_mismatch = (cpu_count == 0) != (cuda_count == 0)
    allowed = allowed_absolute_delta(cpu_count, cuda_count)
    absolute = abs(cpu_count - cuda_count)
    return {
        "cpu": cpu_count,
        "cuda": cuda_count,
        "absolute_delta": absolute,
        "allowed_absolute_delta": allowed,
        "minimum_to_maximum_ratio_diagnostic_only": ratio,
        "legacy_ratio_is_gate": False,
        "zero_versus_nonzero": zero_mismatch,
        "passed": bool(not zero_mismatch and absolute <= allowed),
    }


def compare_candidate_counts_v2(cpu: Any, cuda: Any) -> dict[str, Any]:
    """Apply the frozen v2 rule to every available intermediate count."""

    cpu_diagnostics = getattr(cpu, "diagnostics", {})
    cuda_diagnostics = getattr(cuda, "diagnostics", {})
    if not isinstance(cpu_diagnostics, Mapping) or not isinstance(
        cuda_diagnostics, Mapping
    ):
        return {
            "available": False,
            "passed": False,
            "reason": "diagnostics_not_mappings",
        }
    per_scale_keys = (
        "candidates_before_mask",
        "candidates_after_mask",
        "candidates_after_local_maxima",
        "candidates_after_scale_suppression",
        "candidates_after_eigen_ratio",
    )
    aggregate_keys = (*per_scale_keys, "candidates_after_duplicate_removal")
    missing: list[str] = []
    aggregate: dict[str, Any] = {}
    passed = True
    for key in aggregate_keys:
        if key not in cpu_diagnostics or key not in cuda_diagnostics:
            missing.append(key)
            passed = False
            continue
        row = count_equivalence(cpu_diagnostics[key], cuda_diagnostics[key])
        aggregate[key] = row
        passed = passed and row["passed"]

    cpu_scales = cpu_diagnostics.get("scales")
    cuda_scales = cuda_diagnostics.get("scales")
    per_scale: dict[str, Any] = {}
    if not isinstance(cpu_scales, Mapping) or not isinstance(cuda_scales, Mapping):
        passed = False
    else:
        names = sorted(set(cpu_scales) | set(cuda_scales), key=int)
        for name in names:
            cpu_scale = cpu_scales.get(name, {})
            cuda_scale = cuda_scales.get(name, {})
            cpu_counts = (
                cpu_scale.get("counts", {}) if isinstance(cpu_scale, Mapping) else {}
            )
            cuda_counts = (
                cuda_scale.get("counts", {})
                if isinstance(cuda_scale, Mapping)
                else {}
            )
            rows: dict[str, Any] = {}
            for key in per_scale_keys:
                if key not in cpu_counts or key not in cuda_counts:
                    rows[key] = {"passed": False, "reason": "missing"}
                    passed = False
                    continue
                row = count_equivalence(cpu_counts[key], cuda_counts[key])
                rows[key] = row
                passed = passed and row["passed"]
            per_scale[str(name)] = rows
    return {
        "available": True,
        "rule": "max(2, ceil(0.005 * max(cpu_count, cuda_count)))",
        "zero_versus_nonzero_always_fails": True,
        "missing_aggregate_fields": missing,
        "aggregate": aggregate,
        "per_scale": per_scale,
        "passed": bool(passed),
    }


def _response_map_comparison(
    cpu: Any,
    cuda: Any,
    *,
    minimum_pixel_coverage: float,
) -> dict[str, Any]:
    cpu_maps = _ordered_response_maps(cpu)
    cuda_maps = _ordered_response_maps(cuda)
    if not cpu_maps or [index for index, _ in cpu_maps] != [
        index for index, _ in cuda_maps
    ]:
        raise HarrisZPlusPreflightV2Error(
            "CPU/CUDA validation requires equal non-empty per-scale response maps."
        )
    rows: list[dict[str, Any]] = []
    passed = True
    for (scale_index, cpu_map), (_, cuda_map) in zip(
        cpu_maps, cuda_maps, strict=True
    ):
        cpu_array = np.asarray(cpu_map, dtype=np.float64)
        cuda_array = np.asarray(cuda_map, dtype=np.float64)
        if cpu_array.shape != cuda_array.shape:
            raise HarrisZPlusPreflightV2Error(
                f"Response-map shape mismatch at scale {scale_index}."
            )
        finite = bool(
            np.isfinite(cpu_array).all() and np.isfinite(cuda_array).all()
        )
        delta = np.abs(cpu_array - cuda_array)
        close_mask = np.isclose(
            cpu_array,
            cuda_array,
            atol=RESPONSE_ATOL,
            rtol=RESPONSE_RTOL,
            equal_nan=False,
        )
        pixel_count = int(cpu_array.size)
        close_count = int(np.count_nonzero(close_mask)) if finite else 0
        close_fraction = 1.0 if pixel_count == 0 else close_count / pixel_count
        max_delta = float(np.max(delta)) if delta.size else 0.0
        mean_delta = float(np.mean(delta)) if delta.size else 0.0
        rmse = (
            float(np.sqrt(np.mean(np.square(cpu_array - cuda_array))))
            if delta.size
            else 0.0
        )
        cpu_rms = (
            float(np.sqrt(np.mean(np.square(cpu_array)))) if cpu_array.size else 0.0
        )
        normalized_rmse = rmse / max(cpu_rms, np.finfo(np.float64).eps)
        statistic_rows: dict[str, Any] = {}
        statistics_passed = finite
        for label, cpu_value, cuda_value in (
            ("minimum", np.min(cpu_array), np.min(cuda_array)),
            ("maximum", np.max(cpu_array), np.max(cuda_array)),
            ("mean", np.mean(cpu_array), np.mean(cuda_array)),
            (
                "sample_stddev",
                np.std(cpu_array, ddof=1) if cpu_array.size > 1 else 0.0,
                np.std(cuda_array, ddof=1) if cuda_array.size > 1 else 0.0,
            ),
        ):
            cpu_number = float(cpu_value)
            cuda_number = float(cuda_value)
            within = bool(
                finite
                and math.isclose(
                    cpu_number,
                    cuda_number,
                    abs_tol=RESPONSE_ATOL,
                    rel_tol=RESPONSE_RTOL,
                )
            )
            statistic_rows[label] = {
                "cpu": cpu_number,
                "cuda": cuda_number,
                "absolute_delta": abs(cpu_number - cuda_number),
                "within_tolerance": within,
            }
            statistics_passed = statistics_passed and within
        row_passed = bool(
            finite
            and close_fraction >= minimum_pixel_coverage
            and max_delta <= RESPONSE_MAX_ABSOLUTE_DELTA
            and statistics_passed
        )
        rows.append(
            {
                "scale_index": int(scale_index),
                "shape": list(cpu_array.shape),
                "finite": finite,
                "maximum_absolute_difference": max_delta,
                "mean_absolute_difference": mean_delta,
                "normalized_rmse": normalized_rmse,
                "sign_disagreement_count_at_0": int(
                    np.count_nonzero((cpu_array > 0.0) != (cuda_array > 0.0))
                ),
                "mask_disagreement_count_at_0.31": int(
                    np.count_nonzero((cpu_array > 0.31) != (cuda_array > 0.31))
                ),
                "pixel_count": pixel_count,
                "allclose_pixel_count": close_count,
                "allclose_pixel_fraction": close_fraction,
                "minimum_allclose_pixel_fraction": minimum_pixel_coverage,
                "maximum_allowed_absolute_delta": RESPONSE_MAX_ABSOLUTE_DELTA,
                "statistics": statistic_rows,
                "passed": row_passed,
            }
        )
        passed = passed and row_passed
    return {
        "response_atol": RESPONSE_ATOL,
        "response_rtol": RESPONSE_RTOL,
        "response_max_absolute_delta": RESPONSE_MAX_ABSOLUTE_DELTA,
        "minimum_pixel_coverage": minimum_pixel_coverage,
        "tolerances_identical_to_v1": True,
        "per_scale": rows,
        "passed": bool(passed),
    }


def _relative_scale_delta(first: float, second: float) -> float:
    denominator = max(abs(first), abs(second))
    return 0.0 if denominator == 0.0 else abs(first - second) / denominator


def _directional_keypoint_matches(
    source: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, float, float]]:
    unused = set(range(len(target)))
    matches: list[tuple[int, int, float, float]] = []
    for source_index, point in enumerate(source):
        candidates: list[tuple[float, float, float, int]] = []
        for target_index in unused:
            other = target[target_index]
            spatial = math.hypot(
                float(point["x"]) - float(other["x"]),
                float(point["y"]) - float(other["y"]),
            )
            scale_delta = _relative_scale_delta(
                float(point["sigma"]), float(other["sigma"])
            )
            if (
                spatial <= SPATIAL_TOLERANCE_ORIGINAL_PX
                and scale_delta <= RELATIVE_SCALE_TOLERANCE
            ):
                candidates.append(
                    (
                        spatial,
                        scale_delta,
                        abs(float(point["response"]) - float(other["response"])),
                        target_index,
                    )
                )
        if candidates:
            spatial, scale_delta, _, target_index = min(candidates)
            unused.remove(target_index)
            matches.append((source_index, target_index, spatial, scale_delta))
    return matches


def compare_final_keypoints_v2(cpu: Any, cuda: Any) -> dict[str, Any]:
    """Match final lists in both directions at the frozen semantic tolerances."""

    cpu_points = [_keypoint_record(point) for point in _keypoints(cpu)]
    cuda_points = [_keypoint_record(point) for point in _keypoints(cuda)]
    forward = _directional_keypoint_matches(cpu_points, cuda_points)
    reverse = _directional_keypoint_matches(cuda_points, cpu_points)
    forward_fraction = (
        1.0 if not cpu_points and not cuda_points else len(forward) / len(cpu_points)
        if cpu_points
        else 0.0
    )
    reverse_fraction = (
        1.0 if not cpu_points and not cuda_points else len(reverse) / len(cuda_points)
        if cuda_points
        else 0.0
    )
    bidirectional_fraction = min(forward_fraction, reverse_fraction)
    cpu_order = _canonical_validation_keypoint_index_order(cpu_points)
    cuda_order = _canonical_validation_keypoint_index_order(cuda_points)
    cpu_rank = {raw: rank for rank, raw in enumerate(cpu_order)}
    cuda_rank = {raw: rank for rank, raw in enumerate(cuda_order)}
    rank_pairs = [
        (cpu_rank[cpu_index], cuda_rank[cuda_index])
        for cpu_index, cuda_index, _, _ in forward
    ]
    spearman = _spearman_rank_correlation(rank_pairs)
    count_gate = count_equivalence(len(cpu_points), len(cuda_points))
    matched_gate = (
        bidirectional_fraction >= MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION
    )
    spearman_gate = spearman >= MINIMUM_RESPONSE_RANK_SPEARMAN
    return {
        "cpu_keypoint_count": len(cpu_points),
        "cuda_keypoint_count": len(cuda_points),
        "count_equivalence": count_gate,
        "spatial_tolerance_original_pixels": SPATIAL_TOLERANCE_ORIGINAL_PX,
        "relative_scale_tolerance": RELATIVE_SCALE_TOLERANCE,
        "cpu_to_cuda_matched_count": len(forward),
        "cuda_to_cpu_matched_count": len(reverse),
        "cpu_to_cuda_matched_fraction": forward_fraction,
        "cuda_to_cpu_matched_fraction": reverse_fraction,
        "bidirectional_matched_fraction": bidirectional_fraction,
        "minimum_bidirectional_matched_fraction": (
            MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION
        ),
        "maximum_matched_spatial_delta_original_pixels": max(
            (row[2] for row in forward), default=0.0
        ),
        "maximum_matched_relative_scale_delta": max(
            (row[3] for row in forward), default=0.0
        ),
        "response_rank_spearman": spearman,
        "minimum_response_rank_spearman": MINIMUM_RESPONSE_RANK_SPEARMAN,
        "count_gate_passed": count_gate["passed"],
        "bidirectional_matching_passed": matched_gate,
        "spearman_passed": spearman_gate,
        "passed": bool(count_gate["passed"] and matched_gate and spearman_gate),
    }


def compare_detector_results_v2(
    cpu: Any,
    cuda: Any,
    *,
    minimum_response_pixel_coverage: float,
) -> dict[str, Any]:
    responses = _response_map_comparison(
        cpu,
        cuda,
        minimum_pixel_coverage=minimum_response_pixel_coverage,
    )
    candidates = compare_candidate_counts_v2(cpu, cuda)
    keypoints = compare_final_keypoints_v2(cpu, cuda)
    return {
        "response_maps": responses,
        "intermediate_candidate_counts": candidates,
        "final_keypoint_equivalence": keypoints,
        "passed": bool(
            responses["passed"] and candidates["passed"] and keypoints["passed"]
        ),
    }


def _detector_absolute_conditions(
    result: Any,
    *,
    expected_backend: str,
    image_shape: Sequence[int],
) -> dict[str, Any]:
    height, width = int(image_shape[0]), int(image_shape[1])
    points = [_keypoint_record(point) for point in _keypoints(result)]
    finite = True
    for _, response in _ordered_response_maps(result):
        finite = finite and bool(np.isfinite(np.asarray(response)).all())
    coordinates = all(
        0.0 <= float(point["x"]) < width and 0.0 <= float(point["y"]) < height
        for point in points
    )
    positive_scales = all(
        float(getattr(point, "sigma")) > 0.0
        and float(getattr(point, "integration_sigma")) > 0.0
        and float(getattr(point, "size")) > 0.0
        for point in _keypoints(result)
    )
    backend_exact = getattr(result, "backend", None) == expected_backend
    cap_ok = len(points) <= MAX_KEYPOINTS
    return {
        "finite": bool(finite),
        "coordinates_within_image_bounds": coordinates,
        "positive_scales": positive_scales,
        "keypoint_cap_ok": cap_ok,
        "backend": getattr(result, "backend", None),
        "expected_backend": expected_backend,
        "hidden_cpu_fallback_absent": backend_exact,
        "passed": bool(
            finite and coordinates and positive_scales and cap_ok and backend_exact
        ),
    }


def _representation_record(outcome: Any) -> dict[str, Any]:
    representation = outcome.representation
    payload = representation.payload
    descriptors = np.asarray(payload.descriptors)
    points = np.asarray(payload.points)
    sizes = np.asarray(payload.sizes)
    finite = bool(
        np.isfinite(descriptors).all()
        and np.isfinite(points).all()
        and np.isfinite(sizes).all()
    )
    width = int(payload.width)
    height = int(payload.height)
    coordinates = bool(
        points.ndim == 2
        and points.shape[1] == 2
        and np.all(points[:, 0] >= 0.0)
        and np.all(points[:, 0] < width)
        and np.all(points[:, 1] >= 0.0)
        and np.all(points[:, 1] < height)
    )
    positive_scales = bool(np.all(sizes > 0.0))
    backend = representation.metadata.get("detector_backend")
    deterministic_diagnostics = _without_timing_fields(outcome.diagnostics)
    return {
        "representation_sha256": representation_sha256(representation),
        "deterministic_diagnostics_sha256": stable_hash(
            deterministic_diagnostics
        ),
        "descriptor_count": int(descriptors.shape[0]),
        "keypoint_count": int(points.shape[0]),
        "descriptors_finite": finite,
        "coordinates_within_image_bounds": coordinates,
        "positive_scales": positive_scales,
        "keypoint_cap_ok": int(points.shape[0]) <= MAX_KEYPOINTS,
        "detector_backend": backend,
        "hidden_cpu_fallback_absent": backend in ("reference_cpu", "cuda"),
        "prepare_total_ms": _finite_number(
            outcome.diagnostics.get("prepare_total_ms")
        ),
        "detector_gpu_wall_ms": _finite_number(
            outcome.diagnostics.get("detector_gpu_wall_ms")
        ),
        "descriptor_cpu_ms": _finite_number(
            outcome.diagnostics.get("descriptor_cpu_ms")
        ),
        "peak_vram_allocated": _finite_number(
            outcome.diagnostics.get("peak_vram_allocated")
        ),
        "peak_vram_reserved": _finite_number(
            outcome.diagnostics.get("peak_vram_reserved")
        ),
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _run_pair_once(
    adapter: HarrisZPlusGeometricAdapter,
    pair: PairRecord,
) -> dict[str, Any]:
    prepare_a = None
    prepare_b = None
    try:
        prepare_a = adapter.prepare(pair.path_a, pair.image_metadata_a())
    except MethodExecutionError as exc:
        return _pair_failure(PREPARE_A_FAILURE, "prepare_a", exc)
    try:
        prepare_b = adapter.prepare(pair.path_b, pair.image_metadata_b())
    except MethodExecutionError as exc:
        return _pair_failure(PREPARE_B_FAILURE, "prepare_b", exc)
    try:
        comparison = adapter.compare(
            prepare_a.representation, prepare_b.representation
        )
    except MethodExecutionError as exc:
        return _pair_failure(COMPARISON_FAILURE, "compare", exc)
    score = int(comparison.raw_score)
    if score < 0 or float(comparison.raw_score) != score:
        raise HarrisZPlusPreflightV2Error(
            "HarrisZ+ raw score must be a non-negative integer."
        )
    a_record = _representation_record(prepare_a)
    b_record = _representation_record(prepare_b)
    diagnostics = _without_timing_fields(comparison.diagnostics)
    record = {
        "status": OK,
        "failure_stage": None,
        "raw_score": score,
        "decision_threshold_4": score >= THRESHOLD,
        "prepare_a": a_record,
        "prepare_b": b_record,
        "compare_deterministic_diagnostics_sha256": stable_hash(diagnostics),
        "compare_total_ms": _finite_number(
            comparison.diagnostics.get("compare_total_ms")
        ),
    }
    record["payload_sha256"] = stable_hash(
        {
            "status": record["status"],
            "failure_stage": record["failure_stage"],
            "raw_score": record["raw_score"],
            "decision_threshold_4": record["decision_threshold_4"],
            "prepare_a_sha256": a_record["representation_sha256"],
            "prepare_b_sha256": b_record["representation_sha256"],
            "prepare_a_diagnostics_sha256": a_record[
                "deterministic_diagnostics_sha256"
            ],
            "prepare_b_diagnostics_sha256": b_record[
                "deterministic_diagnostics_sha256"
            ],
            "compare_diagnostics_sha256": record[
                "compare_deterministic_diagnostics_sha256"
            ],
        }
    )
    return record


def _pair_failure(
    status: str,
    stage: str,
    exc: MethodExecutionError,
) -> dict[str, Any]:
    record = {
        "status": status,
        "failure_stage": stage,
        "error_code": exc.error_code,
        "error_message": exc.message,
        "raw_score": None,
        "decision_threshold_4": False,
        "prepare_a": None,
        "prepare_b": None,
    }
    record["payload_sha256"] = stable_hash(record)
    return record


def _semantic_pair_comparison(
    cpu: Mapping[str, Any],
    cuda: Mapping[str, Any],
) -> dict[str, Any]:
    status_equal = cpu["status"] == cuda["status"]
    failure_stage_equal = cpu["failure_stage"] == cuda["failure_stage"]
    decision_equal = cpu["decision_threshold_4"] == cuda["decision_threshold_4"]
    if cpu["raw_score"] is None or cuda["raw_score"] is None:
        raw_delta = None
        raw_within = cpu["raw_score"] == cuda["raw_score"]
        raw_exact = raw_within
    else:
        raw_delta = abs(int(cpu["raw_score"]) - int(cuda["raw_score"]))
        raw_within = raw_delta <= MAXIMUM_RAW_SCORE_DELTA
        raw_exact = raw_delta == 0
    descriptor_rows: dict[str, Any] = {}
    descriptor_passed = True
    for side in ("prepare_a", "prepare_b"):
        cpu_prepare = cpu.get(side)
        cuda_prepare = cuda.get(side)
        if not isinstance(cpu_prepare, Mapping) or not isinstance(
            cuda_prepare, Mapping
        ):
            row = {
                "available": False,
                "passed": cpu_prepare is None and cuda_prepare is None,
            }
        else:
            row = count_equivalence(
                int(cpu_prepare["descriptor_count"]),
                int(cuda_prepare["descriptor_count"]),
            )
            row["available"] = True
        descriptor_rows[side] = row
        descriptor_passed = descriptor_passed and row["passed"]
    return {
        "status_equal": status_equal,
        "failure_stage_equal": failure_stage_equal,
        "decision_threshold_4_equal": decision_equal,
        "raw_score_absolute_delta": raw_delta,
        "raw_score_within_one_inlier": raw_within,
        "raw_score_exact": raw_exact,
        "descriptor_counts": descriptor_rows,
        "descriptor_counts_passed": descriptor_passed,
        "passed": bool(
            status_equal
            and failure_stage_equal
            and decision_equal
            and raw_within
            and descriptor_passed
        ),
    }


def _cuda_repeat_comparison(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "status": first["status"] == second["status"],
        "failure_stage": first["failure_stage"] == second["failure_stage"],
        "raw_score": first["raw_score"] == second["raw_score"],
        "decision": (
            first["decision_threshold_4"] == second["decision_threshold_4"]
        ),
        "selected_keypoint_and_descriptor_payload": (
            first["payload_sha256"] == second["payload_sha256"]
        ),
    }
    return {
        "first_payload_sha256": first["payload_sha256"],
        "repeat_payload_sha256": second["payload_sha256"],
        "exact_fields": fields,
        "passed": all(fields.values()),
    }


def _load_real_cases(
    *,
    selection: Sequence[Any],
    source_manifest_root: Path,
    data_root: Path,
) -> list[dict[str, Any]]:
    real_cases: list[dict[str, Any]] = []
    sample = selection[:5]
    for dataset in DATASETS:
        manifest_path = source_manifest_root / dataset / "plain_self.csv"
        validate_authoritative_manifest(
            manifest_path,
            dataset=dataset,
            protocol="plain_self",
            selection=selection,
            data_root=data_root,
            require_self=True,
        )
        pairs = read_pair_manifest(manifest_path)
        for selected, pair in zip(sample, pairs[:5], strict=True):
            if selected.identity != (
                pair.subject_id,
                pair.canonical_finger_position,
            ):
                raise HarrisZPlusPreflightV2Error(
                    "Real preflight identity alignment failed."
                )
            image = cv2.imread(str(pair.path_a), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise HarrisZPlusPreflightV2Error(
                    f"Cannot decode real preflight image: {pair.path_a}"
                )
            real_cases.append(
                {
                    "case_id": f"{dataset}:{selected.selection_index}:plain_self",
                    "kind": "real",
                    "dataset": dataset,
                    "ppi": pair.ppi,
                    "path": pair.path_a,
                    "pair": pair,
                    "image": image.astype(np.float32, copy=False),
                    "source_manifest_sha256": file_sha256(manifest_path),
                }
            )
    return real_cases


def _timing_projection(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        rows = [row for row in pair_rows if row["dataset"] == dataset]
        prepares = [
            float(record["prepare_total_ms"])
            for row in rows
            for record in (
                row["cuda_first"].get("prepare_a"),
                row["cuda_first"].get("prepare_b"),
            )
            if isinstance(record, Mapping)
            and record.get("prepare_total_ms") is not None
        ]
        detectors = [
            float(record["detector_gpu_wall_ms"])
            for row in rows
            for record in (
                row["cuda_first"].get("prepare_a"),
                row["cuda_first"].get("prepare_b"),
            )
            if isinstance(record, Mapping)
            and record.get("detector_gpu_wall_ms") is not None
        ]
        descriptors = [
            float(record["descriptor_cpu_ms"])
            for row in rows
            for record in (
                row["cuda_first"].get("prepare_a"),
                row["cuda_first"].get("prepare_b"),
            )
            if isinstance(record, Mapping)
            and record.get("descriptor_cpu_ms") is not None
        ]
        compares = [
            float(row["cuda_first"]["compare_total_ms"])
            for row in rows
            if row["cuda_first"].get("compare_total_ms") is not None
        ]
        pair_median_ms = (
            2.0 * statistics.median(prepares) + statistics.median(compares)
            if prepares and compares
            else None
        )
        by_dataset[dataset] = {
            "prepare_sample_count": len(prepares),
            "median_prepare_ms": statistics.median(prepares) if prepares else None,
            "median_detector_ms": statistics.median(detectors)
            if detectors
            else None,
            "median_descriptor_ms": statistics.median(descriptors)
            if descriptors
            else None,
            "median_compare_ms": statistics.median(compares) if compares else None,
            "projected_500_pair_run_ms": (
                500.0 * pair_median_ms if pair_median_ms is not None else None
            ),
        }
    peak_values = [
        float(record[key])
        for row in pair_rows
        for record in (
            row["cuda_first"].get("prepare_a"),
            row["cuda_first"].get("prepare_b"),
        )
        if isinstance(record, Mapping)
        for key in ("peak_vram_allocated", "peak_vram_reserved")
        if record.get(key) is not None
    ]
    b_ms = by_dataset["sd300b"]["projected_500_pair_run_ms"]
    c_ms = by_dataset["sd300c"]["projected_500_pair_run_ms"]
    all_eight_ms = (
        4.0 * float(b_ms) + 4.0 * float(c_ms)
        if b_ms is not None and c_ms is not None
        else None
    )
    return {
        "information_only_not_correctness_gate": True,
        "cold_pair": True,
        "prepare_operations_per_pair": 2,
        "cross_pair_cache": False,
        "datasets": by_dataset,
        "peak_vram_bytes": max(peak_values, default=None),
        "projected_all_eight_runs_ms": all_eight_ms,
        "projected_all_eight_runs_hours": (
            all_eight_ms / 3_600_000.0 if all_eight_ms is not None else None
        ),
    }


def _algorithm_identity(project_root: Path) -> dict[str, Any]:
    source_hashes = implementation_source_hashes(strict=True)[
        "required_score_producing_sources"
    ]
    algorithm_sources_unchanged = (
        source_hashes == EXPECTED_V1_ALGORITHM_SOURCE_SHA256
    )
    adapter = HarrisZPlusGeometricAdapter(
        HarrisZPlusConfig(backend="cuda", device="cuda:0")
    )
    try:
        runner_config = _effective_runner_config(adapter.metadata())
    finally:
        adapter.close()
    candidate_hash = stable_config_hash(runner_config)
    v1_failure = (
        project_root
        / "results/harriszplus_rootsift_geometric/preflight"
        / "engineering_preflight_failure.json"
    )
    return {
        "algorithm_changed": not algorithm_sources_unchanged,
        "algorithm_source_sha256": source_hashes,
        "expected_v1_algorithm_source_sha256": (
            EXPECTED_V1_ALGORITHM_SOURCE_SHA256
        ),
        "algorithm_sources_byte_exact_to_v1": algorithm_sources_unchanged,
        "candidate_config_sha256": candidate_hash,
        "expected_candidate_config_sha256": EXPECTED_CANDIDATE_CONFIG_SHA256,
        "candidate_config_unchanged": (
            candidate_hash == EXPECTED_CANDIDATE_CONFIG_SHA256
        ),
        "v1_failure_report_sha256": file_sha256(v1_failure),
        "v1_failure_report_unchanged": (
            file_sha256(v1_failure) == EXPECTED_V1_FAILURE_SHA256
        ),
        "validation_or_provenance_code_change_only": True,
    }


def run_engineering_preflight_v2(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Execute the frozen v2 suite and return complete pass/fail evidence."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    contract_path = project_root / CONTRACT_RELATIVE
    if file_sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise HarrisZPlusPreflightV2Error(
            "Frozen engineering_preflight_contract_v2.json changed."
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("status") != "frozen"
        or contract.get("immutable") is not True
        or contract.get("frozen_before_preflight") is not True
    ):
        raise HarrisZPlusPreflightV2Error("The v2 preflight contract is not frozen.")

    identity = _algorithm_identity(project_root)
    if (
        identity["algorithm_changed"]
        or not identity["candidate_config_unchanged"]
        or not identity["v1_failure_report_unchanged"]
    ):
        raise HarrisZPlusPreflightV2Error(
            "Algorithm/config/v1 immutability tripwire failed before v2 preflight."
        )

    selection_path = (
        project_root / SOURCE_PILOT_RELATIVE / "selected_identities.csv"
    )
    source_manifest_root = project_root / SOURCE_PILOT_RELATIVE / "manifests"
    selection = load_and_verify_selection(selection_path)
    real_cases = _load_real_cases(
        selection=selection,
        source_manifest_root=source_manifest_root,
        data_root=data_root,
    )
    b_identities = [
        (
            case["pair"].subject_id,
            case["pair"].canonical_finger_position,
        )
        for case in real_cases
        if case["dataset"] == "sd300b"
    ]
    c_identities = [
        (
            case["pair"].subject_id,
            case["pair"].canonical_finger_position,
        )
        for case in real_cases
        if case["dataset"] == "sd300c"
    ]
    same_five = b_identities == c_identities and len(b_identities) == 5
    cases = [
        {
            "case_id": f"synthetic:{name}",
            "kind": "synthetic",
            "dataset": None,
            "ppi": 1000,
            "path": None,
            "pair": None,
            "image": image,
        }
        for name, image in synthetic_suite().items()
    ] + real_cases

    torch, environment = _configure_and_describe_cuda("cuda:0")
    cpu_config = HarrisZPlusConfig(backend="reference_cpu", device=None)
    cuda_config = HarrisZPlusConfig(backend="cuda", device="cuda:0")
    detector_rows: list[dict[str, Any]] = []
    for case in cases:
        source_u8 = np.ascontiguousarray(
            np.clip(np.rint(case["image"]), 0.0, 255.0), dtype=np.uint8
        )
        image = source_u8.astype(np.float32)
        doubled = cv2.resize(
            source_u8,
            (int(source_u8.shape[1]) * 2, int(source_u8.shape[0]) * 2),
            interpolation=cv2.INTER_LANCZOS4,
        ).astype(np.float32, copy=False)
        cpu = detect_harriszplus_cpu(
            image,
            cpu_config,
            doubled_image=doubled,
            return_response_maps=True,
        )
        torch.cuda.synchronize("cuda:0")
        cuda_first = detect_harriszplus_cuda(
            image,
            cuda_config,
            device="cuda:0",
            doubled_image=doubled,
            return_response_maps=True,
        )
        torch.cuda.synchronize("cuda:0")
        cuda_second = detect_harriszplus_cuda(
            image,
            cuda_config,
            device="cuda:0",
            doubled_image=doubled,
            return_response_maps=True,
        )
        torch.cuda.synchronize("cuda:0")
        comparison = compare_detector_results_v2(
            cpu,
            cuda_first,
            minimum_response_pixel_coverage=(
                SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE
                if case["kind"] == "synthetic"
                else MINIMUM_RESPONSE_PIXEL_COVERAGE
            ),
        )
        cpu_absolute = _detector_absolute_conditions(
            cpu,
            expected_backend="reference_cpu",
            image_shape=image.shape,
        )
        cuda_absolute = _detector_absolute_conditions(
            cuda_first,
            expected_backend="cuda",
            image_shape=image.shape,
        )
        first_hash = detector_result_sha256(cuda_first)
        repeat_hash = detector_result_sha256(cuda_second)
        repeat_exact = first_hash == repeat_hash
        flat_zero = (
            len(_keypoints(cpu)) == 0 and len(_keypoints(cuda_first)) == 0
            if case["case_id"] == "synthetic:flat"
            else None
        )
        passed = bool(
            comparison["passed"]
            and cpu_absolute["passed"]
            and cuda_absolute["passed"]
            and repeat_exact
            and (flat_zero is not False)
        )
        detector_rows.append(
            {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "dataset": case["dataset"],
                "ppi": case["ppi"],
                "path": str(case["path"]) if case["path"] is not None else None,
                "image_shape": list(image.shape),
                "cpu_keypoint_count": len(_keypoints(cpu)),
                "cuda_keypoint_count": len(_keypoints(cuda_first)),
                "cpu_absolute_conditions": cpu_absolute,
                "cuda_absolute_conditions": cuda_absolute,
                "cpu_cuda": comparison,
                "cuda_first_payload_sha256": first_hash,
                "cuda_repeat_payload_sha256": repeat_hash,
                "cuda_repeat_exact": repeat_exact,
                "flat_zero_keypoints": flat_zero,
                "passed": passed,
            }
        )

    cpu_adapter = HarrisZPlusGeometricAdapter(cpu_config)
    cuda_adapter = HarrisZPlusGeometricAdapter(cuda_config)
    pair_rows: list[dict[str, Any]] = []
    try:
        for case in real_cases:
            pair = case["pair"]
            cpu_run = _run_pair_once(cpu_adapter, pair)
            cuda_first = _run_pair_once(cuda_adapter, pair)
            cuda_second = _run_pair_once(cuda_adapter, pair)
            semantic = _semantic_pair_comparison(cpu_run, cuda_first)
            repeat = _cuda_repeat_comparison(cuda_first, cuda_second)
            representation_conditions = all(
                record is None
                or (
                    record["descriptors_finite"]
                    and record["coordinates_within_image_bounds"]
                    and record["positive_scales"]
                    and record["keypoint_cap_ok"]
                    and record["hidden_cpu_fallback_absent"]
                )
                for run in (cpu_run, cuda_first, cuda_second)
                for record in (run.get("prepare_a"), run.get("prepare_b"))
            )
            status_valid = all(
                run["status"] in VALID_STATUSES
                for run in (cpu_run, cuda_first, cuda_second)
            )
            pair_rows.append(
                {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "pair_id": pair.pair_id,
                    "path_a": str(pair.path_a),
                    "path_b": str(pair.path_b),
                    "cpu": cpu_run,
                    "cuda_first": cuda_first,
                    "cuda_repeat": cuda_second,
                    "cpu_cuda_semantic_equivalence": semantic,
                    "cuda_repeat_exact": repeat,
                    "representation_absolute_conditions_passed": (
                        representation_conditions
                    ),
                    "statuses_valid": status_valid,
                    "passed": bool(
                        semantic["passed"]
                        and repeat["passed"]
                        and representation_conditions
                        and status_valid
                    ),
                }
            )
    finally:
        cpu_adapter.close()
        cuda_adapter.close()

    exact_raw_scores = sum(
        row["cpu_cuda_semantic_equivalence"]["raw_score_exact"]
        for row in pair_rows
    )
    downstream = {
        "pair_count": len(pair_rows),
        "status_all_equal": all(
            row["cpu_cuda_semantic_equivalence"]["status_equal"]
            for row in pair_rows
        ),
        "failure_stage_all_equal": all(
            row["cpu_cuda_semantic_equivalence"]["failure_stage_equal"]
            for row in pair_rows
        ),
        "decision_threshold_4_all_equal": all(
            row["cpu_cuda_semantic_equivalence"]["decision_threshold_4_equal"]
            for row in pair_rows
        ),
        "raw_score_all_within_one_inlier": all(
            row["cpu_cuda_semantic_equivalence"]["raw_score_within_one_inlier"]
            for row in pair_rows
        ),
        "exact_raw_score_count": exact_raw_scores,
        "minimum_exact_raw_score_count": MINIMUM_EXACT_RAW_SCORES,
        "descriptor_counts_all_passed": all(
            row["cpu_cuda_semantic_equivalence"]["descriptor_counts_passed"]
            for row in pair_rows
        ),
    }
    downstream["passed"] = bool(
        downstream["pair_count"] == 10
        and downstream["status_all_equal"]
        and downstream["failure_stage_all_equal"]
        and downstream["decision_threshold_4_all_equal"]
        and downstream["raw_score_all_within_one_inlier"]
        and downstream["exact_raw_score_count"]
        >= downstream["minimum_exact_raw_score_count"]
        and downstream["descriptor_counts_all_passed"]
    )
    reproducibility = {
        "synthetic_detector_all_exact": all(
            row["cuda_repeat_exact"]
            for row in detector_rows
            if row["kind"] == "synthetic"
        ),
        "real_image_detector_all_exact": all(
            row["cuda_repeat_exact"]
            for row in detector_rows
            if row["kind"] == "real"
        ),
        "real_pair_all_exact": all(
            row["cuda_repeat_exact"]["passed"] for row in pair_rows
        ),
        "synthetic_payload_sha256": stable_hash(
            [
                {
                    "case_id": row["case_id"],
                    "first": row["cuda_first_payload_sha256"],
                    "repeat": row["cuda_repeat_payload_sha256"],
                }
                for row in detector_rows
                if row["kind"] == "synthetic"
            ]
        ),
        "real_image_payload_sha256": stable_hash(
            [
                {
                    "case_id": row["case_id"],
                    "first": row["cuda_first_payload_sha256"],
                    "repeat": row["cuda_repeat_payload_sha256"],
                }
                for row in detector_rows
                if row["kind"] == "real"
            ]
        ),
        "real_pair_payload_sha256": stable_hash(
            [
                {
                    "case_id": row["case_id"],
                    "first": row["cuda_first"]["payload_sha256"],
                    "repeat": row["cuda_repeat"]["payload_sha256"],
                }
                for row in pair_rows
            ]
        ),
    }
    reproducibility["passed"] = all(
        reproducibility[key]
        for key in (
            "synthetic_detector_all_exact",
            "real_image_detector_all_exact",
            "real_pair_all_exact",
        )
    )
    correctness_passed = bool(
        same_five
        and len(detector_rows) == 17
        and len(pair_rows) == 10
        and all(row["passed"] for row in detector_rows)
        and all(row["passed"] for row in pair_rows)
        and downstream["passed"]
        and reproducibility["passed"]
    )
    timing = _timing_projection(pair_rows)
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "preflight_contract": PREFLIGHT_CONTRACT,
        "parent_failed_candidate": PARENT_FAILED_CANDIDATE,
        "purpose": "engineering_validation_only_not_parameter_selection",
        "passed": correctness_passed,
        "pilot_500_authorized": correctness_passed,
        "contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
            "frozen_before_preflight": True,
        },
        "selection": {
            "path": str(selection_path),
            "sha256": file_sha256(selection_path),
            "expected_sha256": EXPECTED_SELECTION_SHA256,
            "row_count": len(selection),
        },
        "algorithm_identity": identity,
        "algorithm_changed": False,
        "validation_contract_changed": True,
        "same_five_identities_in_b_and_c": same_five,
        "synthetic_case_count": 7,
        "real_image_count": len(real_cases),
        "real_pair_count": len(pair_rows),
        "detector_cases": detector_rows,
        "real_pair_checks": pair_rows,
        "downstream_semantic_validation": downstream,
        "cuda_reproducibility": reproducibility,
        "performance_projection": timing,
        "environment": environment,
        "no_parameter_tuning_performed": True,
        "no_tolerance_changed_after_result": True,
        "no_500_result_observed": True,
    }
    report["report_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def publish_engineering_preflight_v2(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Run once and immutably publish a pass or failure artifact."""

    project_root = project_root.resolve()
    pass_path = project_root / PASS_RELATIVE
    failure_path = project_root / FAILURE_RELATIVE
    existing = [path for path in (pass_path, failure_path) if path.exists()]
    if existing:
        if len(existing) != 1:
            raise HarrisZPlusPreflightV2Error(
                "Both pass and failure artifacts exist for immutable preflight v2."
            )
        report = json.loads(existing[0].read_text(encoding="utf-8"))
        expected_pass = existing[0] == pass_path
        if bool(report.get("passed")) != expected_pass:
            raise HarrisZPlusPreflightV2Error(
                "Existing v2 preflight artifact name/status mismatch."
            )
        return {
            **report,
            "path": str(existing[0]),
            "sha256": file_sha256(existing[0]),
            "reused_immutable_artifact": True,
        }
    report = run_engineering_preflight_v2(
        project_root=project_root,
        data_root=data_root,
    )
    target = pass_path if report["passed"] else failure_path
    _publish_exclusive_json(target, report)
    return {
        **report,
        "path": str(target),
        "sha256": file_sha256(target),
        "reused_immutable_artifact": False,
    }


def require_pilot_authorization(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    """Reject every 500 workflow until the immutable v2 pass artifact exists."""

    project_root = project_root.resolve()
    pass_path = project_root / PASS_RELATIVE
    failure_path = project_root / FAILURE_RELATIVE
    if failure_path.exists():
        raise HarrisZPlusPreflightV2Error(
            "The frozen v2 preflight failed; a 500 run is forbidden."
        )
    if not pass_path.is_file():
        raise HarrisZPlusPreflightV2Error(
            "No engineering_preflight_pass.json exists; a 500 run is forbidden."
        )
    report = json.loads(pass_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or report.get("passed") is not True
        or report.get("pilot_500_authorized") is not True
        or report.get("algorithm_changed") is not False
        or report.get("no_500_result_observed") is not True
        or report.get("contract", {}).get("sha256")
        != EXPECTED_CONTRACT_SHA256
    ):
        raise HarrisZPlusPreflightV2Error(
            "The v2 pass artifact does not authorize the immutable 500 protocol."
        )
    return {
        **report,
        "path": str(pass_path),
        "sha256": file_sha256(pass_path),
    }


def _publish_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen HarrisZ+ engineering preflight v2."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = publish_engineering_preflight_v2(
            project_root=args.project_root,
            data_root=args.data_root,
        )
    except (HarrisZPlusPreflightV2Error, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "pilot_500_authorized": report["pilot_500_authorized"],
                "path": report["path"],
                "sha256": report["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
