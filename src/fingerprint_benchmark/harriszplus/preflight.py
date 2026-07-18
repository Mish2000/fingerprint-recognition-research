"""Engineering preflight, immutable configuration freeze, and narrow integrity.

This module deliberately does not discover or content-hash the complete SD300
image trees.  The 500-identity selection and its already-published manifests
define the authoritative pilot image index; those 2,000 unique images are
content-hashed, while every image referenced by current protocol manifests is
stat-inventoried.  Prior protected-tree attestations cover the remaining
read-only dataset trees.  Current code, protocol manifests, and recorded result
manifests are also hashed before and after the HarrisZ+ pilot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from ..contract import (
    BENCHMARK_CONTRACT_VERSION,
    HIGHER_IS_MORE_SIMILAR,
    OK,
    TIMING_MODE_COLD_PAIR,
    WARMUP_POLICY,
    MethodAdapter,
    MethodMetadata,
)
from ..hashing import canonical_json_bytes, file_sha256, stable_config_hash, stable_hash
from ..manifest import PairRecord, read_pair_manifest
from ..provenance import implementation_provenance


DATASETS = ("sd300b", "sd300c")
SELF_PROTOCOLS = ("plain_self", "roll_self")
EXPECTED_PPI = {"sd300b": 1000, "sd300c": 2000}

SELECTION_COLUMNS = (
    "selection_index",
    "subject_id",
    "canonical_finger_position",
    "source_data_row",
    "source_csv_line",
    "source_identity_key",
)
EXPECTED_SELECTION_SHA256 = "942363780986aab4b28df97ab67421ac8322ead5c9fd5131446f90eb8cdca7e9"
EXPECTED_SELECTION_COUNT = 500

# Frozen before observing any HarrisZ+ 500-pair result.  These values must not
# be widened automatically after a preflight failure.
RESPONSE_ATOL = 5e-4
RESPONSE_RTOL = 2e-4
RESPONSE_MAX_ABSOLUTE_DELTA = 0.1
MINIMUM_RESPONSE_PIXEL_COVERAGE = 0.9999
SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE = 1.0
CANDIDATE_PREUNIFORM_MINIMUM_RATIO = 0.9995
SPATIAL_TOLERANCE_ORIGINAL_PX = 0.75
SCALE_TOLERANCE = 1e-6
MINIMUM_MATCHED_KEYPOINT_FRACTION = 0.95
MINIMUM_ORDER_SPEARMAN_RANK_CORRELATION = 0.99
ORDERING_RESPONSE_TIE_ATOL = 1e-6
ORDERING_COORDINATE_TIE_ATOL = 1e-3
MAX_KEYPOINTS = 3000
RNG_SEED = 0
MAXIMUM_PREFLIGHT_VRAM_FRACTION = 0.90
REQUIRED_DETECTOR_TIMING_FIELDS = (
    "detector_gpu_kernel_ms",
    "candidate_transfer_ms",
    "selection_cpu_ms",
    "total_ms",
)
REQUIRED_PREPARE_TIMING_FIELDS = (
    "image_load_ms",
    "lanczos_ms",
    "detector_gpu_wall_ms",
    "detector_gpu_kernel_ms",
    "candidate_transfer_ms",
    "selection_cpu_ms",
    "orientation_cpu_ms",
    "descriptor_cpu_ms",
    "prepare_total_ms",
)
REQUIRED_COMPARE_TIMING_FIELDS = (
    "matcher_cpu_ms",
    "ransac_cpu_ms",
    "compare_total_ms",
    "end_to_end_wall_ms",
)

PREFLIGHT_SCHEMA_VERSION = "harriszplus-engineering-preflight-v3"
FREEZE_SCHEMA_VERSION = "harriszplus-config-freeze-v2"
INTEGRITY_SCHEMA_VERSION = "harriszplus-narrow-protected-integrity-v1"
METHOD_NAME = "harriszplus_rootsift_geometric"
METHOD_VERSION = "harriszplus-rootsift-geometric-v1"


class HarrisZPlusPreflightError(ValueError):
    """Raised when engineering evidence is insufficient to authorize the pilot."""


@dataclass(frozen=True)
class SelectedIdentity:
    selection_index: int
    subject_id: str
    canonical_finger_position: int
    source_data_row: int
    source_csv_line: int
    source_identity_key: str

    @property
    def identity(self) -> tuple[str, int]:
        return self.subject_id, self.canonical_finger_position


def load_and_verify_selection(
    selection_path: Path,
    *,
    expected_sha256: str = EXPECTED_SELECTION_SHA256,
) -> list[SelectedIdentity]:
    """Load the authoritative 500 rows and verify bytes, order, and identity keys."""

    selection_path = selection_path.resolve()
    if not selection_path.is_file():
        raise HarrisZPlusPreflightError(f"Selection file does not exist: {selection_path}")
    actual_sha = file_sha256(selection_path)
    if actual_sha != expected_sha256:
        raise HarrisZPlusPreflightError(
            f"500-identity selection SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}."
        )
    with selection_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SELECTION_COLUMNS:
            raise HarrisZPlusPreflightError(
                f"Selection schema mismatch: expected {SELECTION_COLUMNS}, got {reader.fieldnames}."
            )
        selected: list[SelectedIdentity] = []
        for csv_line, row in enumerate(reader, start=2):
            if None in row:
                raise HarrisZPlusPreflightError(
                    f"Selection has unnamed values at CSV line {csv_line}."
                )
            try:
                item = SelectedIdentity(
                    selection_index=int(row["selection_index"]),
                    subject_id=row["subject_id"],
                    canonical_finger_position=int(row["canonical_finger_position"]),
                    source_data_row=int(row["source_data_row"]),
                    source_csv_line=int(row["source_csv_line"]),
                    source_identity_key=row["source_identity_key"],
                )
            except (TypeError, ValueError) as exc:
                raise HarrisZPlusPreflightError(
                    f"Selection contains an invalid integer at CSV line {csv_line}."
                ) from exc
            expected_key = f"{item.subject_id}|{item.canonical_finger_position}"
            if item.source_identity_key != expected_key:
                raise HarrisZPlusPreflightError(
                    f"Selection identity key mismatch at index {item.selection_index}: "
                    f"expected {expected_key!r}, got {item.source_identity_key!r}."
                )
            selected.append(item)
    if len(selected) != EXPECTED_SELECTION_COUNT:
        raise HarrisZPlusPreflightError(
            f"Selection must contain exactly {EXPECTED_SELECTION_COUNT} rows; got {len(selected)}."
        )
    if [row.selection_index for row in selected] != list(range(1, EXPECTED_SELECTION_COUNT + 1)):
        raise HarrisZPlusPreflightError("selection_index must be exactly contiguous 1..500 in file order.")
    identities = [row.identity for row in selected]
    if len(identities) != len(set(identities)):
        raise HarrisZPlusPreflightError("Selection contains duplicate subject/finger identities.")
    if any(not 1 <= row.canonical_finger_position <= 10 for row in selected):
        raise HarrisZPlusPreflightError("Selection contains a non-canonical finger position.")
    return selected


def validate_authoritative_manifest(
    manifest_path: Path,
    *,
    dataset: str,
    protocol: str,
    selection: Sequence[SelectedIdentity],
    data_root: Path = DEFAULT_DATA_ROOT,
    expected_sha256: str | None = None,
    require_self: bool = False,
    require_genuine: bool = False,
) -> dict[str, Any]:
    """Validate a 500-row recorded manifest without discovering the dataset tree."""

    if dataset not in DATASETS:
        raise HarrisZPlusPreflightError(f"Unsupported dataset: {dataset}")
    manifest_path = manifest_path.resolve()
    actual_sha = file_sha256(manifest_path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise HarrisZPlusPreflightError(
            f"Recorded manifest SHA-256 mismatch for {dataset}/{protocol}: "
            f"expected {expected_sha256}, got {actual_sha}."
        )
    pairs = read_pair_manifest(manifest_path)
    if len(pairs) != len(selection):
        raise HarrisZPlusPreflightError(
            f"Recorded manifest row count mismatch for {dataset}/{protocol}: "
            f"expected {len(selection)}, got {len(pairs)}."
        )
    expected_identities = [row.identity for row in selection]
    actual_identities = [(pair.subject_id, pair.canonical_finger_position) for pair in pairs]
    if actual_identities != expected_identities:
        raise HarrisZPlusPreflightError(
            f"Recorded manifest identity/order mismatch for {dataset}/{protocol}."
        )
    data_root = data_root.resolve()
    paths: set[Path] = set()
    for pair in pairs:
        if pair.dataset != dataset or pair.protocol != protocol:
            raise HarrisZPlusPreflightError(
                f"Manifest pair {pair.pair_id!r} has {pair.dataset}/{pair.protocol}; "
                f"expected {dataset}/{protocol}."
            )
        if pair.ppi != EXPECTED_PPI[dataset]:
            raise HarrisZPlusPreflightError(
                f"Manifest pair {pair.pair_id!r} has PPI {pair.ppi}; expected {EXPECTED_PPI[dataset]}."
            )
        if require_self and (
            pair.path_a != pair.path_b or pair.raw_frgp_a != pair.raw_frgp_b
        ):
            raise HarrisZPlusPreflightError(f"Self pair is not byte-path self: {pair.pair_id}")
        if require_genuine and pair.path_a == pair.path_b:
            raise HarrisZPlusPreflightError(f"Genuine PLAIN/ROLL pair reuses one image: {pair.pair_id}")
        for path in (pair.path_a, pair.path_b):
            resolved = path.resolve()
            if not _is_relative_to(resolved, data_root):
                raise HarrisZPlusPreflightError(
                    f"Manifest image escapes read-only data root: {resolved}"
                )
            if not resolved.is_file():
                raise HarrisZPlusPreflightError(f"Manifest image does not exist: {resolved}")
            paths.add(resolved)
    return {
        "validation_mode": "recorded_manifest_exact_500_no_dataset_tree_scan",
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_sha,
        "row_count": len(pairs),
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "selection_order_exact": True,
        "dataset": dataset,
        "protocol": protocol,
        "ppi": EXPECTED_PPI[dataset],
        "unique_referenced_image_count": len(paths),
        "all_referenced_images_exist": True,
        "all_referenced_images_under_data_root": True,
        "dataset_tree_scanned": False,
    }


def fixed_validation_policy() -> dict[str, Any]:
    """Return the immutable CPU/CUDA comparison policy."""

    return {
        "response_atol": RESPONSE_ATOL,
        "response_rtol": RESPONSE_RTOL,
        "response_max_absolute_delta": RESPONSE_MAX_ABSOLUTE_DELTA,
        "minimum_response_pixel_coverage": MINIMUM_RESPONSE_PIXEL_COVERAGE,
        "synthetic_minimum_response_pixel_coverage": (
            SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE
        ),
        "response_statistics_required": [
            "minimum",
            "maximum",
            "mean",
            "sample_stddev",
        ],
        "candidate_preuniform_minimum_count_ratio": CANDIDATE_PREUNIFORM_MINIMUM_RATIO,
        "uniform_and_final_candidate_counts_exact": True,
        "spatial_tolerance_original_pixels": SPATIAL_TOLERANCE_ORIGINAL_PX,
        "scale_tolerance": SCALE_TOLERANCE,
        "scale_index_exact": True,
        "minimum_matched_keypoint_fraction": MINIMUM_MATCHED_KEYPOINT_FRACTION,
        "minimum_order_spearman_rank_correlation": (
            MINIMUM_ORDER_SPEARMAN_RANK_CORRELATION
        ),
        "ordering_response_tie_atol": ORDERING_RESPONSE_TIE_ATOL,
        "ordering_coordinate_tie_atol": ORDERING_COORDINATE_TIE_ATOL,
        "cuda_repeat_exact": True,
        "required_detector_timing_fields": list(REQUIRED_DETECTOR_TIMING_FIELDS),
        "required_prepare_timing_fields": list(REQUIRED_PREPARE_TIMING_FIELDS),
        "required_compare_timing_fields": list(REQUIRED_COMPARE_TIMING_FIELDS),
        "canonical_pilot_cuda_device": "cuda:0",
        "noncanonical_device_override_allowed": False,
        "maximum_peak_vram_fraction_of_device": MAXIMUM_PREFLIGHT_VRAM_FRACTION,
        "peak_vram_must_be_positive_finite_and_reserved_ge_allocated": True,
        "peak_memory_aggregation": "maximum_across_all_detector_and_prepare_calls",
        "auto_relaxation_allowed": False,
        "frozen_before_500_results": True,
    }


def synthetic_suite(size: int = 256) -> dict[str, np.ndarray]:
    """Build the seven deterministic detector-only engineering patterns."""

    if size < 96:
        raise ValueError("Synthetic suite size must be at least 96 pixels.")
    flat = np.full((size, size), 127.0, dtype=np.float32)

    single = np.zeros((size, size), dtype=np.float32)
    single[size // 2 :, size // 2 :] = 255.0

    yy, xx = np.indices((size, size))
    tile = max(8, size // 16)
    checker = (((xx // tile + yy // tile) % 2) * 255.0).astype(np.float32)

    rotation = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), 31.0, 1.0)
    rotated = cv2.warpAffine(
        single.astype(np.uint8),
        rotation,
        (size, size),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)

    scaled = np.zeros((size, size), dtype=np.float32)
    start = size // 3
    scaled[start:, start:] = 255.0

    rng = np.random.default_rng(RNG_SEED)
    noisy = np.clip(single + rng.normal(0.0, 8.0, single.shape), 0.0, 255.0).astype(np.float32)

    edge = np.zeros((size, size), dtype=np.float32)
    edge[:, size // 2 :] = 255.0
    suite = {
        "flat": flat,
        "single_corner": single,
        "checkerboard": checker,
        "rotated_corner": rotated,
        "scaled_corner": scaled,
        "noisy_corner": noisy,
        "edge_only": edge,
    }
    for name, image in suite.items():
        if image.dtype != np.float32:
            raise AssertionError(f"Synthetic case {name} is not float32.")
        if not np.isfinite(image).all() or float(np.min(image)) < 0.0 or float(np.max(image)) > 255.0:
            raise AssertionError(f"Synthetic case {name} violates the [0, 255] image contract.")
    return suite


def _canonical_pilot_cuda_device(device: str | None) -> str:
    """Resolve the only device identity supported by the immutable pilot workflow."""

    if device not in (None, "cuda", "cuda:0"):
        raise HarrisZPlusPreflightError(
            "The immutable HarrisZ+ pilot is bound to logical device cuda:0; "
            "only None, 'cuda', or 'cuda:0' are accepted."
        )
    return "cuda:0"


def run_engineering_preflight(
    *,
    project_root: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    selection_path: Path | None = None,
    source_manifest_root: Path | None = None,
    config: Any | None = None,
    adapter: MethodAdapter | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run synthetic + 5 B/same 5 C engineering checks, never parameter tuning."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    canonical_device = _canonical_pilot_cuda_device(device)
    selection_path = (
        selection_path
        or project_root / "results/pilots/sourceafis_joint_500_v1/selected_identities.csv"
    ).resolve()
    source_manifest_root = (
        source_manifest_root
        or project_root / "results/pilots/sourceafis_joint_500_v1/manifests"
    ).resolve()
    selection = load_and_verify_selection(selection_path)

    if config is None:
        from .config import HarrisZPlusConfig

        config = HarrisZPlusConfig(device=canonical_device)
    else:
        configured_device = getattr(config, "device", None)
        if _canonical_pilot_cuda_device(configured_device) != canonical_device:
            raise HarrisZPlusPreflightError(
                "Preflight config device does not resolve to the canonical pilot device."
            )
        if configured_device != canonical_device:
            changed = getattr(config, "changed", None)
            if not callable(changed):
                raise HarrisZPlusPreflightError(
                    "Preflight config must bind device='cuda:0' explicitly."
                )
            config = changed(device=canonical_device)
    if adapter is None:
        from .adapter import HarrisZPlusGeometricAdapter

        adapter = HarrisZPlusGeometricAdapter(config)
        close_adapter = True
    else:
        close_adapter = False

    adapter_config = getattr(adapter, "config", None)
    if getattr(adapter_config, "device", None) != canonical_device:
        raise HarrisZPlusPreflightError(
            "The preflight adapter must bind the canonical pilot device cuda:0 explicitly."
        )

    _validate_adapter_contract(adapter)
    torch, environment = _configure_and_describe_cuda(canonical_device)
    from .cuda_detector import detect_harriszplus_cuda
    from .reference_cpu import detect_harriszplus_cpu

    real_cases: list[dict[str, Any]] = []
    sample_selection = selection[:5]
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
        for selected, pair in zip(sample_selection, pairs[:5], strict=True):
            if selected.identity != (pair.subject_id, pair.canonical_finger_position):
                raise HarrisZPlusPreflightError("Real preflight identity alignment failed.")
            image = cv2.imread(str(pair.path_a), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise HarrisZPlusPreflightError(f"Cannot decode real preflight image: {pair.path_a}")
            real_cases.append(
                {
                    "case_id": f"{dataset}:{selected.selection_index}:plain_self",
                    "kind": "real",
                    "dataset": dataset,
                    "ppi": pair.ppi,
                    "path": pair.path_a,
                    "pair": pair,
                    "image": image.astype(np.float32, copy=False),
                }
            )

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

    detector_results: list[dict[str, Any]] = []
    peak_memory_observations: list[dict[str, Any]] = []
    timing_checks: list[dict[str, Any]] = []
    all_passed = True
    try:
        for case in cases:
            source_u8 = np.ascontiguousarray(
                np.clip(np.rint(case["image"]), 0.0, 255.0), dtype=np.uint8
            )
            image = source_u8.astype(np.float32)
            doubled_image = cv2.resize(
                source_u8,
                (int(source_u8.shape[1]) * 2, int(source_u8.shape[0]) * 2),
                interpolation=cv2.INTER_LANCZOS4,
            ).astype(np.float32, copy=False)
            cpu = detect_harriszplus_cpu(
                image,
                config,
                doubled_image=doubled_image,
                return_response_maps=True,
            )
            torch.cuda.synchronize(canonical_device)
            cuda_start = perf_counter_ns()
            cuda_first = detect_harriszplus_cuda(
                image,
                config,
                device=canonical_device,
                doubled_image=doubled_image,
                return_response_maps=True,
            )
            torch.cuda.synchronize(canonical_device)
            synchronized_first_ms = (perf_counter_ns() - cuda_start) / 1_000_000.0
            torch.cuda.synchronize(canonical_device)
            cuda_repeat_start = perf_counter_ns()
            cuda_second = detect_harriszplus_cuda(
                image,
                config,
                device=canonical_device,
                doubled_image=doubled_image,
                return_response_maps=True,
            )
            torch.cuda.synchronize(canonical_device)
            synchronized_repeat_ms = (perf_counter_ns() - cuda_repeat_start) / 1_000_000.0
            comparison = compare_detector_results(
                cpu,
                cuda_first,
                minimum_response_pixel_coverage=(
                    SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE
                    if case["kind"] == "synthetic"
                    else MINIMUM_RESPONSE_PIXEL_COVERAGE
                ),
            )
            cuda_hash_1 = detector_result_sha256(cuda_first)
            cuda_hash_2 = detector_result_sha256(cuda_second)
            first_memory = _peak_memory_observation(
                getattr(cuda_first, "diagnostics", {}),
                context=f"{case['case_id']}:cuda_first",
                allocated_key="peak_vram_allocated_bytes",
                reserved_key="peak_vram_reserved_bytes",
            )
            repeat_memory = _peak_memory_observation(
                getattr(cuda_second, "diagnostics", {}),
                context=f"{case['case_id']}:cuda_repeat",
                allocated_key="peak_vram_allocated_bytes",
                reserved_key="peak_vram_reserved_bytes",
            )
            peak_memory_observations.extend((first_memory, repeat_memory))
            first_timing = _required_timing_evidence(
                getattr(cuda_first, "timings", {}),
                REQUIRED_DETECTOR_TIMING_FIELDS,
                context=f"{case['case_id']}:cuda_first",
            )
            repeat_timing = _required_timing_evidence(
                getattr(cuda_second, "timings", {}),
                REQUIRED_DETECTOR_TIMING_FIELDS,
                context=f"{case['case_id']}:cuda_repeat",
            )
            synchronized_timing = _required_timing_evidence(
                {
                    "synchronized_first_wall_ms": synchronized_first_ms,
                    "synchronized_repeat_wall_ms": synchronized_repeat_ms,
                },
                ("synchronized_first_wall_ms", "synchronized_repeat_wall_ms"),
                context=f"{case['case_id']}:outer_synchronized_wall",
            )
            timing_checks.extend((first_timing, repeat_timing, synchronized_timing))
            exact_repeat = cuda_hash_1 == cuda_hash_2
            finite = _detector_result_is_finite(cpu) and _detector_result_is_finite(cuda_first)
            cap_ok = (
                len(_keypoints(cpu)) <= MAX_KEYPOINTS
                and len(_keypoints(cuda_first)) <= MAX_KEYPOINTS
            )
            case_timing_passed = bool(
                first_timing["passed"]
                and repeat_timing["passed"]
                and synchronized_timing["passed"]
            )
            case_memory_observations_valid = bool(
                first_memory["values_valid"] and repeat_memory["values_valid"]
            )
            case_passed = bool(
                comparison["passed"]
                and exact_repeat
                and finite
                and cap_ok
                and case_timing_passed
                and case_memory_observations_valid
            )
            if case["case_id"] == "synthetic:flat":
                flat_zero = len(_keypoints(cpu)) == 0 and len(_keypoints(cuda_first)) == 0
                case_passed = case_passed and flat_zero
            else:
                flat_zero = None
            all_passed = all_passed and case_passed
            detector_results.append(
                {
                    "case_id": case["case_id"],
                    "kind": case["kind"],
                    "dataset": case["dataset"],
                    "ppi": case["ppi"],
                    "image_shape": list(image.shape),
                    "shared_lanczos_doubled_image": True,
                    "lanczos_source_policy": "uint8 source then cv2.INTER_LANCZOS4 then float32",
                    "cpu_keypoint_count": len(_keypoints(cpu)),
                    "cuda_keypoint_count": len(_keypoints(cuda_first)),
                    "finite": finite,
                    "keypoint_cap_ok": cap_ok,
                    "flat_zero_keypoints": flat_zero,
                    "cpu_cuda": comparison,
                    "cuda_first_sha256": cuda_hash_1,
                    "cuda_repeat_sha256": cuda_hash_2,
                    "cuda_repeat_exact": exact_repeat,
                    "synchronized_first_wall_ms": synchronized_first_ms,
                    "synchronized_repeat_wall_ms": synchronized_repeat_ms,
                    "required_timing_fields": {
                        "cuda_first": first_timing,
                        "cuda_repeat": repeat_timing,
                        "outer_synchronized_wall": synchronized_timing,
                        "passed": case_timing_passed,
                    },
                    "peak_memory": {
                        "cuda_first": first_memory,
                        "cuda_repeat": repeat_memory,
                        "observations_valid": case_memory_observations_valid,
                    },
                    "cpu_timings": _json_safe(getattr(cpu, "timings", {})),
                    "cuda_timings": _json_safe(getattr(cuda_first, "timings", {})),
                    "passed": case_passed,
                }
            )

        pair_results: list[dict[str, Any]] = []
        score_projection: list[dict[str, Any]] = []
        repeat_score_projection: list[dict[str, Any]] = []
        for case in real_cases:
            pair: PairRecord = case["pair"]
            metadata_a = pair.image_metadata_a()
            metadata_b = pair.image_metadata_b()
            first_a = adapter.prepare(pair.path_a, metadata_a)
            first_b = adapter.prepare(pair.path_b, metadata_b)
            first_compare = adapter.compare(first_a.representation, first_b.representation)
            repeat_compare = adapter.compare(first_a.representation, first_b.representation)
            prepare_a_timing = _required_timing_evidence(
                first_a.diagnostics,
                REQUIRED_PREPARE_TIMING_FIELDS,
                context=f"{case['case_id']}:prepare_a",
            )
            prepare_b_timing = _required_timing_evidence(
                first_b.diagnostics,
                REQUIRED_PREPARE_TIMING_FIELDS,
                context=f"{case['case_id']}:prepare_b",
            )
            compare_timing = _required_timing_evidence(
                first_compare.diagnostics,
                REQUIRED_COMPARE_TIMING_FIELDS,
                context=f"{case['case_id']}:compare_first",
            )
            repeat_compare_timing = _required_timing_evidence(
                repeat_compare.diagnostics,
                REQUIRED_COMPARE_TIMING_FIELDS,
                context=f"{case['case_id']}:compare_repeat",
            )
            timing_checks.extend(
                (prepare_a_timing, prepare_b_timing, compare_timing, repeat_compare_timing)
            )
            prepare_a_memory = _peak_memory_observation(
                first_a.diagnostics,
                context=f"{case['case_id']}:prepare_a",
                allocated_key="peak_vram_allocated",
                reserved_key="peak_vram_reserved",
            )
            prepare_b_memory = _peak_memory_observation(
                first_b.diagnostics,
                context=f"{case['case_id']}:prepare_b",
                allocated_key="peak_vram_allocated",
                reserved_key="peak_vram_reserved",
            )
            peak_memory_observations.extend((prepare_a_memory, prepare_b_memory))
            first_hash = representation_sha256(first_a.representation)
            repeat_hash = representation_sha256(first_b.representation)
            score_first = _nonnegative_integer_score(first_compare.raw_score)
            score_repeat = _nonnegative_integer_score(repeat_compare.raw_score)
            representation_exact = first_hash == repeat_hash
            score_exact = score_first == score_repeat
            descriptor_ok = all(
                _descriptor_count(outcome) >= 2
                for outcome in (first_a, first_b)
            )
            pair_timing_passed = all(
                evidence["passed"]
                for evidence in (
                    prepare_a_timing,
                    prepare_b_timing,
                    compare_timing,
                    repeat_compare_timing,
                )
            )
            pair_memory_observations_valid = bool(
                prepare_a_memory["values_valid"] and prepare_b_memory["values_valid"]
            )
            ppi_coordinate_handling = _ppi_coordinate_handling_evidence(
                dataset=str(case["dataset"]),
                manifest_ppi=pair.ppi,
                prepared_a=first_a,
                prepared_b=first_b,
                compare_diagnostics=first_compare.diagnostics,
            )
            pair_passed = (
                representation_exact
                and score_exact
                and descriptor_ok
                and ppi_coordinate_handling["passed"]
                and pair_timing_passed
                and pair_memory_observations_valid
            )
            all_passed = all_passed and pair_passed
            score_projection.append(
                {
                    "case_id": case["case_id"],
                    "score": score_first,
                    "status": OK,
                }
            )
            repeat_score_projection.append(
                {
                    "case_id": case["case_id"],
                    "score": score_repeat,
                    "status": OK,
                }
            )
            first_score_hash = hashlib.sha256(
                canonical_json_bytes(score_projection[-1])
            ).hexdigest()
            repeat_score_hash = hashlib.sha256(
                canonical_json_bytes(repeat_score_projection[-1])
            ).hexdigest()
            pair_results.append(
                {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "ppi": case["ppi"],
                    "path": str(case["path"]),
                    "prepare_a_sha256": first_hash,
                    "prepare_b_sha256": repeat_hash,
                    "representation_repeat_exact": representation_exact,
                    "score": score_first,
                    "repeat_score": score_repeat,
                    "score_sha256": first_score_hash,
                    "repeat_score_sha256": repeat_score_hash,
                    "score_repeat_exact": score_exact,
                    "descriptor_extraction_ok": descriptor_ok,
                    "ppi_coordinate_handling": ppi_coordinate_handling,
                    "required_timing_fields": {
                        "prepare_a": prepare_a_timing,
                        "prepare_b": prepare_b_timing,
                        "compare_first": compare_timing,
                        "compare_repeat": repeat_compare_timing,
                        "passed": pair_timing_passed,
                    },
                    "peak_memory": {
                        "prepare_a": prepare_a_memory,
                        "prepare_b": prepare_b_memory,
                        "observations_valid": pair_memory_observations_valid,
                    },
                    "prepare_a_diagnostics": _json_safe(first_a.diagnostics),
                    "prepare_b_diagnostics": _json_safe(first_b.diagnostics),
                    "compare_diagnostics": _json_safe(first_compare.diagnostics),
                    "passed": pair_passed,
                }
            )

        if len(real_cases) != 10:
            raise HarrisZPlusPreflightError(
                f"Engineering preflight must contain exactly 10 real self pairs; got {len(real_cases)}."
            )
        if {case["dataset"] for case in real_cases} != set(DATASETS):
            raise HarrisZPlusPreflightError("Engineering preflight lacks one B/C condition.")
        b_identities = [
            (case["pair"].subject_id, case["pair"].canonical_finger_position)
            for case in real_cases
            if case["dataset"] == "sd300b"
        ]
        c_identities = [
            (case["pair"].subject_id, case["pair"].canonical_finger_position)
            for case in real_cases
            if case["dataset"] == "sd300c"
        ]
        same_five = b_identities == c_identities and len(b_identities) == 5
        all_passed = all_passed and same_five
        ppi_coordinate_handling_all_passed = all(
            row["ppi_coordinate_handling"]["passed"] for row in pair_results
        )
        all_passed = all_passed and ppi_coordinate_handling_all_passed

        first_score_payload_hash = hashlib.sha256(
            canonical_json_bytes(score_projection)
        ).hexdigest()
        repeat_score_payload_hash = hashlib.sha256(
            canonical_json_bytes(repeat_score_projection)
        ).hexdigest()
        metadata = adapter.metadata()
        preflight_runner_config = _effective_runner_config(metadata)
        runtime_identity = _canonical_runtime_identity(project_root, metadata.runtime)
        device_binding = {
            "requested_device": device,
            "canonical_device": canonical_device,
            "direct_detector_device": environment.get("device"),
            "adapter_runtime_selected_device": runtime_identity.get("selected_device"),
            "adapter_runtime_device_index": runtime_identity.get("device_index"),
            "passed": bool(
                canonical_device == "cuda:0"
                and environment.get("device") == "cuda:0"
                and runtime_identity.get("selected_device") == "cuda:0"
                and runtime_identity.get("device_index") == 0
            ),
        }
        all_passed = all_passed and device_binding["passed"]
        _, _, preflight_implementation_hash = implementation_provenance(
            adapter=adapter,
            method_metadata=metadata,
            startup_validation={},
            runner_source_path=(project_root / "src/fingerprint_benchmark/runner.py").resolve(),
        )
        peak_memory_report = _evaluate_peak_memory(
            peak_memory_observations,
            total_vram_bytes=int(environment["vram_bytes"]),
        )
        required_timing_fields_all_passed = bool(
            timing_checks and all(check["passed"] for check in timing_checks)
        )
        all_passed = (
            all_passed
            and peak_memory_report["passed"]
            and required_timing_fields_all_passed
        )
        report = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "purpose": "engineering_validation_only_not_parameter_selection",
            "passed": bool(all_passed),
            "pilot_500_authorized": bool(all_passed),
            "selection": {
                "path": str(selection_path),
                "sha256": file_sha256(selection_path),
                "row_count": len(selection),
            },
            "validation_policy": fixed_validation_policy(),
            "synthetic_case_count": len(synthetic_suite()),
            "real_image_count": len(real_cases),
            "real_pair_count": len(pair_results),
            "same_five_identities_in_b_and_c": same_five,
            "ppi_coordinate_handling_all_passed": ppi_coordinate_handling_all_passed,
            "detector_cases": detector_results,
            "real_pair_checks": pair_results,
            "determinism": {
                "cuda_detector_all_repeat_exact": all(
                    row["cuda_repeat_exact"] for row in detector_results
                ),
                "representation_all_repeat_exact": all(
                    row["representation_repeat_exact"] for row in pair_results
                ),
                "score_all_repeat_exact": all(row["score_repeat_exact"] for row in pair_results),
                "score_payload_sha256": first_score_payload_hash,
                "repeat_score_payload_sha256": repeat_score_payload_hash,
                "score_payload_repeat_exact": first_score_payload_hash == repeat_score_payload_hash,
            },
            "timing_synchronization": {
                "torch_cuda_synchronize_before_and_after_every_timed_detector_call": True,
                "all_synchronized_wall_times_nonnegative": all(
                    row["synchronized_first_wall_ms"] >= 0
                    and row["synchronized_repeat_wall_ms"] >= 0
                    for row in detector_results
                ),
                "required_timing_check_count": len(timing_checks),
                "required_timing_fields_all_passed": required_timing_fields_all_passed,
                "checks": timing_checks,
            },
            "memory": peak_memory_report,
            "device_binding": device_binding,
            "environment": environment,
            "runtime_identity": runtime_identity,
            "candidate_freeze_identity": {
                "canonical_config_hash": stable_config_hash(preflight_runner_config),
                "implementation_hash": preflight_implementation_hash,
                "runtime_identity_hash": runtime_identity["runtime_identity_hash"],
            },
            "no_parameter_tuning_performed": True,
            "no_500_result_observed": True,
        }
    finally:
        if close_adapter:
            adapter.close()
    if not report["passed"]:
        raise HarrisZPlusPreflightError(
            "Engineering preflight failed; the 500-identity pilot is not authorized. "
            f"Evidence: {json.dumps(report, ensure_ascii=True, sort_keys=True)}"
        )
    return report


def compare_detector_results(
    cpu: Any,
    cuda: Any,
    *,
    minimum_response_pixel_coverage: float = MINIMUM_RESPONSE_PIXEL_COVERAGE,
) -> dict[str, Any]:
    """Compare CPU/CUDA response maps and final ordered keypoints at frozen tolerances."""

    if not 0.0 <= minimum_response_pixel_coverage <= 1.0:
        raise ValueError("minimum_response_pixel_coverage must be in [0, 1].")

    cpu_maps = _ordered_response_maps(cpu)
    cuda_maps = _ordered_response_maps(cuda)
    if not cpu_maps or [item[0] for item in cpu_maps] != [item[0] for item in cuda_maps]:
        raise HarrisZPlusPreflightError(
            "CPU/CUDA validation requires equal non-empty per-scale response maps."
        )
    response_rows: list[dict[str, Any]] = []
    responses_pass = True
    for (scale_index, cpu_map), (_, cuda_map) in zip(cpu_maps, cuda_maps, strict=True):
        cpu_array = np.asarray(cpu_map)
        cuda_array = np.asarray(cuda_map)
        if cpu_array.shape != cuda_array.shape:
            raise HarrisZPlusPreflightError(
                f"Response-map shape mismatch at scale {scale_index}: "
                f"{cpu_array.shape} != {cuda_array.shape}."
            )
        finite = bool(np.isfinite(cpu_array).all() and np.isfinite(cuda_array).all())
        close_mask = np.isclose(
            cpu_array,
            cuda_array,
            atol=RESPONSE_ATOL,
            rtol=RESPONSE_RTOL,
            equal_nan=False,
        )
        close_count = int(np.count_nonzero(close_mask)) if finite else 0
        pixel_count = int(cpu_array.size)
        close_fraction = 1.0 if pixel_count == 0 else close_count / pixel_count
        outlier_count = pixel_count - close_count
        cpu_statistics = _array_statistics(cpu_array)
        cuda_statistics = _array_statistics(cuda_array)
        statistic_rows: dict[str, dict[str, Any]] = {}
        statistics_passed = finite
        for statistic, statistics_key in (
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("mean", "mean"),
            ("sample_stddev", "stddev"),
        ):
            cpu_value = cpu_statistics[statistics_key]
            cuda_value = cuda_statistics[statistics_key]
            within_tolerance = bool(
                cpu_value is not None
                and cuda_value is not None
                and math.isclose(
                    cpu_value,
                    cuda_value,
                    abs_tol=RESPONSE_ATOL,
                    rel_tol=RESPONSE_RTOL,
                )
            )
            statistic_rows[statistic] = {
                "cpu": cpu_value,
                "cuda": cuda_value,
                "absolute_delta": (
                    abs(cpu_value - cuda_value)
                    if cpu_value is not None and cuda_value is not None
                    else None
                ),
                "within_tolerance": within_tolerance,
            }
            statistics_passed = statistics_passed and within_tolerance
        coverage_passed = finite and close_fraction >= minimum_response_pixel_coverage
        delta = np.abs(cpu_array.astype(np.float64) - cuda_array.astype(np.float64))
        maximum_absolute_delta = float(np.max(delta)) if delta.size else 0.0
        maximum_delta_passed = finite and maximum_absolute_delta <= RESPONSE_MAX_ABSOLUTE_DELTA
        close = bool(statistics_passed and coverage_passed and maximum_delta_passed)
        response_rows.append(
            {
                "scale_index": scale_index,
                "shape": list(cpu_array.shape),
                "finite": finite,
                "within_tolerance": close,
                "all_pixels_within_tolerance": bool(finite and outlier_count == 0),
                "pixel_count": pixel_count,
                "allclose_pixel_count": close_count,
                "outlier_pixel_count": outlier_count,
                "allclose_pixel_fraction": close_fraction,
                "minimum_allclose_pixel_fraction": minimum_response_pixel_coverage,
                "coverage_passed": coverage_passed,
                "statistics": statistic_rows,
                "statistics_passed": statistics_passed,
                "maximum_absolute_delta": maximum_absolute_delta,
                "maximum_allowed_absolute_delta": RESPONSE_MAX_ABSOLUTE_DELTA,
                "maximum_delta_passed": maximum_delta_passed,
                "mean_absolute_delta": float(np.mean(delta)) if delta.size else 0.0,
                "cpu": cpu_statistics,
                "cuda": cuda_statistics,
            }
        )
        responses_pass = responses_pass and close

    cpu_points = [_keypoint_record(point) for point in _keypoints(cpu)]
    cuda_points = [_keypoint_record(point) for point in _keypoints(cuda)]
    candidate_count_comparison = _compare_candidate_counts(cpu, cuda)
    ordered_aligned_count = 0
    ordered_coordinate_deltas: list[float] = []
    ordered_response_deltas: list[float] = []
    for cpu_point, cuda_point in zip(cpu_points, cuda_points):
        spatial_delta = math.hypot(
            cpu_point["x"] - cuda_point["x"],
            cpu_point["y"] - cuda_point["y"],
        )
        response_delta = abs(cpu_point["response"] - cuda_point["response"])
        ordered_coordinate_deltas.append(spatial_delta)
        ordered_response_deltas.append(response_delta)
        if _keypoints_within_frozen_tolerance(cpu_point, cuda_point):
            ordered_aligned_count += 1

    # Nearest-unused correspondence checks set agreement independently of the
    # final-list order.  The separate ordering gate below canonicalizes only
    # contiguous mathematical tie groups; distinct response groups retain the
    # raw detector order and therefore cannot be repaired by global sorting.
    bucket_size = SPATIAL_TOLERANCE_ORIGINAL_PX
    cuda_spatial_buckets: dict[tuple[int, int, int], list[int]] = {}
    for cuda_index, cuda_point in enumerate(cuda_points):
        bucket_key = (
            int(cuda_point["scale_index"]),
            math.floor(float(cuda_point["x"]) / bucket_size),
            math.floor(float(cuda_point["y"]) / bucket_size),
        )
        cuda_spatial_buckets.setdefault(bucket_key, []).append(cuda_index)
    unused_cuda = set(range(len(cuda_points)))
    matched_pairs: list[tuple[int, int, float, float]] = []
    for cpu_index, cpu_point in enumerate(cpu_points):
        candidates: list[tuple[float, float, int]] = []
        scale_index = int(cpu_point["scale_index"])
        center_bucket_x = math.floor(float(cpu_point["x"]) / bucket_size)
        center_bucket_y = math.floor(float(cpu_point["y"]) / bucket_size)
        for neighbor_y in range(center_bucket_y - 1, center_bucket_y + 2):
            for neighbor_x in range(center_bucket_x - 1, center_bucket_x + 2):
                for cuda_index in cuda_spatial_buckets.get(
                    (scale_index, neighbor_x, neighbor_y), ()
                ):
                    if cuda_index not in unused_cuda:
                        continue
                    cuda_point = cuda_points[cuda_index]
                    if not _keypoints_within_frozen_tolerance(cpu_point, cuda_point):
                        continue
                    spatial_delta = math.hypot(
                        cpu_point["x"] - cuda_point["x"],
                        cpu_point["y"] - cuda_point["y"],
                    )
                    response_delta = abs(cpu_point["response"] - cuda_point["response"])
                    candidates.append((spatial_delta, response_delta, cuda_index))
        if not candidates:
            continue
        spatial_delta, response_delta, cuda_index = min(candidates)
        unused_cuda.remove(cuda_index)
        matched_pairs.append((cpu_index, cuda_index, spatial_delta, response_delta))
    denominator = max(len(cpu_points), len(cuda_points))
    matched_fraction = 1.0 if denominator == 0 else len(matched_pairs) / denominator
    ordered_fraction = 1.0 if denominator == 0 else ordered_aligned_count / denominator
    canonical_cpu_indices = _canonical_validation_keypoint_index_order(cpu_points)
    canonical_cuda_indices = _canonical_validation_keypoint_index_order(cuda_points)
    canonical_cpu_points = [cpu_points[index] for index in canonical_cpu_indices]
    canonical_cuda_points = [cuda_points[index] for index in canonical_cuda_indices]
    canonical_ordered_aligned_count = sum(
        _keypoints_within_frozen_tolerance(cpu_point, cuda_point)
        for cpu_point, cuda_point in zip(canonical_cpu_points, canonical_cuda_points)
    )
    canonical_ordered_fraction = (
        1.0 if denominator == 0 else canonical_ordered_aligned_count / denominator
    )
    cpu_canonical_rank = {
        raw_index: rank for rank, raw_index in enumerate(canonical_cpu_indices)
    }
    cuda_canonical_rank = {
        raw_index: rank for rank, raw_index in enumerate(canonical_cuda_indices)
    }
    matched_canonical_rank_pairs = [
        (cpu_canonical_rank[cpu_index], cuda_canonical_rank[cuda_index])
        for cpu_index, cuda_index, _, _ in matched_pairs
    ]
    order_spearman_rank_correlation = _spearman_rank_correlation(
        matched_canonical_rank_pairs
    )
    nearest_keypoints_pass = matched_fraction >= MINIMUM_MATCHED_KEYPOINT_FRACTION
    ordered_keypoints_pass = (
        order_spearman_rank_correlation
        >= MINIMUM_ORDER_SPEARMAN_RANK_CORRELATION
    )
    keypoints_pass = nearest_keypoints_pass and ordered_keypoints_pass
    return {
        "response_maps": response_rows,
        "response_maps_passed": responses_pass,
        "minimum_response_pixel_coverage": minimum_response_pixel_coverage,
        "cpu_keypoint_count": len(cpu_points),
        "cuda_keypoint_count": len(cuda_points),
        "nearest_unused_matched_keypoint_count": len(matched_pairs),
        "matched_keypoint_fraction": matched_fraction,
        "ordered_aligned_keypoint_count": ordered_aligned_count,
        "ordered_matched_keypoint_fraction": ordered_fraction,
        "ordered_final_list_exact_within_tolerance": ordered_aligned_count == denominator,
        "canonical_ordered_aligned_keypoint_count": canonical_ordered_aligned_count,
        "canonical_ordered_matched_keypoint_fraction": canonical_ordered_fraction,
        "canonical_ordered_final_list_exact_within_tolerance": (
            canonical_ordered_aligned_count == denominator
        ),
        "ordering_response_tie_atol": ORDERING_RESPONSE_TIE_ATOL,
        "ordering_coordinate_tie_atol": ORDERING_COORDINATE_TIE_ATOL,
        "minimum_matched_keypoint_fraction": MINIMUM_MATCHED_KEYPOINT_FRACTION,
        "matched_order_spearman_rank_correlation": order_spearman_rank_correlation,
        "minimum_order_spearman_rank_correlation": (
            MINIMUM_ORDER_SPEARMAN_RANK_CORRELATION
        ),
        "maximum_matched_canonical_rank_delta": max(
            (abs(cpu_rank - cuda_rank) for cpu_rank, cuda_rank in matched_canonical_rank_pairs),
            default=0,
        ),
        "nearest_keypoints_passed": nearest_keypoints_pass,
        "order_correlation_passed": ordered_keypoints_pass,
        "ordered_keypoints_passed": ordered_keypoints_pass,
        "maximum_matched_spatial_delta_original_pixels": max(
            (item[2] for item in matched_pairs), default=0.0
        ),
        "maximum_matched_response_delta": max(
            (item[3] for item in matched_pairs), default=0.0
        ),
        "maximum_ordered_position_spatial_delta_original_pixels": max(
            ordered_coordinate_deltas, default=0.0
        ),
        "maximum_ordered_position_response_delta": max(
            ordered_response_deltas, default=0.0
        ),
        "scale_index_exact_for_matched": True,
        "unmatched_cpu_keypoint_count": len(cpu_points) - len(matched_pairs),
        "unmatched_cuda_keypoint_count": len(cuda_points) - len(matched_pairs),
        "candidate_counts": candidate_count_comparison,
        "candidate_counts_passed": candidate_count_comparison["passed"],
        "keypoints_passed": keypoints_pass,
        "passed": bool(
            responses_pass and candidate_count_comparison["passed"] and keypoints_pass
        ),
    }


def _candidate_count_ratio(first: int, second: int) -> float:
    maximum = max(first, second)
    return 1.0 if maximum == 0 else min(first, second) / maximum


def _canonical_coordinate_tie_order(
    indexed_points: list[tuple[int, Mapping[str, Any]]],
) -> list[tuple[int, Mapping[str, Any]]]:
    output: list[tuple[int, Mapping[str, Any]]] = []
    for scale_index in sorted(
        {int(point["scale_index"]) for _, point in indexed_points}, reverse=True
    ):
        scale_points = [
            item for item in indexed_points if int(item[1]["scale_index"]) == scale_index
        ]
        scale_points.sort(key=lambda item: (float(item[1]["y"]), float(item[1]["x"]), item[0]))
        offset = 0
        while offset < len(scale_points):
            y_anchor = float(scale_points[offset][1]["y"])
            y_stop = offset + 1
            while (
                y_stop < len(scale_points)
                and float(scale_points[y_stop][1]["y"]) - y_anchor
                <= ORDERING_COORDINATE_TIE_ATOL
            ):
                y_stop += 1
            y_group = scale_points[offset:y_stop]
            y_group.sort(key=lambda item: (float(item[1]["x"]), int(item[1]["source_index"]), item[0]))
            x_offset = 0
            while x_offset < len(y_group):
                x_anchor = float(y_group[x_offset][1]["x"])
                x_stop = x_offset + 1
                while (
                    x_stop < len(y_group)
                    and float(y_group[x_stop][1]["x"]) - x_anchor
                    <= ORDERING_COORDINATE_TIE_ATOL
                ):
                    x_stop += 1
                x_group = y_group[x_offset:x_stop]
                x_group.sort(key=lambda item: (int(item[1]["source_index"]), item[0]))
                output.extend(x_group)
                x_offset = x_stop
            offset = y_stop
    return output


def _canonical_validation_keypoint_order(
    points: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Canonical projection for validating mathematical response/coordinate ties.

    The detector's raw list and raw response values are never changed.  This
    projection is used only for the separately reported ordering gate.
    """

    return [points[index] for index in _canonical_validation_keypoint_index_order(points)]


def _canonical_validation_keypoint_index_order(
    points: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Return raw indices in validation-only tie-canonicalized final-rank order."""

    indexed = list(enumerate(points))
    output: list[int] = []
    offset = 0
    while offset < len(indexed):
        response_minimum = float(indexed[offset][1]["response"])
        response_maximum = response_minimum
        stop = offset + 1
        while stop < len(indexed):
            candidate_response = float(indexed[stop][1]["response"])
            next_minimum = min(response_minimum, candidate_response)
            next_maximum = max(response_maximum, candidate_response)
            if next_maximum - next_minimum > ORDERING_RESPONSE_TIE_ATOL:
                break
            response_minimum = next_minimum
            response_maximum = next_maximum
            stop += 1
        output.extend(
            index for index, _ in _canonical_coordinate_tie_order(indexed[offset:stop])
        )
        offset = stop
    return output


def _spearman_rank_correlation(rank_pairs: Sequence[tuple[int, int]]) -> float:
    """Return Pearson correlation of already-ranked positions (Spearman's rho)."""

    if len(rank_pairs) <= 1:
        return 1.0
    cpu_mean = sum(cpu_rank for cpu_rank, _ in rank_pairs) / len(rank_pairs)
    cuda_mean = sum(cuda_rank for _, cuda_rank in rank_pairs) / len(rank_pairs)
    numerator = sum(
        (cpu_rank - cpu_mean) * (cuda_rank - cuda_mean)
        for cpu_rank, cuda_rank in rank_pairs
    )
    cpu_sum_squares = sum(
        (cpu_rank - cpu_mean) ** 2 for cpu_rank, _ in rank_pairs
    )
    cuda_sum_squares = sum(
        (cuda_rank - cuda_mean) ** 2 for _, cuda_rank in rank_pairs
    )
    denominator = math.sqrt(cpu_sum_squares * cuda_sum_squares)
    return 1.0 if denominator == 0.0 else numerator / denominator


def _candidate_count_row(first: Any, second: Any, *, exact: bool) -> dict[str, Any]:
    cpu_count = int(first)
    cuda_count = int(second)
    ratio = _candidate_count_ratio(cpu_count, cuda_count)
    passed = (
        cpu_count == cuda_count
        if exact
        else ratio >= CANDIDATE_PREUNIFORM_MINIMUM_RATIO
    )
    return {
        "cpu": cpu_count,
        "cuda": cuda_count,
        "absolute_delta": abs(cpu_count - cuda_count),
        "minimum_to_maximum_ratio": ratio,
        "required_exact": exact,
        "minimum_required_ratio": 1.0 if exact else CANDIDATE_PREUNIFORM_MINIMUM_RATIO,
        "passed": passed,
    }


def _compare_candidate_counts(cpu: Any, cuda: Any) -> dict[str, Any]:
    """Compare every available detector selection-stage count without hiding deltas."""

    cpu_diagnostics = getattr(cpu, "diagnostics", {})
    cuda_diagnostics = getattr(cuda, "diagnostics", {})
    if not isinstance(cpu_diagnostics, Mapping) or not isinstance(cuda_diagnostics, Mapping):
        return {"available": False, "passed": False, "reason": "diagnostics_not_mappings"}
    per_scale_keys = (
        "candidates_before_mask",
        "candidates_after_mask",
        "candidates_after_local_maxima",
        "candidates_after_scale_suppression",
        "candidates_after_eigen_ratio",
    )
    aggregate_preuniform_keys = (*per_scale_keys, "candidates_after_duplicate_removal")
    exact_keys = ("candidates_after_uniform_selection", "final_keypoint_count")
    expected_keys = (*aggregate_preuniform_keys, *exact_keys)
    cpu_has_any = any(key in cpu_diagnostics for key in expected_keys)
    cuda_has_any = any(key in cuda_diagnostics for key in expected_keys)
    if not cpu_has_any and not cuda_has_any:
        # Lightweight unit-test records may intentionally omit detector
        # diagnostics. Formal detector results always populate all stages.
        return {"available": False, "passed": True, "reason": "not_supplied"}

    missing = [
        key
        for key in expected_keys
        if key not in cpu_diagnostics or key not in cuda_diagnostics
    ]
    aggregate: dict[str, Any] = {}
    passed = not missing
    for key in expected_keys:
        if key in cpu_diagnostics and key in cuda_diagnostics:
            row = _candidate_count_row(
                cpu_diagnostics[key],
                cuda_diagnostics[key],
                exact=key in exact_keys,
            )
            aggregate[key] = row
            passed = passed and row["passed"]

    cpu_scales = cpu_diagnostics.get("scales", {})
    cuda_scales = cuda_diagnostics.get("scales", {})
    per_scale: dict[str, Any] = {}
    if not isinstance(cpu_scales, Mapping) or not isinstance(cuda_scales, Mapping):
        passed = False
    else:
        scale_names = sorted(set(cpu_scales) | set(cuda_scales), key=lambda item: int(item))
        for scale_name in scale_names:
            cpu_scale = cpu_scales.get(scale_name, {})
            cuda_scale = cuda_scales.get(scale_name, {})
            cpu_counts = cpu_scale.get("counts", {}) if isinstance(cpu_scale, Mapping) else {}
            cuda_counts = cuda_scale.get("counts", {}) if isinstance(cuda_scale, Mapping) else {}
            scale_rows: dict[str, Any] = {}
            for key in per_scale_keys:
                if key not in cpu_counts or key not in cuda_counts:
                    passed = False
                    scale_rows[key] = {"passed": False, "reason": "missing"}
                    continue
                row = _candidate_count_row(cpu_counts[key], cuda_counts[key], exact=False)
                scale_rows[key] = row
                passed = passed and row["passed"]
            per_scale[str(scale_name)] = scale_rows
    return {
        "available": True,
        "minimum_preuniform_count_ratio": CANDIDATE_PREUNIFORM_MINIMUM_RATIO,
        "uniform_and_final_counts_exact": True,
        "missing_aggregate_fields": missing,
        "aggregate": aggregate,
        "per_scale": per_scale,
        "passed": passed,
    }


def detector_result_sha256(result: Any) -> str:
    """Hash response bytes, final ordering, diagnostics, and timing-independent content."""

    digest = hashlib.sha256()
    digest.update(b"harriszplus-detector-result-v1\0")
    for scale_index, response in _ordered_response_maps(result):
        digest.update(canonical_json_bytes(scale_index) + b"\0")
        _update_array_hash(digest, np.asarray(response))
    digest.update(canonical_json_bytes([_keypoint_record(point) for point in _keypoints(result)]))
    diagnostics = _json_safe(getattr(result, "diagnostics", {}))
    # Diagnostics may contain wall/kernel timings.  They are evidence, not part
    # of deterministic representation identity.
    digest.update(canonical_json_bytes(_without_timing_fields(diagnostics)))
    return digest.hexdigest()


def representation_sha256(representation: Any) -> str:
    """Hash an opaque prepared representation without relying on object identity."""

    try:
        from .provenance import representation_sha256 as declared_hash

        return str(declared_hash(getattr(representation, "payload", representation)))
    except (ImportError, AttributeError, TypeError, ValueError):
        digest = hashlib.sha256()
        digest.update(b"harriszplus-prepared-representation-v1\0")
        _update_value_hash(digest, representation)
        return digest.hexdigest()


def freeze_configuration(
    *,
    project_root: Path,
    adapter: MethodAdapter,
    preflight_report: Mapping[str, Any],
    config_directory: Path | None = None,
) -> dict[str, Any]:
    """Atomically freeze config/decision/provenance, refusing every overwrite."""

    if (
        preflight_report.get("passed") is not True
        or preflight_report.get("pilot_500_authorized") is not True
        or preflight_report.get("ppi_coordinate_handling_all_passed") is not True
        or not isinstance(preflight_report.get("device_binding"), Mapping)
        or preflight_report["device_binding"].get("passed") is not True
        or not isinstance(preflight_report.get("memory"), Mapping)
        or preflight_report["memory"].get("passed") is not True
        or not isinstance(preflight_report.get("timing_synchronization"), Mapping)
        or preflight_report["timing_synchronization"].get(
            "required_timing_fields_all_passed"
        )
        is not True
    ):
        raise HarrisZPlusPreflightError(
            "A passing engineering preflight, including PPI coordinate handling, is required "
            "before config freeze."
        )
    project_root = project_root.resolve()
    config_directory = (
        config_directory
        or project_root / "results/harriszplus_rootsift_geometric/config"
    ).resolve()
    if config_directory.exists():
        return validate_frozen_configuration(
            config_directory=config_directory,
            adapter=adapter,
            preflight_report=preflight_report,
        )

    metadata = adapter.metadata()
    _validate_method_metadata(metadata)
    runner_config = _effective_runner_config(metadata)
    canonical_config_hash = stable_config_hash(runner_config)
    runtime_identity = _canonical_runtime_identity(project_root, metadata.runtime)
    provenance, implementation_components, implementation_hash = implementation_provenance(
        adapter=adapter,
        method_metadata=metadata,
        startup_validation={},
        runner_source_path=(project_root / "src/fingerprint_benchmark/runner.py").resolve(),
    )
    candidate_identity = preflight_report.get("candidate_freeze_identity")
    expected_identity = {
        "canonical_config_hash": canonical_config_hash,
        "implementation_hash": implementation_hash,
        "runtime_identity_hash": runtime_identity["runtime_identity_hash"],
    }
    if candidate_identity != expected_identity:
        raise HarrisZPlusPreflightError(
            "Adapter config or implementation changed after engineering preflight; "
            "a new preflight is required before freeze."
        )
    algorithm_config = _algorithm_config(adapter, metadata)
    decision = {
        "schema_version": "harriszplus-operational-decision-rule-v1",
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "status_field": "status",
        "required_status": OK,
        "score_field": "geometric_inlier_count",
        "score_direction": HIGHER_IS_MORE_SIMILAR,
        "threshold": 4,
        "acceptance_operator": "status == ok and geometric_inlier_count >= 4",
        "tie_policy": "a score exactly equal to 4 is accepted",
        "rationale": (
            "The raw score and geometric backend are semantically aligned with the existing SIFT pilot; "
            "threshold 4 is frozen only to make the operational pilot report parallel."
        ),
        "calibration_statement": (
            "Operational pilot threshold only; it was not calibrated for HarrisZ+ at a target FAR. "
            "Future accuracy work must calibrate on separate development impostors."
        ),
        "sourceafis_threshold_40_used": False,
        "frozen_before_500_results": True,
        "tuning_on_500_results": False,
    }
    decision_hash = stable_hash(decision)
    decision_with_hash = {**decision, "decision_rule_hash": decision_hash}
    environment = {
        "runtime": _json_safe(metadata.runtime),
        "platform": platform.platform(),
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "runtime_identity": runtime_identity,
    }

    config_directory.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(
        tempfile.mkdtemp(
            dir=config_directory.parent,
            prefix=f".{config_directory.name}.candidate-",
        )
    )
    try:
        payloads = {
            "algorithm_config.json": algorithm_config,
            "runner_config.json": runner_config,
            "decision_rule.json": decision_with_hash,
            "environment.json": environment,
            "runtime_identity.json": runtime_identity,
            "implementation_components.json": implementation_components,
            "implementation_provenance.json": provenance,
            "preflight_reference.json": {
                "schema_version": "harriszplus-preflight-freeze-reference-v1",
                "preflight_schema_version": preflight_report.get("schema_version"),
                "preflight_sha256": hashlib.sha256(
                    canonical_json_bytes(preflight_report)
                ).hexdigest(),
                "passed": True,
                "ppi_coordinate_handling_all_passed": True,
                "validation_policy": fixed_validation_policy(),
            },
        }
        for filename, payload in payloads.items():
            (candidate / filename).write_bytes(_pretty_json_bytes(payload))
        files = {
            filename: {
                "sha256": file_sha256(candidate / filename),
                "size": (candidate / filename).stat().st_size,
            }
            for filename in sorted(payloads)
        }
        freeze = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "method": METHOD_NAME,
            "method_version": METHOD_VERSION,
            "canonical_config_hash": canonical_config_hash,
            "implementation_hash": implementation_hash,
            "runtime_identity_hash": runtime_identity["runtime_identity_hash"],
            "decision_rule_hash": decision_hash,
            "config_file_sha256": files["runner_config.json"]["sha256"],
            "decision_rule_file_sha256": files["decision_rule.json"]["sha256"],
            "preflight_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(preflight_report)
            ).hexdigest(),
            "files": files,
            "immutable": True,
            "overwrite_allowed": False,
            "frozen_before_500_results": True,
            "validation_policy": fixed_validation_policy(),
        }
        (candidate / "freeze_manifest.json").write_bytes(_pretty_json_bytes(freeze))
        os.replace(candidate, config_directory)
    except Exception:
        if candidate.exists():
            import shutil

            shutil.rmtree(candidate)
        raise
    return validate_frozen_configuration(
        config_directory=config_directory,
        adapter=adapter,
        preflight_report=preflight_report,
    )


def validate_frozen_configuration(
    *,
    config_directory: Path,
    adapter: MethodAdapter | None = None,
    preflight_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every frozen file and, optionally, current adapter identity."""

    config_directory = config_directory.resolve()
    freeze_path = config_directory / "freeze_manifest.json"
    freeze = _read_json(freeze_path)
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise HarrisZPlusPreflightError(f"Unsupported config freeze: {freeze_path}")
    if freeze.get("immutable") is not True or freeze.get("overwrite_allowed") is not False:
        raise HarrisZPlusPreflightError("Frozen configuration lacks immutable/no-overwrite assertions.")
    recorded_files = _required_mapping(freeze, "files")
    expected_logical_paths: set[str] = set()
    for filename, record in recorded_files.items():
        if not isinstance(filename, str) or Path(filename).as_posix() != filename:
            raise HarrisZPlusPreflightError("Frozen config manifest contains an invalid path.")
        path = (config_directory / filename).resolve()
        if not _is_relative_to(path, config_directory):
            raise HarrisZPlusPreflightError("Frozen config manifest path escapes its namespace.")
        expected_logical_paths.add(filename)
        if not path.is_file():
            raise HarrisZPlusPreflightError(f"Frozen config file is missing: {path}")
        if file_sha256(path) != record.get("sha256") or path.stat().st_size != record.get("size"):
            raise HarrisZPlusPreflightError(f"Frozen config file changed: {path}")
    expected_logical_paths.add("freeze_manifest.json")
    actual_logical_paths = {
        path.relative_to(config_directory).as_posix()
        for path in config_directory.rglob("*")
        if path.is_file()
    }
    if actual_logical_paths != expected_logical_paths:
        raise HarrisZPlusPreflightError(
            "Frozen config namespace contains added, removed, or unlisted files."
        )
    runner_config = _read_json(config_directory / "runner_config.json")
    if stable_config_hash(runner_config) != freeze.get("canonical_config_hash"):
        raise HarrisZPlusPreflightError("Frozen canonical config hash mismatch.")
    decision = _read_json(config_directory / "decision_rule.json")
    embedded_decision_hash = decision.pop("decision_rule_hash", None)
    if embedded_decision_hash != stable_hash(decision) or embedded_decision_hash != freeze.get(
        "decision_rule_hash"
    ):
        raise HarrisZPlusPreflightError("Frozen decision-rule hash mismatch.")
    if decision.get("threshold") != 4 or decision.get("sourceafis_threshold_40_used") is not False:
        raise HarrisZPlusPreflightError("Frozen operational decision rule changed.")
    frozen_runtime_identity = _read_json(config_directory / "runtime_identity.json")
    embedded_runtime_hash = frozen_runtime_identity.pop("runtime_identity_hash", None)
    if (
        embedded_runtime_hash != stable_hash(frozen_runtime_identity)
        or embedded_runtime_hash != freeze.get("runtime_identity_hash")
    ):
        raise HarrisZPlusPreflightError("Frozen runtime/dependency identity hash mismatch.")
    if preflight_report is not None:
        current_preflight_hash = hashlib.sha256(
            canonical_json_bytes(preflight_report)
        ).hexdigest()
        if current_preflight_hash != freeze.get("preflight_canonical_sha256"):
            raise HarrisZPlusPreflightError("Preflight evidence does not match the config freeze.")
    if adapter is not None:
        metadata = adapter.metadata()
        _validate_method_metadata(metadata)
        if _effective_runner_config(metadata) != runner_config:
            raise HarrisZPlusPreflightError("Current adapter config differs from frozen runner config.")
        _, components, current_hash = implementation_provenance(
            adapter=adapter,
            method_metadata=metadata,
            startup_validation={},
            runner_source_path=(
                config_directory.parents[2] / "src/fingerprint_benchmark/runner.py"
            ).resolve(),
        )
        frozen_components = _read_json(config_directory / "implementation_components.json")
        if components != frozen_components or current_hash != freeze.get("implementation_hash"):
            raise HarrisZPlusPreflightError("Current implementation differs from the frozen implementation.")
        current_runtime_identity = _canonical_runtime_identity(
            config_directory.parents[2], metadata.runtime
        )
        current_runtime_hash = current_runtime_identity.pop("runtime_identity_hash")
        if (
            current_runtime_hash != embedded_runtime_hash
            or current_runtime_identity != frozen_runtime_identity
        ):
            raise HarrisZPlusPreflightError(
                "Current Python/OpenCV/NumPy/PyTorch/CUDA/cuDNN/GPU/driver/dependency "
                "identity differs from the frozen runtime."
            )
    return {**freeze, "config_directory": str(config_directory), "validated": True}


def collect_narrow_protected_snapshot(project_root: Path) -> dict[str, Any]:
    """Hash protected code/results through manifests, explicitly avoiding 115GB data scans."""

    project_root = project_root.resolve()
    direct_paths: set[Path] = set()
    for relative in (
        "pyproject.toml",
        "environment.yml",
        "src/fingerprint_benchmark/contract.py",
        "src/fingerprint_benchmark/runner.py",
        "src/fingerprint_benchmark/bundle.py",
        "src/fingerprint_benchmark/hashing.py",
        "src/fingerprint_benchmark/io.py",
        "src/fingerprint_benchmark/manifest.py",
        "src/fingerprint_benchmark/preflight.py",
        "src/fingerprint_benchmark/provenance.py",
    ):
        path = project_root / relative
        if path.is_file():
            direct_paths.add(path.resolve())
    for pattern in (
        "src/fingerprint_benchmark/sift/*.py",
        "src/fingerprint_benchmark/sourceafis*.py",
        "src/fingerprint_benchmark/shared_accuracy*.py",
        "src/fingerprint_benchmark/harriszplus/*.py",
        "protocols/sd300b/*.csv",
        "protocols/sd300c/*.csv",
        "apps/sourceafis-sidecar/pom.xml",
        "apps/sourceafis-sidecar/src/main/**/*.java",
        "apps/sourceafis-sidecar/src/main/**/*.properties",
        "apps/sourceafis-sidecar/target/sourceafis-sidecar-0.2.0-shaded.jar",
    ):
        direct_paths.update(path.resolve() for path in project_root.glob(pattern) if path.is_file())

    manifest_paths = [
        project_root / "results/pilots/sourceafis_joint_500_v1/artifact_manifest.json",
        project_root / "results/pilots/sift_geometric_joint_500_v1/artifact_manifest.json",
        project_root / "results/shared_accuracy/sourceafis_sift_v1/artifact_manifest.json",
    ]
    recorded_manifests: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise HarrisZPlusPreflightError(
                f"Required protected artifact manifest is missing: {manifest_path}"
            )
        validated_files = _validate_recorded_artifact_manifest(manifest_path)
        direct_paths.add(manifest_path.resolve())
        direct_paths.update(path for path, _, _ in validated_files)
        recorded_manifests.append(
            {
                "path": str(manifest_path.resolve()),
                "sha256": file_sha256(manifest_path),
                "validated_file_count": len(validated_files),
            }
        )

    prior_attestations = []
    for relative in (
        "results/pilots/sift_geometric_joint_500_v1/integrity/protected_artifact_integrity.json",
        "results/pilots/sourceafis_joint_500_v1/integrity/protected_repository_integrity.json",
    ):
        path = project_root / relative
        if not path.is_file():
            raise HarrisZPlusPreflightError(
                f"Required prior protected-tree attestation is missing: {path}"
            )
        payload = _read_json(path)
        if payload.get("protected_artifacts_unchanged") is not True:
            raise HarrisZPlusPreflightError(
                f"Prior protected-tree attestation did not pass: {path}"
            )
        direct_paths.add(path.resolve())
        prior_attestations.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "protected_artifacts_unchanged": payload.get("protected_artifacts_unchanged"),
                "before": payload.get("before"),
                "after": payload.get("after"),
            }
        )

    selection_path = project_root / "results/pilots/sourceafis_joint_500_v1/selected_identities.csv"
    load_and_verify_selection(selection_path)
    direct_paths.add(selection_path.resolve())
    files = []
    for path in sorted(direct_paths, key=lambda item: str(item).lower()):
        if _is_relative_to(path, project_root):
            logical_path = path.relative_to(project_root).as_posix()
        else:
            logical_path = str(path)
        files.append(
            {
                "path": logical_path,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    tree_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in files)
    ).hexdigest()
    return {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "algorithm": "sha256",
        "scope": "narrow_protected_code_protocols_and_recorded_result_manifests",
        "dataset_tree_scan_performed": False,
        "dataset_integrity_basis": (
            "read-only source policy plus prior 115GB before/after protected-tree attestations; "
            "all current pilot image references are validated from authoritative manifests"
        ),
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "recorded_artifact_manifests": recorded_manifests,
        "prior_dataset_attestations": prior_attestations,
        "file_count": len(files),
        "total_bytes": sum(record["size"] for record in files),
        "tree_sha256": tree_sha,
        "files": files,
    }


def compare_protected_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare narrow snapshots and reject any changed/added/removed protected file."""

    before_files = {record["path"]: record for record in before.get("files", [])}
    after_files = {record["path"]: record for record in after.get("files", [])}
    added = sorted(set(after_files) - set(before_files))
    removed = sorted(set(before_files) - set(after_files))
    changed = sorted(
        path
        for path in set(before_files) & set(after_files)
        if before_files[path] != after_files[path]
    )
    passed = not added and not removed and not changed and before.get("tree_sha256") == after.get(
        "tree_sha256"
    )
    report = {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "passed": passed,
        "protected_artifacts_unchanged": passed,
        "dataset_tree_rescanned": False,
        "before_tree_sha256": before.get("tree_sha256"),
        "after_tree_sha256": after.get("tree_sha256"),
        "before_file_count": before.get("file_count"),
        "after_file_count": after.get("file_count"),
        "added_paths": added,
        "removed_paths": removed,
        "changed_paths": changed,
        "mismatch_count": len(added) + len(removed) + len(changed),
    }
    if not passed:
        raise HarrisZPlusPreflightError(
            f"Protected artifacts changed during the pilot: {json.dumps(report, sort_keys=True)}"
        )
    return report


def _configure_and_describe_cuda(device: str | None) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise HarrisZPlusPreflightError("PyTorch is required for HarrisZ+ CUDA preflight.") from exc
    if not torch.cuda.is_available():
        raise HarrisZPlusPreflightError("CUDA is unavailable; the 500-identity pilot cannot start.")
    torch.manual_seed(RNG_SEED)
    torch.cuda.manual_seed_all(RNG_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.is_autocast_enabled():
        raise HarrisZPlusPreflightError("Autocast must be disabled for HarrisZ+.")
    selected_device = torch.device(_canonical_pilot_cuda_device(device))
    torch.cuda.set_device(selected_device)
    properties = torch.cuda.get_device_properties(selected_device)
    torch.cuda.reset_peak_memory_stats(selected_device)
    runtime_version = getattr(torch.version, "cuda", None)
    driver_version = None
    try:
        driver_version = torch.cuda.driver_version()
    except (AttributeError, RuntimeError):
        pass
    return torch, {
        "gpu_model": properties.name,
        "vram_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "nvidia_driver": driver_version,
        "cuda_runtime": runtime_version,
        "pytorch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "device": str(selected_device),
        "canonical_pilot_device_enforced": True,
        "dtype": "float32",
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "fp16_used": False,
        "bf16_used": False,
    }


def _required_timing_evidence(
    values: Mapping[str, Any] | Any,
    required_fields: Sequence[str],
    *,
    context: str,
) -> dict[str, Any]:
    mapping = values if isinstance(values, Mapping) else {}
    observed: dict[str, float | None] = {}
    missing: list[str] = []
    nonfinite: list[str] = []
    negative: list[str] = []
    for field in required_fields:
        if field not in mapping or mapping[field] is None:
            observed[field] = None
            missing.append(field)
            continue
        try:
            numeric = float(mapping[field])
        except (TypeError, ValueError):
            observed[field] = None
            nonfinite.append(field)
            continue
        observed[field] = numeric
        if not math.isfinite(numeric):
            nonfinite.append(field)
        elif numeric < 0.0:
            negative.append(field)
    return {
        "context": context,
        "required_fields": list(required_fields),
        "observed_ms": observed,
        "missing_fields": missing,
        "nonfinite_fields": nonfinite,
        "negative_fields": negative,
        "passed": not missing and not nonfinite and not negative,
    }


def _peak_memory_observation(
    values: Mapping[str, Any] | Any,
    *,
    context: str,
    allocated_key: str,
    reserved_key: str,
) -> dict[str, Any]:
    mapping = values if isinstance(values, Mapping) else {}

    def numeric(key: str) -> int | None:
        value = mapping.get(key)
        if isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed) or parsed < 0.0 or not parsed.is_integer():
            return None
        return int(parsed)

    allocated = numeric(allocated_key)
    reserved = numeric(reserved_key)
    values_valid = (
        allocated is not None
        and reserved is not None
        and allocated > 0
        and reserved > 0
        and reserved >= allocated
    )
    return {
        "context": context,
        "allocated_key": allocated_key,
        "reserved_key": reserved_key,
        "peak_vram_allocated_bytes": allocated,
        "peak_vram_reserved_bytes": reserved,
        "values_valid": values_valid,
    }


def _evaluate_peak_memory(
    observations: Sequence[Mapping[str, Any]],
    *,
    total_vram_bytes: int,
) -> dict[str, Any]:
    device_vram_valid = isinstance(total_vram_bytes, int) and total_vram_bytes > 0
    observations_valid = bool(observations) and all(
        observation.get("values_valid") is True for observation in observations
    )
    allocated_values = [
        int(observation["peak_vram_allocated_bytes"])
        for observation in observations
        if observation.get("peak_vram_allocated_bytes") is not None
    ]
    reserved_values = [
        int(observation["peak_vram_reserved_bytes"])
        for observation in observations
        if observation.get("peak_vram_reserved_bytes") is not None
    ]
    maximum_allocated = max(allocated_values, default=0)
    maximum_reserved = max(reserved_values, default=0)
    allocated_fraction = (
        maximum_allocated / total_vram_bytes if device_vram_valid else math.inf
    )
    reserved_fraction = (
        maximum_reserved / total_vram_bytes if device_vram_valid else math.inf
    )
    positive = maximum_allocated > 0 and maximum_reserved > 0
    within_device_vram = (
        device_vram_valid
        and maximum_allocated <= total_vram_bytes
        and maximum_reserved <= total_vram_bytes
        and allocated_fraction <= MAXIMUM_PREFLIGHT_VRAM_FRACTION
        and reserved_fraction <= MAXIMUM_PREFLIGHT_VRAM_FRACTION
    )
    return {
        "device_vram_bytes": total_vram_bytes,
        "observation_count": len(observations),
        "observations": [dict(observation) for observation in observations],
        "all_observations_valid": observations_valid,
        "peak_vram_allocated_bytes": maximum_allocated,
        "peak_vram_reserved_bytes": maximum_reserved,
        "peak_allocated_fraction_of_device": allocated_fraction,
        "peak_reserved_fraction_of_device": reserved_fraction,
        "maximum_allowed_fraction_of_device": MAXIMUM_PREFLIGHT_VRAM_FRACTION,
        "positive_peak_observed": positive,
        "within_device_vram": within_device_vram,
        "aggregation": "maximum_across_all_detector_and_prepare_calls",
        "passed": observations_valid and positive and within_device_vram,
    }


def _canonical_runtime_identity(
    project_root: Path,
    runtime: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Freeze only execution-relevant runtime/dependency fields, then hash them canonically."""

    if not isinstance(runtime, Mapping):
        raise HarrisZPlusPreflightError("Adapter runtime metadata must be a mapping.")
    torch_runtime = runtime.get("torch")
    if not isinstance(torch_runtime, Mapping):
        raise HarrisZPlusPreflightError("Adapter runtime metadata lacks PyTorch identity.")
    if torch_runtime.get("installed") is not True or torch_runtime.get("cuda_available") is not True:
        raise HarrisZPlusPreflightError("Frozen runtime requires CUDA-enabled PyTorch.")
    dependency_hashes = runtime.get("dependency_artifact_sha256")
    if not isinstance(dependency_hashes, Mapping):
        raise HarrisZPlusPreflightError("Adapter runtime lacks dependency artifact hashes.")
    project_root = project_root.resolve()
    actual_dependency_hashes = {}
    for filename in ("pyproject.toml", "environment.yml"):
        path = project_root / filename
        if not path.is_file():
            raise HarrisZPlusPreflightError(f"Required dependency artifact is missing: {path}")
        actual_dependency_hashes[filename] = file_sha256(path)
    if dict(dependency_hashes) != actual_dependency_hashes:
        raise HarrisZPlusPreflightError(
            "Runtime dependency hashes do not match current pyproject.toml/environment.yml."
        )

    device_index = torch_runtime.get("device_index")
    nvidia_smi = runtime.get("nvidia_smi")
    selected_gpu = None
    if isinstance(nvidia_smi, list):
        selected_gpu = next(
            (
                record
                for record in nvidia_smi
                if isinstance(record, Mapping) and record.get("index") == device_index
            ),
            None,
        )
    if not isinstance(selected_gpu, Mapping) or not selected_gpu.get("driver_version"):
        raise HarrisZPlusPreflightError(
            "Runtime identity requires the selected GPU's NVIDIA driver from nvidia-smi."
        )
    required_torch_fields = (
        "version",
        "cuda_build_runtime",
        "cudnn_version",
        "selected_device",
        "device_index",
        "gpu_model",
        "total_vram_bytes",
        "compute_capability",
    )
    missing = [field for field in required_torch_fields if torch_runtime.get(field) is None]
    if missing:
        raise HarrisZPlusPreflightError(
            f"Runtime identity is incomplete; missing PyTorch fields: {missing}."
        )
    identity = {
        "schema_version": "harriszplus-runtime-identity-v1",
        "python_version": runtime.get("python_version"),
        "python_executable": runtime.get("python_executable"),
        "opencv_version": runtime.get("opencv_version"),
        "numpy_version": runtime.get("numpy_version"),
        "operating_system": runtime.get("operating_system"),
        "pytorch_version": torch_runtime["version"],
        "torch_cuda_build_runtime": torch_runtime["cuda_build_runtime"],
        "cudnn_version": torch_runtime["cudnn_version"],
        "selected_device": torch_runtime["selected_device"],
        "device_index": torch_runtime["device_index"],
        "gpu_model": torch_runtime["gpu_model"],
        "total_vram_bytes": torch_runtime["total_vram_bytes"],
        "compute_capability": torch_runtime["compute_capability"],
        "nvidia_driver_version": selected_gpu["driver_version"],
        "dependency_artifact_sha256": actual_dependency_hashes,
    }
    missing_identity = [
        key for key, value in identity.items() if value is None or value == ""
    ]
    if missing_identity:
        raise HarrisZPlusPreflightError(
            f"Runtime identity contains empty required fields: {missing_identity}."
        )
    return {**identity, "runtime_identity_hash": stable_hash(identity)}


def _validate_adapter_contract(adapter: MethodAdapter) -> None:
    metadata = adapter.metadata()
    _validate_method_metadata(metadata)
    cache_values = [
        metadata.config.get("representation_cache"),
        metadata.config.get("cross_pair_cache"),
    ]
    if tuple(cache_values) != (False, False):
        raise HarrisZPlusPreflightError(
            "Adapter metadata must explicitly declare representation_cache=false and "
            "cross_pair_cache=false."
        )
    if WARMUP_POLICY.get("prepare_operations_per_pair") != 2:
        raise HarrisZPlusPreflightError("Generic runner warm-up no longer prepares both sides.")


def _validate_method_metadata(metadata: MethodMetadata) -> None:
    if metadata.method != METHOD_NAME or metadata.method_version != METHOD_VERSION:
        raise HarrisZPlusPreflightError(
            f"Unexpected adapter identity: {metadata.method}/{metadata.method_version}."
        )
    if metadata.score_direction != HIGHER_IS_MORE_SIMILAR:
        raise HarrisZPlusPreflightError("HarrisZ+ score direction must be higher_is_more_similar.")
    if "inlier" not in metadata.score_semantics.lower():
        raise HarrisZPlusPreflightError("HarrisZ+ score semantics must identify geometric inliers.")


def _effective_runner_config(metadata: MethodMetadata) -> dict[str, Any]:
    return {
        **metadata.config,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "method": metadata.method,
        "method_version": metadata.method_version,
        "score_direction": metadata.score_direction,
        "score_semantics": metadata.score_semantics,
        "timing_mode": TIMING_MODE_COLD_PAIR,
        "warm_up_policy": WARMUP_POLICY,
    }


def _algorithm_config(adapter: MethodAdapter, metadata: MethodMetadata) -> dict[str, Any]:
    config = getattr(adapter, "config", None)
    if config is not None and callable(getattr(config, "as_dict", None)):
        value = config.as_dict()
        if isinstance(value, dict):
            scale_table = getattr(config, "scale_table", None)
            if callable(scale_table):
                value = {**value, "derived_scale_table": scale_table()}
            return _json_safe(value)
    return _json_safe(metadata.config)


def _ppi_coordinate_handling_evidence(
    *,
    dataset: str,
    manifest_ppi: int | float,
    prepared_a: Any,
    prepared_b: Any,
    compare_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate manifest PPI propagation and the frozen physical RANSAC geometry."""

    if dataset not in EXPECTED_PPI:
        raise HarrisZPlusPreflightError(f"Unsupported PPI preflight dataset: {dataset!r}.")

    def finite_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    expected_manifest_ppi = float(EXPECTED_PPI[dataset])
    recorded_manifest_ppi = finite_float(manifest_ppi)
    payload_a = getattr(getattr(prepared_a, "representation", None), "payload", None)
    payload_b = getattr(getattr(prepared_b, "representation", None), "payload", None)
    prepared_ppi_a = finite_float(getattr(payload_a, "ppi", None))
    prepared_ppi_b = finite_float(getattr(payload_b, "ppi", None))
    normalization = compare_diagnostics.get("coordinate_normalization")
    reference_ppi = finite_float(compare_diagnostics.get("reference_ppi"))
    threshold_reference = finite_float(
        compare_diagnostics.get("ransac_threshold_reference_pixels")
    )
    native_threshold = (
        threshold_reference * recorded_manifest_ppi / reference_ppi
        if threshold_reference is not None
        and recorded_manifest_ppi is not None
        and reference_ppi is not None
        and reference_ppi > 0.0
        else None
    )
    expected_native_threshold = 3.0 * expected_manifest_ppi / 1000.0
    payload_ppi_passed = (
        recorded_manifest_ppi == expected_manifest_ppi
        and prepared_ppi_a == recorded_manifest_ppi
        and prepared_ppi_b == recorded_manifest_ppi
    )
    geometry_diagnostics_passed = (
        normalization == "ppi_to_reference"
        and reference_ppi == 1000.0
        and threshold_reference == 3.0
        and native_threshold == expected_native_threshold
    )
    return {
        "dataset": dataset,
        "manifest_ppi": recorded_manifest_ppi,
        "expected_manifest_ppi": expected_manifest_ppi,
        "prepared_payload_ppi_a": prepared_ppi_a,
        "prepared_payload_ppi_b": prepared_ppi_b,
        "both_prepared_payloads_carry_manifest_ppi": payload_ppi_passed,
        "coordinate_normalization": normalization,
        "expected_coordinate_normalization": "ppi_to_reference",
        "reference_ppi": reference_ppi,
        "expected_reference_ppi": 1000.0,
        "ransac_threshold_reference_pixels": threshold_reference,
        "expected_ransac_threshold_reference_pixels": 3.0,
        "native_equivalent_threshold_pixels": native_threshold,
        "expected_native_equivalent_threshold_pixels": expected_native_threshold,
        "geometry_diagnostics_passed": geometry_diagnostics_passed,
        "passed": payload_ppi_passed and geometry_diagnostics_passed,
    }


def _descriptor_count(outcome: Any) -> int:
    candidates = (
        outcome.diagnostics.get("descriptor_count"),
        outcome.representation.metadata.get("descriptor_count"),
        outcome.representation.metadata.get("keypoint_count"),
    )
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    payload = outcome.representation.payload
    descriptors = getattr(payload, "descriptors", None)
    if descriptors is not None:
        return int(np.asarray(descriptors).shape[0])
    raise HarrisZPlusPreflightError("Prepared representation does not expose descriptor count.")


def _nonnegative_integer_score(value: Any) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise HarrisZPlusPreflightError(
            f"HarrisZ+ raw score must be a finite non-negative integer; got {value!r}."
        )
    return int(numeric)


def _keypoints(result: Any) -> Sequence[Any]:
    points = getattr(result, "keypoints", None)
    if points is None:
        raise HarrisZPlusPreflightError("Detector result lacks keypoints.")
    return points


def _keypoint_record(point: Any) -> dict[str, Any]:
    if isinstance(point, Mapping):
        source = point
    elif is_dataclass(point):
        source = asdict(point)
    else:
        source = {
            name: getattr(point, name)
            for name in (
                "x",
                "y",
                "response",
                "scale_index",
                "sigma",
                "effective_support_diameter",
                "support_diameter",
                "size",
                "source_index",
            )
            if hasattr(point, name)
        }
    required = ("x", "y", "response", "scale_index", "sigma")
    if any(name not in source for name in required):
        raise HarrisZPlusPreflightError(f"Keypoint lacks required fields: {source}")
    return {
        "x": float(source["x"]),
        "y": float(source["y"]),
        "response": float(source["response"]),
        "scale_index": int(source["scale_index"]),
        "sigma": float(source["sigma"]),
        "support_diameter": float(
            source.get(
                "effective_support_diameter",
                source.get("support_diameter", source.get("size", 0.0)),
            )
        ),
        "size": float(
            source.get(
                "size",
                source.get("effective_support_diameter", source.get("support_diameter", 0.0)),
            )
        ),
        "source_index": int(source.get("source_index", -1)),
    }


def _keypoints_within_frozen_tolerance(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    return bool(
        int(first["scale_index"]) == int(second["scale_index"])
        and math.hypot(
            float(first["x"]) - float(second["x"]),
            float(first["y"]) - float(second["y"]),
        )
        <= SPATIAL_TOLERANCE_ORIGINAL_PX
        and abs(float(first["sigma"]) - float(second["sigma"])) <= SCALE_TOLERANCE
        and math.isclose(
            float(first["response"]),
            float(second["response"]),
            abs_tol=RESPONSE_ATOL,
            rel_tol=RESPONSE_RTOL,
        )
    )


def _detector_result_is_finite(result: Any) -> bool:
    for _, response in _ordered_response_maps(result):
        if not np.isfinite(np.asarray(response)).all():
            return False
    return all(
        all(math.isfinite(float(value)) for key, value in _keypoint_record(point).items() if key != "scale_index")
        for point in _keypoints(result)
    )


def _array_statistics(array: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(array, dtype=np.float64)
    if not values.size:
        return {"minimum": None, "maximum": None, "mean": None, "stddev": None}
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "stddev": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def _ordered_response_maps(result: Any) -> list[tuple[int, Any]]:
    maps = getattr(result, "response_maps", None)
    if maps is None:
        return []
    if isinstance(maps, Mapping):
        return [(int(index), maps[index]) for index in sorted(maps)]
    return list(enumerate(maps))


def _without_timing_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_timing_fields(item)
            for key, item in value.items()
            if not any(
                token in str(key).lower()
                for token in ("time", "timing", "_ms", "vram", "memory")
            )
        }
    if isinstance(value, list):
        return [_without_timing_fields(item) for item in value]
    return value


def _update_array_hash(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(contiguous.shape)))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))


def _update_value_hash(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        digest.update(b"array\0")
        _update_array_hash(digest, value)
    elif is_dataclass(value):
        digest.update(f"dataclass:{value.__class__.__qualname__}\0".encode("utf-8"))
        for field in fields(value):
            digest.update(field.name.encode("utf-8") + b"\0")
            _update_value_hash(digest, getattr(value, field.name))
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_value_hash(digest, str(key))
            _update_value_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}\0".encode("ascii"))
        for item in value:
            _update_value_hash(digest, item)
    elif isinstance(value, Path):
        _update_value_hash(digest, str(value))
    elif isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}\0".encode("ascii"))
        digest.update(value)
    elif isinstance(value, np.generic):
        _update_value_hash(digest, value.item())
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(canonical_json_bytes(value) + b"\0")
    elif hasattr(value, "__dict__"):
        _update_value_hash(digest, vars(value))
    else:
        raise HarrisZPlusPreflightError(
            f"Cannot hash prepared representation component {type(value).__name__}."
        )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarrisZPlusPreflightError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarrisZPlusPreflightError(f"JSON artifact must contain an object: {path}")
    return value


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise HarrisZPlusPreflightError(f"Required mapping {key!r} is missing.")
    return result


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_recorded_artifact_manifest(
    manifest_path: Path,
) -> list[tuple[Path, int, str]]:
    payload = _read_json(manifest_path)
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise HarrisZPlusPreflightError(
            f"Recorded artifact manifest has no file list: {manifest_path}"
        )
    validated: list[tuple[Path, int, str]] = []
    expected_logical_paths: set[str] = set()
    for record in raw_files:
        if not isinstance(record, Mapping):
            raise HarrisZPlusPreflightError(f"Invalid artifact record in {manifest_path}")
        raw_path = record.get("path")
        expected_sha = record.get("sha256")
        expected_size = record.get("size")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise HarrisZPlusPreflightError(f"Incomplete artifact record in {manifest_path}")
        if raw_path in expected_logical_paths:
            raise HarrisZPlusPreflightError(
                f"Duplicate artifact path {raw_path!r} in {manifest_path}"
            )
        expected_logical_paths.add(raw_path)
        path = (manifest_path.parent / raw_path).resolve()
        if not _is_relative_to(path, manifest_path.parent.resolve()):
            raise HarrisZPlusPreflightError(
                f"Recorded artifact escapes its protected namespace: {path}"
            )
        if not path.is_file():
            raise HarrisZPlusPreflightError(f"Recorded artifact is missing: {path}")
        size = path.stat().st_size
        if expected_size is not None and size != int(expected_size):
            raise HarrisZPlusPreflightError(f"Recorded artifact size changed: {path}")
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise HarrisZPlusPreflightError(f"Recorded artifact hash changed: {path}")
        validated.append((path, size, actual_sha))
    actual_logical_paths = {
        path.relative_to(manifest_path.parent).as_posix()
        for path in manifest_path.parent.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if actual_logical_paths != expected_logical_paths:
        added = sorted(actual_logical_paths - expected_logical_paths)
        removed = sorted(expected_logical_paths - actual_logical_paths)
        raise HarrisZPlusPreflightError(
            f"Protected artifact namespace inventory changed for {manifest_path}: "
            f"added={added[:8]}, removed={removed[:8]}."
        )
    tree_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in raw_files)
    ).hexdigest()
    if (
        payload.get("file_count") != len(raw_files)
        or payload.get("total_bytes") != sum(size for _, size, _ in validated)
        or payload.get("tree_sha256") != tree_sha
    ):
        raise HarrisZPlusPreflightError(
            f"Recorded artifact manifest aggregate identity changed: {manifest_path}"
        )
    return validated
