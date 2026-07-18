"""Frozen engineering preflight for the isolated PPI-aware HarrisZ+ v4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from ..contract import MethodExecutionError, OK, PREPARE_A_FAILURE
from ..hashing import (
    canonical_json_bytes,
    file_sha256,
    stable_config_hash,
    stable_hash,
)
from ..provenance import implementation_provenance
from .adapter_v4 import HarrisZPlusPpiAwareGeometricAdapter
from .detector_v4 import (
    detect_harriszplus_v4_cpu,
    detect_harriszplus_v4_cuda,
)
from .ppi_aware_v4 import (
    DECISION_THRESHOLD,
    METHOD_NAME,
    METHOD_VERSION,
    PARENT_METHOD_VERSION,
    PpiAwareHarrisZPlusConfig,
    build_physical_scale_contract,
)
from .preflight import (
    DATASETS,
    MINIMUM_RESPONSE_PIXEL_COVERAGE,
    SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE,
    _configure_and_describe_cuda,
    _effective_runner_config,
    _without_timing_fields,
    detector_result_sha256,
    fixed_validation_policy,
    synthetic_suite,
)
from . import preflight as freeze_support
from .preflight_v2 import (
    _cuda_repeat_comparison,
    _detector_absolute_conditions,
    _semantic_pair_comparison,
)
from .preflight_v3 import (
    PAIR_COLUMNS,
    EngineeringPair,
    _compare_prepared_pair,
    _real_image_metadata,
    _real_representation_comparison,
    _synthetic_comparison,
    aggregate_decision_equivalence,
    load_engineering_identities,
    load_engineering_pairs,
)
from .provenance import implementation_source_hashes
from .v4_integrity import compare_inventories, protected_v1_v3_inventory


DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
METHOD_RESULTS_RELATIVE = Path(
    "results/harriszplus_rootsift_geometric_ppi_aware_v4"
)
PILOT_RELATIVE = Path(
    "results/pilots/harriszplus_rootsift_geometric_ppi_aware_joint_500_v4"
)
PREFLIGHT_SCHEMA_VERSION = "harriszplus-engineering-preflight-v4"
FREEZE_SCHEMA_VERSION = "harriszplus-config-freeze-v4"
PASS_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_pass.json"
)
FAILURE_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_failure.json"
)
AUTHORIZATION_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight.json"
)
PHYSICAL_CONTRACT_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/physical_scale_contract_v4.json"
)
IDENTITIES_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "fixtures/engineering_identities_v4.csv"
)
PAIRS_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "fixtures/engineering_pairs_v4.csv"
)
BEFORE_INTEGRITY_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "integrity/v1_v3_before.json"
)
SELECTION_RELATIVE = Path(
    "results/pilots/sourceafis_joint_500_v1/selected_identities.csv"
)
EXPECTED_SELECTION_SHA256 = (
    "942363780986aab4b28df97ab67421ac8322ead5c9fd5131446f90eb8cdca7e9"
)
PAIR_CLASSES = ("plain_self", "roll_self", "genuine", "negative")
MAX_KEYPOINTS = 3000
MINIMUM_EXACT_SCORE_FRACTION = 0.95
MAXIMUM_RAW_SCORE_DELTA = 1


class HarrisZPlusPreflightV4Error(ValueError):
    """Raised when a frozen v4 gate fails."""


def _pretty_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _publish_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HarrisZPlusPreflightV4Error(
            f"Immutable v4 artifact already exists: {path}"
        )
    path.write_bytes(payload)


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(columns), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return stream.getvalue().encode("utf-8")


def _fixture_pairs(project_root: Path) -> tuple[list[Any], list[EngineeringPair]]:
    """Reuse the exact v3 identities/partners, expanding self to all ten."""

    identities = load_engineering_identities(project_root)
    v3_pairs = load_engineering_pairs(project_root)
    lookup = {
        (pair.dataset, pair.pair_class, pair.anchor_selection_index): pair
        for pair in v3_pairs
    }
    output: list[EngineeringPair] = []
    for dataset in DATASETS:
        ppi = 1000 if dataset == "sd300b" else 2000
        for pair_class in PAIR_CLASSES:
            for identity in identities:
                index = identity.selection_index
                if pair_class in ("genuine", "negative"):
                    source = lookup[(dataset, pair_class, index)]
                    output.append(
                        EngineeringPair(
                            pair_id=source.pair_id.replace(
                                "engineering_v3_", "engineering_v4_"
                            ),
                            dataset=dataset,
                            pair_class=pair_class,
                            subject_id_a=source.subject_id_a,
                            subject_id_b=source.subject_id_b,
                            canonical_finger_position=(
                                source.canonical_finger_position
                            ),
                            ppi=ppi,
                            raw_frgp_a=source.raw_frgp_a,
                            raw_frgp_b=source.raw_frgp_b,
                            path_a=source.path_a,
                            path_b=source.path_b,
                            anchor_selection_index=index,
                            negative_shift=source.negative_shift,
                        )
                    )
                    continue
                prefix = dataset
                path = getattr(
                    identity,
                    f"{prefix}_{'plain' if pair_class == 'plain_self' else 'roll'}_path",
                )
                raw = getattr(
                    identity,
                    f"{prefix}_{'plain' if pair_class == 'plain_self' else 'roll'}_raw_frgp",
                )
                output.append(
                    EngineeringPair(
                        pair_id=(
                            f"engineering_v4_{dataset}_{pair_class}_{index:02d}_"
                            f"{identity.subject_id}_"
                            f"{identity.canonical_finger_position:02d}"
                        ),
                        dataset=dataset,
                        pair_class=pair_class,
                        subject_id_a=identity.subject_id,
                        subject_id_b=identity.subject_id,
                        canonical_finger_position=(
                            identity.canonical_finger_position
                        ),
                        ppi=ppi,
                        raw_frgp_a=raw,
                        raw_frgp_b=raw,
                        path_a=path,
                        path_b=path,
                        anchor_selection_index=index,
                        negative_shift=0,
                    )
                )
    if len(identities) != 10 or len(output) != 80:
        raise HarrisZPlusPreflightV4Error(
            "v4 engineering fixture must contain 10 identities and 80 pairs."
        )
    return identities, output


def publish_engineering_fixtures(
    project_root: Path,
) -> dict[str, Any]:
    identities, pairs = _fixture_pairs(project_root)
    identity_source = (
        project_root
        / "results/harriszplus_rootsift_geometric_v3/fixtures/"
        "engineering_identities_v3.csv"
    )
    identity_target = project_root / IDENTITIES_RELATIVE
    pair_target = project_root / PAIRS_RELATIVE
    if not identity_target.exists():
        _publish_new(identity_target, identity_source.read_bytes())
    pair_rows = [
        {
            "pair_id": pair.pair_id,
            "dataset": pair.dataset,
            "pair_class": pair.pair_class,
            "subject_id_a": pair.subject_id_a,
            "subject_id_b": pair.subject_id_b,
            "canonical_finger_position": pair.canonical_finger_position,
            "ppi": pair.ppi,
            "raw_frgp_a": pair.raw_frgp_a,
            "raw_frgp_b": pair.raw_frgp_b,
            "path_a": str(pair.path_a),
            "path_b": str(pair.path_b),
            "anchor_selection_index": pair.anchor_selection_index,
            "negative_shift": pair.negative_shift,
        }
        for pair in pairs
    ]
    if not pair_target.exists():
        _publish_new(pair_target, _csv_bytes(pair_rows, PAIR_COLUMNS))
    return {
        "identity_count": len(identities),
        "pair_count": len(pairs),
        "pair_counts": {
            f"{dataset}/{pair_class}": sum(
                pair.dataset == dataset and pair.pair_class == pair_class
                for pair in pairs
            )
            for dataset in DATASETS
            for pair_class in PAIR_CLASSES
        },
        "identity_source_v3_sha256": file_sha256(identity_source),
        "identities": {
            "path": str(identity_target),
            "sha256": file_sha256(identity_target),
        },
        "pairs": {
            "path": str(pair_target),
            "sha256": file_sha256(pair_target),
        },
        "new_identity_selection_performed": False,
        "score_based_selection_performed": False,
    }


def _prepare_record_v4(
    adapter: HarrisZPlusPpiAwareGeometricAdapter,
    *,
    path: Path,
    metadata: Mapping[str, Any],
    expected_backend: str,
) -> dict[str, Any]:
    try:
        outcome = adapter.prepare(path, metadata)
    except MethodExecutionError as exc:
        return {
            "status": PREPARE_A_FAILURE,
            "failure_stage": "prepare",
            "error_code": exc.error_code,
            "error_message": exc.message,
            "outcome": None,
            "descriptor_count": 0,
            "descriptor_available": False,
            "representation_sha256": None,
            "deterministic_diagnostics_sha256": stable_hash(
                _without_timing_fields(exc.diagnostics)
            ),
        }
    payload = outcome.representation.payload
    points = np.asarray(payload.points)
    descriptors = np.asarray(payload.descriptors)
    sizes = np.asarray(payload.sizes)
    diagnostics = outcome.diagnostics
    total = diagnostics.get("vram_physical_total_bytes")
    allocated = diagnostics.get("peak_vram_allocated", 0)
    reserved = diagnostics.get("peak_vram_reserved", 0)
    memory_valid = (
        expected_backend != "cuda"
        or (
            diagnostics.get("vram_measurement_valid") is True
            and isinstance(total, int)
            and int(allocated) <= total
            and int(reserved) <= total
        )
    )
    return {
        "status": OK,
        "failure_stage": None,
        "outcome": outcome,
        "representation_sha256": diagnostics["representation_sha256"],
        "deterministic_diagnostics_sha256": stable_hash(
            _without_timing_fields(diagnostics)
        ),
        "descriptor_count": int(descriptors.shape[0]),
        "descriptor_available": int(descriptors.shape[0]) >= 2,
        "keypoint_count": int(points.shape[0]),
        "descriptors_finite": bool(
            np.isfinite(descriptors).all()
            and np.isfinite(points).all()
            and np.isfinite(sizes).all()
        ),
        "coordinates_within_image_bounds": bool(
            points.ndim == 2
            and points.shape[1] == 2
            and np.all(points[:, 0] >= 0.0)
            and np.all(points[:, 0] < int(payload.width))
            and np.all(points[:, 1] >= 0.0)
            and np.all(points[:, 1] < int(payload.height))
        ),
        "positive_scales": bool(np.all(sizes > 0.0)),
        "keypoint_cap_ok": int(points.shape[0]) <= MAX_KEYPOINTS,
        "detector_backend": diagnostics.get("detector_backend"),
        "hidden_cpu_fallback_absent": (
            diagnostics.get("detector_backend") == expected_backend
        ),
        "no_hidden_resize": diagnostics.get("no_hidden_resize") is True,
        "manifest_ppi": diagnostics.get("manifest_ppi"),
        "spatial_scale": diagnostics.get("spatial_scale"),
        "payload_manifest_ppi": payload.metadata.get("manifest_ppi"),
        "payload_spatial_scale": payload.metadata.get("spatial_scale"),
        "candidates_after_duplicate_removal": int(
            diagnostics.get("candidates_after_duplicate_removal", -1)
        ),
        "prepare_total_ms": diagnostics.get("prepare_total_ms"),
        "detector_gpu_wall_ms": diagnostics.get("detector_gpu_wall_ms"),
        "descriptor_cpu_ms": diagnostics.get("descriptor_cpu_ms"),
        "peak_vram_allocated": allocated,
        "peak_vram_reserved": reserved,
        "current_vram_allocated": diagnostics.get(
            "vram_allocated_after_bytes", 0
        ),
        "current_vram_reserved": diagnostics.get(
            "vram_reserved_after_bytes", 0
        ),
        "physical_vram_bytes": total,
        "vram_measurement_valid": memory_valid,
    }


def _json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "outcome"}


def _candidate_freeze_identity(
    project_root: Path,
    adapter: HarrisZPlusPpiAwareGeometricAdapter,
) -> dict[str, str]:
    metadata = adapter.metadata()
    runner_config = _effective_runner_config(metadata)
    runtime_identity = freeze_support._canonical_runtime_identity(
        project_root, metadata.runtime
    )
    _, _, implementation_hash = implementation_provenance(
        adapter=adapter,
        method_metadata=metadata,
        startup_validation={},
        runner_source_path=(
            project_root / "src/fingerprint_benchmark/runner.py"
        ).resolve(),
    )
    return {
        "canonical_config_hash": stable_config_hash(runner_config),
        "implementation_hash": implementation_hash,
        "runtime_identity_hash": runtime_identity["runtime_identity_hash"],
    }


def _performance_report(
    real_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        records = [
            row["cuda_first"]
            for row in real_rows
            if row["dataset"] == dataset
            and row["cuda_first"]["status"] == OK
        ]
        comparisons = [
            row["cuda_first"]
            for row in pair_rows
            if row["dataset"] == dataset
            and row["cuda_first"]["status"] == OK
        ]
        datasets[dataset] = {
            "median_prepare_ms": statistics.median(
                float(row["prepare_total_ms"]) for row in records
            )
            if records
            else None,
            "median_detector_ms": statistics.median(
                float(row["detector_gpu_wall_ms"]) for row in records
            )
            if records
            else None,
            "median_descriptor_ms": statistics.median(
                float(row["descriptor_cpu_ms"]) for row in records
            )
            if records
            else None,
            "median_compare_ms": statistics.median(
                float(row["compare_total_ms"]) for row in comparisons
            )
            if comparisons
            else None,
        }
    memory_rows = [
        record
        for row in real_rows
        for record in (row["cuda_first"], row["cuda_repeat"])
        if record["status"] == OK
    ]
    return {
        "information_only_not_correctness_gate": True,
        "datasets": datasets,
        "vram": {
            "peak_allocated_bytes": max(
                (int(row["peak_vram_allocated"]) for row in memory_rows),
                default=0,
            ),
            "peak_reserved_bytes": max(
                (int(row["peak_vram_reserved"]) for row in memory_rows),
                default=0,
            ),
            "maximum_current_allocated_bytes": max(
                (int(row["current_vram_allocated"]) for row in memory_rows),
                default=0,
            ),
            "maximum_current_reserved_bytes": max(
                (int(row["current_vram_reserved"]) for row in memory_rows),
                default=0,
            ),
            "physical_device_bytes": max(
                (int(row["physical_vram_bytes"]) for row in memory_rows),
                default=0,
            ),
            "allocated_and_reserved_summed": False,
            "peaks_across_processes_or_runs_summed": False,
            "all_measurements_valid": all(
                row["vram_measurement_valid"] for row in memory_rows
            ),
        },
    }


def run_engineering_preflight_v4(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Execute all frozen v4 gates without reading a 500-result bundle."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    selection_path = project_root / SELECTION_RELATIVE
    if file_sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise HarrisZPlusPreflightV4Error(
            "The frozen 500-identity selection changed."
        )
    before_path = project_root / BEFORE_INTEGRITY_RELATIVE
    before = json.loads(before_path.read_text(encoding="utf-8"))
    integrity_comparison = compare_inventories(
        before, protected_v1_v3_inventory(project_root)
    )
    integrity = {
        **integrity_comparison,
        "passed": integrity_comparison["byte_identical"],
        "before": before,
    }
    if not integrity["passed"]:
        raise HarrisZPlusPreflightV4Error(
            "Protected v1-v3 inputs changed before v4 preflight."
        )

    fixture = publish_engineering_fixtures(project_root)
    identities, pairs = _fixture_pairs(project_root)
    physical_contract = build_physical_scale_contract(
        PpiAwareHarrisZPlusConfig()
    )
    physical_path = project_root / PHYSICAL_CONTRACT_RELATIVE
    if not physical_path.exists():
        _publish_new(physical_path, _pretty_json(physical_contract))
    if (
        not physical_contract["passed"]
        or json.loads(physical_path.read_text(encoding="utf-8"))
        != physical_contract
    ):
        raise HarrisZPlusPreflightV4Error(
            "Physical-scale contract failed before matcher execution."
        )

    torch, environment = _configure_and_describe_cuda("cuda:0")
    operational = PpiAwareHarrisZPlusConfig().changed(
        backend="cuda", device="cuda:0"
    )
    cpu_operational = operational.changed(
        backend="reference_cpu", device=None
    )
    synthetic_rows: list[dict[str, Any]] = []
    for ppi in (1000, 2000):
        for name, source in synthetic_suite().items():
            source_u8 = np.ascontiguousarray(
                np.clip(np.rint(source), 0.0, 255.0), dtype=np.uint8
            )
            image = source_u8.astype(np.float32)
            doubled = cv2.resize(
                source_u8,
                (source_u8.shape[1] * 2, source_u8.shape[0] * 2),
                interpolation=cv2.INTER_LANCZOS4,
            ).astype(np.float32, copy=False)
            cpu = detect_harriszplus_v4_cpu(
                image,
                cpu_operational.runtime(ppi),
                doubled_image=doubled,
                return_response_maps=True,
            )
            cuda_first = detect_harriszplus_v4_cuda(
                image,
                operational.runtime(ppi),
                doubled_image=doubled,
                device="cuda:0",
                return_response_maps=True,
            )
            torch.cuda.synchronize("cuda:0")
            cuda_repeat = detect_harriszplus_v4_cuda(
                image,
                operational.runtime(ppi),
                doubled_image=doubled,
                device="cuda:0",
                return_response_maps=True,
            )
            torch.cuda.synchronize("cuda:0")
            comparison = _synthetic_comparison(cpu, cuda_first)
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
            repeat_hash = detector_result_sha256(cuda_repeat)
            repeat_exact = first_hash == repeat_hash
            flat_zero = (
                len(cpu.keypoints) == len(cuda_first.keypoints) == 0
                if name == "flat"
                else None
            )
            synthetic_rows.append(
                {
                    "case_id": f"synthetic:{ppi}:{name}",
                    "manifest_ppi": ppi,
                    "spatial_scale": ppi / 1000.0,
                    "cpu_keypoint_count": len(cpu.keypoints),
                    "cuda_keypoint_count": len(cuda_first.keypoints),
                    "cpu_cuda": comparison,
                    "cpu_absolute_conditions": cpu_absolute,
                    "cuda_absolute_conditions": cuda_absolute,
                    "cuda_first_sha256": first_hash,
                    "cuda_repeat_sha256": repeat_hash,
                    "cuda_repeat_exact": repeat_exact,
                    "flat_zero_keypoints": flat_zero,
                    "passed": bool(
                        comparison["passed"]
                        and cpu_absolute["passed"]
                        and cuda_absolute["passed"]
                        and repeat_exact
                        and flat_zero is not False
                    ),
                }
            )

    real_images = _real_image_metadata(pairs)
    cpu_adapter = HarrisZPlusPpiAwareGeometricAdapter(cpu_operational)
    cuda_adapter = HarrisZPlusPpiAwareGeometricAdapter(operational)
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    real_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    try:
        for image in real_images:
            cpu = _prepare_record_v4(
                cpu_adapter,
                path=image["path"],
                metadata=image["metadata"],
                expected_backend="reference_cpu",
            )
            cuda_first = _prepare_record_v4(
                cuda_adapter,
                path=image["path"],
                metadata=image["metadata"],
                expected_backend="cuda",
            )
            cuda_repeat = _prepare_record_v4(
                cuda_adapter,
                path=image["path"],
                metadata=image["metadata"],
                expected_backend="cuda",
            )
            comparison = _real_representation_comparison(cpu, cuda_first)
            expected_scale = image["ppi"] / 1000.0
            ppi_passed = all(
                row.get("manifest_ppi") == image["ppi"]
                and row.get("payload_manifest_ppi") == image["ppi"]
                and row.get("spatial_scale") == expected_scale
                and row.get("payload_spatial_scale") == expected_scale
                and row.get("no_hidden_resize") is True
                for row in (cpu, cuda_first, cuda_repeat)
            )
            repeat_exact = bool(
                cuda_first["status"] == cuda_repeat["status"]
                and cuda_first["representation_sha256"]
                == cuda_repeat["representation_sha256"]
                and cuda_first["deterministic_diagnostics_sha256"]
                == cuda_repeat["deterministic_diagnostics_sha256"]
            )
            key = str(image["path"].resolve()).lower()
            prepared[("cpu", key)] = cpu
            prepared[("cuda_first", key)] = cuda_first
            prepared[("cuda_repeat", key)] = cuda_repeat
            real_rows.append(
                {
                    "path": str(image["path"]),
                    "dataset": image["dataset"],
                    "subject_id": image["subject_id"],
                    "canonical_finger_position": image[
                        "canonical_finger_position"
                    ],
                    "expected_manifest_ppi": image["ppi"],
                    "expected_spatial_scale": expected_scale,
                    "cpu": _json_record(cpu),
                    "cuda_first": _json_record(cuda_first),
                    "cuda_repeat": _json_record(cuda_repeat),
                    "cpu_cuda": comparison,
                    "manifest_ppi_and_scale_passed": ppi_passed,
                    "cuda_repeat_exact": repeat_exact,
                    "passed": bool(
                        comparison["passed"]
                        and repeat_exact
                        and ppi_passed
                        and cuda_first["vram_measurement_valid"]
                        and cuda_repeat["vram_measurement_valid"]
                    ),
                }
            )

        for pair in pairs:
            a_key = str(pair.path_a.resolve()).lower()
            b_key = str(pair.path_b.resolve()).lower()
            cpu_result = _compare_prepared_pair(
                cpu_adapter,
                prepared[("cpu", a_key)],
                prepared[("cpu", b_key)],
            )
            cuda_first_result = _compare_prepared_pair(
                cuda_adapter,
                prepared[("cuda_first", a_key)],
                prepared[("cuda_first", b_key)],
            )
            cuda_repeat_result = _compare_prepared_pair(
                cuda_adapter,
                prepared[("cuda_repeat", a_key)],
                prepared[("cuda_repeat", b_key)],
            )
            semantic = _semantic_pair_comparison(
                cpu_result, cuda_first_result
            )
            repeat = _cuda_repeat_comparison(
                cuda_first_result, cuda_repeat_result
            )
            pair_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "dataset": pair.dataset,
                    "pair_class": pair.pair_class,
                    "subject_id_a": pair.subject_id_a,
                    "subject_id_b": pair.subject_id_b,
                    "canonical_finger_position": (
                        pair.canonical_finger_position
                    ),
                    "cpu": cpu_result,
                    "cuda_first": cuda_first_result,
                    "cuda_repeat": cuda_repeat_result,
                    "cpu_cuda_semantic_equivalence": semantic,
                    "cuda_repeat_exact": repeat,
                    "passed": bool(semantic["passed"] and repeat["passed"]),
                }
            )
        candidate_identity = _candidate_freeze_identity(
            project_root, cuda_adapter
        )
    finally:
        cpu_adapter.close()
        cuda_adapter.close()

    exact_rate = (
        sum(
            row["cpu_cuda_semantic_equivalence"]["raw_score_exact"]
            for row in pair_rows
        )
        / len(pair_rows)
    )
    maximum_delta = max(
        (
            row["cpu_cuda_semantic_equivalence"][
                "raw_score_absolute_delta"
            ]
            for row in pair_rows
            if row["cpu_cuda_semantic_equivalence"][
                "raw_score_absolute_delta"
            ]
            is not None
        ),
        default=0,
    )
    aggregate = aggregate_decision_equivalence(pair_rows)
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
        "decision_all_equal": all(
            row["cpu_cuda_semantic_equivalence"][
                "decision_threshold_4_equal"
            ]
            for row in pair_rows
        ),
        "exact_score_rate": exact_rate,
        "minimum_exact_score_rate": MINIMUM_EXACT_SCORE_FRACTION,
        "maximum_raw_score_absolute_delta": maximum_delta,
        "maximum_allowed_raw_score_absolute_delta": (
            MAXIMUM_RAW_SCORE_DELTA
        ),
        "aggregate_decision_equivalence": aggregate,
    }
    downstream["passed"] = bool(
        len(pair_rows) == 80
        and downstream["status_all_equal"]
        and downstream["failure_stage_all_equal"]
        and downstream["decision_all_equal"]
        and exact_rate >= MINIMUM_EXACT_SCORE_FRACTION
        and maximum_delta <= MAXIMUM_RAW_SCORE_DELTA
        and aggregate["passed"]
        and all(row["passed"] for row in pair_rows)
    )
    performance = _performance_report(real_rows, pair_rows)
    correctness = bool(
        len(identities) == 10
        and len(pairs) == 80
        and physical_contract["passed"]
        and integrity["passed"]
        and all(row["passed"] for row in synthetic_rows)
        and all(row["passed"] for row in real_rows)
        and downstream["passed"]
        and performance["vram"]["all_measurements_valid"]
    )
    v3_pass_path = (
        project_root
        / "results/harriszplus_rootsift_geometric_v3/preflight/"
        "engineering_preflight_pass.json"
    )
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "parent_method_version": PARENT_METHOD_VERSION,
        "purpose": "engineering_validation_only_not_parameter_selection",
        "passed": correctness,
        "pilot_500_authorized": correctness,
        "physical_scale_contract": {
            "path": str(physical_path),
            "sha256": file_sha256(physical_path),
            "passed": physical_contract["passed"],
            "validated_before_matcher": True,
        },
        "fixture": fixture,
        "synthetic_results": synthetic_rows,
        "real_image_count_after_deduplication": len(real_rows),
        "real_image_results": real_rows,
        "real_pair_results": pair_rows,
        "downstream_semantic_validation": downstream,
        "cuda_reproducibility": {
            "synthetic_all_exact": all(
                row["cuda_repeat_exact"] for row in synthetic_rows
            ),
            "real_representations_all_exact": all(
                row["cuda_repeat_exact"] for row in real_rows
            ),
            "real_comparisons_all_exact": all(
                row["cuda_repeat_exact"]["passed"] for row in pair_rows
            ),
        },
        "performance_projection": performance,
        "environment": environment,
        "candidate_freeze_identity": candidate_identity,
        "validation_policy": {
            **fixed_validation_policy(),
            "same_dataset_and_same_ppi_config_only": True,
            "b_c_keypoint_equality_required": False,
            "maximum_raw_score_delta": MAXIMUM_RAW_SCORE_DELTA,
            "minimum_exact_score_fraction": (
                MINIMUM_EXACT_SCORE_FRACTION
            ),
        },
        "ppi_coordinate_handling_all_passed": all(
            row["manifest_ppi_and_scale_passed"] for row in real_rows
        ),
        "device_binding": {
            "passed": environment.get("device") == "cuda:0",
            "canonical_device": "cuda:0",
            "observed_device": environment.get("device"),
        },
        "memory": {
            **performance["vram"],
            "passed": performance["vram"]["all_measurements_valid"],
        },
        "timing_synchronization": {
            "required_timing_fields_all_passed": all(
                row["cuda_first"].get("prepare_total_ms") is not None
                and row["cuda_first"].get("detector_gpu_wall_ms") is not None
                and row["cuda_first"].get("descriptor_cpu_ms") is not None
                for row in real_rows
            ),
            "explicit_cuda_synchronization": True,
        },
        "integrity": integrity,
        "development_diagnostic_v3_reference": {
            "path": str(v3_pass_path),
            "sha256": file_sha256(v3_pass_path),
            "used_for_parameter_selection": False,
        },
        "no_parameter_tuning_performed": True,
        "no_tolerance_changed_after_result": True,
        "no_500_result_observed": True,
        "selection_file_read_for_scores": False,
        "sourceafis_or_sift_rerun": False,
    }
    report["report_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def v4_validation_policy() -> dict[str, Any]:
    return {
        **fixed_validation_policy(),
        "schema_version": "harriszplus-functional-validation-policy-v4",
        "same_dataset_and_same_ppi_config_only": True,
        "b_c_keypoint_equality_required": False,
        "maximum_downstream_raw_score_delta": MAXIMUM_RAW_SCORE_DELTA,
        "minimum_exact_downstream_score_fraction": (
            MINIMUM_EXACT_SCORE_FRACTION
        ),
        "decision_threshold": DECISION_THRESHOLD,
        "decision_equality_required": True,
        "physical_scale_contract_required_before_matcher": True,
        "auto_relaxation_allowed": False,
        "frozen_before_500_results": True,
    }


def publish_engineering_preflight_v4(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    pass_path = project_root / PASS_RELATIVE
    failure_path = project_root / FAILURE_RELATIVE
    if pass_path.exists() or failure_path.exists():
        raise HarrisZPlusPreflightV4Error(
            "An immutable v4 pass/failure artifact already exists."
        )
    try:
        report = run_engineering_preflight_v4(
            project_root=project_root, data_root=data_root
        )
    except Exception as exc:
        failure = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "method_name": METHOD_NAME,
            "method_version": METHOD_VERSION,
            "passed": False,
            "pilot_500_authorized": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "no_parameter_change_authorized": True,
            "pilot_500_must_not_run": True,
        }
        _publish_new(failure_path, _pretty_json(failure))
        raise
    target = pass_path if report["passed"] else failure_path
    _publish_new(target, _pretty_json(report))
    if not report["passed"]:
        raise HarrisZPlusPreflightV4Error(
            "v4 preflight failed; 500 pilot is not authorized."
        )
    return {**report, "path": str(pass_path), "sha256": file_sha256(pass_path)}


def require_pilot_authorization(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    pass_path = project_root / PASS_RELATIVE
    report = json.loads(pass_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or report.get("passed") is not True
        or report.get("pilot_500_authorized") is not True
        or report.get("no_parameter_tuning_performed") is not True
        or report.get("no_500_result_observed") is not True
    ):
        raise HarrisZPlusPreflightV4Error(
            "A valid frozen v4 preflight pass is required."
        )
    current = protected_v1_v3_inventory(project_root)
    if not compare_inventories(
        report["integrity"]["before"], current
    )["byte_identical"]:
        raise HarrisZPlusPreflightV4Error(
            "v1-v3 changed after the v4 preflight."
        )
    return {**report, "sha256": file_sha256(pass_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen PPI-aware HarrisZ+ v4 preflight."
    )
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT_ROOT
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = publish_engineering_preflight_v4(
            project_root=args.project_root,
            data_root=args.data_root,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_RELATIVE",
    "FAILURE_RELATIVE",
    "FREEZE_SCHEMA_VERSION",
    "METHOD_RESULTS_RELATIVE",
    "PASS_RELATIVE",
    "PILOT_RELATIVE",
    "PREFLIGHT_SCHEMA_VERSION",
    "publish_engineering_preflight_v4",
    "require_pilot_authorization",
    "run_engineering_preflight_v4",
    "v4_validation_policy",
]
