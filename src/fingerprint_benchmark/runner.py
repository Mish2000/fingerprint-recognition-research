"""Serial pairwise benchmark-v2 runner and strict bundle validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
from typing import Any, Callable

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from .bundle import (
    BundlePublicationError,
    create_candidate_directory,
    discard_candidate_directory,
    publish_candidate_directory,
)
from .contract import (
    BENCHMARK_CONTRACT_VERSION,
    COMPARISON_FAILURE,
    OK,
    PREPARE_A_FAILURE,
    PREPARE_B_FAILURE,
    RESULT_SCHEMA_VERSION,
    RESULT_STATUSES,
    TIMING_MODE_COLD_PAIR,
    WARMUP_POLICY,
    BenchmarkRunSpec,
    CompareOutcome,
    MethodAdapter,
    MethodExecutionError,
    MethodMetadata,
    PrepareOutcome,
)
from .hashing import canonical_json_bytes, file_sha256, stable_config_hash, stable_hash
from .io import write_csv_atomic, write_json_atomic
from .manifest import PairRecord
from .preflight import preflight_manifest
from .provenance import implementation_provenance


RESULT_FILENAME = "pairs.csv"
METADATA_FILENAME = "run_metadata.json"
RUN_METADATA_SCHEMA_VERSION = "pairwise-run-metadata-v2"
TIMING_TOLERANCE_MS = 0.001

RESULT_COLUMNS = [
    "pair_id",
    "dataset",
    "protocol",
    "subject_id",
    "canonical_finger_position",
    "method",
    "method_version",
    "benchmark_contract_version",
    "result_schema_version",
    "config_hash",
    "implementation_hash",
    "manifest_sha256",
    "score_direction",
    "score_semantics",
    "raw_score",
    "prepare_a_ms",
    "prepare_b_ms",
    "compare_ms",
    "method_prepare_a_ms",
    "method_prepare_b_ms",
    "method_compare_ms",
    "total_ms",
    "prepare_a_diagnostics",
    "prepare_b_diagnostics",
    "compare_diagnostics",
    "status",
    "error_code",
    "error_message",
]


class ResultValidationError(ValueError):
    """Raised when a result artifact violates benchmark-v2."""


class BundleValidationError(ValueError):
    """Raised when result and metadata files do not form one valid bundle."""


@dataclass(frozen=True)
class BenchmarkRunContext:
    spec: BenchmarkRunSpec
    method_metadata: MethodMetadata
    config: dict[str, Any]
    implementation_provenance: dict[str, Any]
    implementation_hash_components: dict[str, Any]
    bundle_directory: Path


def prepare_run_context(
    *,
    manifest_path: Path,
    expected_dataset: str,
    expected_protocol: str,
    adapter: MethodAdapter,
    results_root: Path,
    startup_validation: dict[str, Any] | None = None,
    bundle_directory: Path | None = None,
) -> BenchmarkRunContext:
    """Construct deterministic config and implementation identity before timing."""

    manifest_path = manifest_path.resolve()
    method_metadata = adapter.metadata()
    config = {
        **method_metadata.config,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "method": method_metadata.method,
        "method_version": method_metadata.method_version,
        "score_direction": method_metadata.score_direction,
        "score_semantics": method_metadata.score_semantics,
        "timing_mode": TIMING_MODE_COLD_PAIR,
        "warm_up_policy": WARMUP_POLICY,
    }
    config_hash = stable_config_hash(config)
    provenance, implementation_components, implementation_hash = implementation_provenance(
        adapter=adapter,
        method_metadata=method_metadata,
        startup_validation=startup_validation or {},
        runner_source_path=Path(__file__).resolve(),
    )
    spec = BenchmarkRunSpec(
        expected_dataset=expected_dataset,
        expected_protocol=expected_protocol,
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        method=method_metadata.method,
        method_version=method_metadata.method_version,
        benchmark_contract_version=BENCHMARK_CONTRACT_VERSION,
        config_hash=config_hash,
        implementation_hash=implementation_hash,
    )
    final_directory = bundle_directory or (
        results_root
        / expected_dataset
        / expected_protocol
        / method_metadata.method
        / BENCHMARK_CONTRACT_VERSION
        / config_hash
    )
    return BenchmarkRunContext(
        spec=spec,
        method_metadata=method_metadata,
        config=config,
        implementation_provenance=provenance,
        implementation_hash_components=implementation_components,
        bundle_directory=final_directory.resolve(),
    )


def run_benchmark_manifest(
    *,
    manifest_path: Path,
    adapter: MethodAdapter,
    expected_dataset: str,
    expected_protocol: str,
    results_root: Path = Path("results"),
    startup_validation: dict[str, Any] | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    dedicated_validator: Callable[[Path, Path], Any] | None = None,
    skip_existing: bool = False,
    bundle_directory: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Validate, warm, execute, validate a candidate, then publish one bundle."""

    context = prepare_run_context(
        manifest_path=manifest_path,
        expected_dataset=expected_dataset,
        expected_protocol=expected_protocol,
        adapter=adapter,
        results_root=results_root,
        startup_validation=startup_validation,
        bundle_directory=bundle_directory,
    )
    pairs, manifest_validation = preflight_manifest(
        manifest_path=manifest_path.resolve(),
        expected_dataset=expected_dataset,
        expected_protocol=expected_protocol,
        run_spec=context.spec,
        data_root=data_root,
        dedicated_validator=dedicated_validator,
    )

    if context.bundle_directory.exists():
        if not skip_existing:
            raise BundlePublicationError(
                "A benchmark-v2 bundle already exists and will not be overwritten: "
                f"{context.bundle_directory}"
            )
        return validate_result_bundle(
            context.bundle_directory,
            manifest_records=pairs,
            run_spec=context.spec,
            score_direction=context.method_metadata.score_direction,
            score_semantics=context.method_metadata.score_semantics,
        )

    warm_up = _run_warm_up(pairs, adapter)
    execution_start = perf_counter_ns()
    rows: list[dict[str, str]] = []
    for index, pair in enumerate(pairs, start=1):
        rows.append(
            _execute_pair(
                pair,
                adapter,
                run_spec=context.spec,
                method_metadata=context.method_metadata,
            )
        )
        if progress_callback is not None and (index == 1 or index % 50 == 0 or index == len(pairs)):
            progress_callback(index, len(pairs))
    execution_wall_ms = (perf_counter_ns() - execution_start) / 1_000_000.0

    candidate = create_candidate_directory(context.bundle_directory)
    try:
        result_path = candidate / RESULT_FILENAME
        metadata_path = candidate / METADATA_FILENAME
        write_csv_atomic(rows, result_path, RESULT_COLUMNS)
        validate_result_contract(
            result_path,
            manifest_records=pairs,
            run_spec=context.spec,
            score_direction=context.method_metadata.score_direction,
            score_semantics=context.method_metadata.score_semantics,
        )
        metadata = _run_metadata(
            context=context,
            pairs=pairs,
            rows=rows,
            result_path=result_path,
            manifest_validation=manifest_validation,
            warm_up=warm_up,
            execution_wall_ms=execution_wall_ms,
            startup_validation=startup_validation or {},
        )
        write_json_atomic(metadata, metadata_path)
        validate_result_bundle(
            candidate,
            manifest_records=pairs,
            run_spec=context.spec,
            score_direction=context.method_metadata.score_direction,
            score_semantics=context.method_metadata.score_semantics,
        )
        publish_candidate_directory(candidate, context.bundle_directory)
        return metadata
    finally:
        discard_candidate_directory(candidate)


def validate_result_contract(
    result_path: Path,
    *,
    manifest_records: list[PairRecord],
    run_spec: BenchmarkRunSpec,
    score_direction: str,
    score_semantics: str,
) -> list[dict[str, str]]:
    """Validate every result field against the exact source manifest and spec."""

    with result_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ResultValidationError(
                f"Result schema mismatch. Expected columns {RESULT_COLUMNS}, got {reader.fieldnames}."
            )
        rows = list(reader)

    if len(rows) != len(manifest_records):
        raise ResultValidationError(
            f"Result row count mismatch: expected {len(manifest_records)}, got {len(rows)}."
        )
    if any(None in row for row in rows):
        raise ResultValidationError("Result contains extra unnamed CSV values.")

    result_pair_ids = [row["pair_id"] for row in rows]
    manifest_pair_ids = [pair.pair_id for pair in manifest_records]
    if result_pair_ids != manifest_pair_ids:
        raise ResultValidationError("Result pair_id sequence does not exactly match the manifest.")
    if len(result_pair_ids) != len(set(result_pair_ids)):
        raise ResultValidationError("Result contains duplicate pair_id values.")

    for pair, row in zip(manifest_records, rows, strict=True):
        _validate_result_row(
            pair,
            row,
            run_spec=run_spec,
            score_direction=score_direction,
            score_semantics=score_semantics,
        )
    return rows


def validate_result_bundle(
    bundle_directory: Path,
    *,
    manifest_records: list[PairRecord],
    run_spec: BenchmarkRunSpec,
    score_direction: str,
    score_semantics: str,
) -> dict[str, Any]:
    """Validate metadata, result bytes, and all manifest identities together."""

    result_path = bundle_directory / RESULT_FILENAME
    metadata_path = bundle_directory / METADATA_FILENAME
    if not result_path.is_file() or not metadata_path.is_file():
        raise BundleValidationError(
            f"Bundle must contain {RESULT_FILENAME} and {METADATA_FILENAME}: {bundle_directory}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"Cannot read bundle metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise BundleValidationError("Run metadata must be a JSON object.")

    rows = validate_result_contract(
        result_path,
        manifest_records=manifest_records,
        run_spec=run_spec,
        score_direction=score_direction,
        score_semantics=score_semantics,
    )
    checks = {
        "metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dataset": run_spec.expected_dataset,
        "protocol": run_spec.expected_protocol,
        "method": run_spec.method,
        "method_version": run_spec.method_version,
        "score_direction": score_direction,
        "score_semantics": score_semantics,
        "config_hash": run_spec.config_hash,
        "implementation_hash": run_spec.implementation_hash,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise BundleValidationError(
                f"Metadata {key} mismatch: expected {expected!r}, got {metadata.get(key)!r}."
            )
    if metadata.get("run_spec") != run_spec.as_dict():
        raise BundleValidationError("Metadata run_spec does not match the current run specification.")
    if stable_config_hash(_required_dict(metadata, "config")) != run_spec.config_hash:
        raise BundleValidationError("Metadata config content does not match config_hash.")
    if stable_hash(_required_dict(metadata, "implementation_hash_components")) != run_spec.implementation_hash:
        raise BundleValidationError("Metadata implementation components do not match implementation_hash.")

    manifest_metadata = _required_dict(metadata, "manifest")
    if manifest_metadata.get("sha256") != run_spec.manifest_sha256:
        raise BundleValidationError("Metadata manifest SHA-256 is stale.")
    if manifest_metadata.get("row_count") != len(manifest_records):
        raise BundleValidationError("Metadata manifest row count is stale.")
    if Path(str(manifest_metadata.get("path"))).resolve() != run_spec.manifest_path.resolve():
        raise BundleValidationError("Metadata manifest path does not match the current manifest.")
    if file_sha256(run_spec.manifest_path) != run_spec.manifest_sha256:
        raise BundleValidationError("Current manifest bytes do not match the run specification.")

    result_metadata = _required_dict(metadata, "result")
    if result_metadata.get("sha256") != file_sha256(result_path):
        raise BundleValidationError("Metadata result SHA-256 does not match pairs.csv.")
    if result_metadata.get("row_count") != len(rows):
        raise BundleValidationError("Metadata result row count does not match pairs.csv.")
    if result_metadata.get("score_payload_sha256") != score_payload_sha256(rows):
        raise BundleValidationError("Metadata score_payload_sha256 does not match score rows.")
    return metadata


def score_payload_sha256(rows: list[dict[str, str]]) -> str:
    projection = [
        {
            "pair_id": row["pair_id"],
            "status": row["status"],
            "raw_score": row["raw_score"],
            "error_code": row["error_code"],
        }
        for row in rows
    ]
    return hashlib.sha256(canonical_json_bytes(projection)).hexdigest()


def _run_warm_up(pairs: list[PairRecord], adapter: MethodAdapter) -> dict[str, Any]:
    pair = pairs[0]
    start = perf_counter_ns()
    prepared_a = _require_prepare_outcome(adapter.prepare(pair.path_a, pair.image_metadata_a()))
    prepared_b = _require_prepare_outcome(adapter.prepare(pair.path_b, pair.image_metadata_b()))
    compared = _require_compare_outcome(
        adapter.compare(prepared_a.representation, prepared_b.representation)
    )
    if not math.isfinite(float(compared.raw_score)):
        raise MethodExecutionError("non_finite_raw_score", "Warm-up comparison returned a non-finite raw score.")
    duration_ms = (perf_counter_ns() - start) / 1_000_000.0
    return {
        "policy": WARMUP_POLICY,
        "pair_ids": [pair.pair_id],
        "operation_count": 3,
        "prepare_operation_count": 2,
        "compare_operation_count": 1,
        "duration_ms": float(_format_timing(duration_ms)),
        "included_in_result_rows": False,
    }


def _execute_pair(
    pair: PairRecord,
    adapter: MethodAdapter,
    *,
    run_spec: BenchmarkRunSpec,
    method_metadata: MethodMetadata,
) -> dict[str, str]:
    row = _base_row(pair, run_spec, method_metadata)
    total_start = perf_counter_ns()

    prepare_a_start = perf_counter_ns()
    try:
        prepare_a = _require_prepare_outcome(adapter.prepare(pair.path_a, pair.image_metadata_a()))
        row["prepare_a_ms"] = _elapsed_ms(prepare_a_start)
        row["method_prepare_a_ms"] = _optional_timing(prepare_a.method_internal_ms)
        row["prepare_a_diagnostics"] = _diagnostics_json(prepare_a.diagnostics)
    except MethodExecutionError as exc:
        row["prepare_a_ms"] = _elapsed_ms(prepare_a_start)
        _record_failure_operation(row, "prepare_a", exc)
        return _failure_row(row, total_start, PREPARE_A_FAILURE, exc)

    prepare_b_start = perf_counter_ns()
    try:
        prepare_b = _require_prepare_outcome(adapter.prepare(pair.path_b, pair.image_metadata_b()))
        row["prepare_b_ms"] = _elapsed_ms(prepare_b_start)
        row["method_prepare_b_ms"] = _optional_timing(prepare_b.method_internal_ms)
        row["prepare_b_diagnostics"] = _diagnostics_json(prepare_b.diagnostics)
    except MethodExecutionError as exc:
        row["prepare_b_ms"] = _elapsed_ms(prepare_b_start)
        _record_failure_operation(row, "prepare_b", exc)
        return _failure_row(row, total_start, PREPARE_B_FAILURE, exc)

    compare_start = perf_counter_ns()
    try:
        comparison = _require_compare_outcome(
            adapter.compare(prepare_a.representation, prepare_b.representation)
        )
        row["compare_ms"] = _elapsed_ms(compare_start)
        row["method_compare_ms"] = _optional_timing(comparison.method_internal_ms)
        row["compare_diagnostics"] = _diagnostics_json(comparison.diagnostics)
    except MethodExecutionError as exc:
        row["compare_ms"] = _elapsed_ms(compare_start)
        _record_failure_operation(row, "compare", exc)
        return _failure_row(row, total_start, COMPARISON_FAILURE, exc)

    score = float(comparison.raw_score)
    if not math.isfinite(score):
        return _failure_row(
            row,
            total_start,
            COMPARISON_FAILURE,
            MethodExecutionError("non_finite_raw_score", "Method returned a non-finite raw score."),
        )

    row["raw_score"] = repr(score)
    row["total_ms"] = _elapsed_ms(total_start)
    row["status"] = OK
    return row


def _base_row(
    pair: PairRecord,
    run_spec: BenchmarkRunSpec,
    method_metadata: MethodMetadata,
) -> dict[str, str]:
    row = {column: "" for column in RESULT_COLUMNS}
    row.update(
        {
            "pair_id": pair.pair_id,
            "dataset": pair.dataset,
            "protocol": pair.protocol,
            "subject_id": pair.subject_id,
            "canonical_finger_position": str(pair.canonical_finger_position),
            "method": run_spec.method,
            "method_version": run_spec.method_version,
            "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "config_hash": run_spec.config_hash,
            "implementation_hash": run_spec.implementation_hash,
            "manifest_sha256": run_spec.manifest_sha256,
            "score_direction": method_metadata.score_direction,
            "score_semantics": method_metadata.score_semantics,
        }
    )
    return row


def _failure_row(
    row: dict[str, str],
    total_start: int,
    status: str,
    exc: MethodExecutionError,
) -> dict[str, str]:
    row["raw_score"] = ""
    row["total_ms"] = _elapsed_ms(total_start)
    row["status"] = status
    row["error_code"] = str(exc.error_code).strip()
    row["error_message"] = str(exc.message).strip()
    return row


def _record_failure_operation(row: dict[str, str], prefix: str, exc: MethodExecutionError) -> None:
    row[f"method_{prefix}_ms"] = _optional_timing(exc.method_internal_ms)
    row[f"{prefix}_diagnostics"] = _diagnostics_json(exc.diagnostics)


def _require_prepare_outcome(value: Any) -> PrepareOutcome:
    if not isinstance(value, PrepareOutcome):
        raise TypeError(f"MethodAdapter.prepare() returned {type(value).__name__}; expected PrepareOutcome.")
    _validate_optional_timing_value(value.method_internal_ms, "prepare method_internal_ms")
    return value


def _require_compare_outcome(value: Any) -> CompareOutcome:
    if not isinstance(value, CompareOutcome):
        raise TypeError(f"MethodAdapter.compare() returned {type(value).__name__}; expected CompareOutcome.")
    _validate_optional_timing_value(value.method_internal_ms, "compare method_internal_ms")
    return value


def _validate_optional_timing_value(value: float | None, label: str) -> None:
    if value is None:
        return
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise MethodExecutionError("invalid_method_timing", f"{label} must be finite and non-negative.")


def _optional_timing(value: float | None) -> str:
    if value is None:
        return ""
    _validate_optional_timing_value(value, "method_internal_ms")
    return _format_timing(float(value))


def _diagnostics_json(value: dict[str, Any]) -> str:
    if not isinstance(value, dict):
        raise TypeError("Operation diagnostics must be a dict.")
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _elapsed_ms(start_ns: int) -> str:
    return _format_timing((perf_counter_ns() - start_ns) / 1_000_000.0)


def _format_timing(value: float) -> str:
    return format(value, ".9g")


def _run_metadata(
    *,
    context: BenchmarkRunContext,
    pairs: list[PairRecord],
    rows: list[dict[str, str]],
    result_path: Path,
    manifest_validation: dict[str, Any],
    warm_up: dict[str, Any],
    execution_wall_ms: float,
    startup_validation: dict[str, Any],
) -> dict[str, Any]:
    status_counts = {status: 0 for status in RESULT_STATUSES}
    for row in rows:
        status_counts[row["status"]] += 1
    logical_result_path = context.bundle_directory / RESULT_FILENAME
    return {
        "metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": context.spec.expected_dataset,
        "protocol": context.spec.expected_protocol,
        "method": context.spec.method,
        "method_version": context.spec.method_version,
        "score_direction": context.method_metadata.score_direction,
        "score_semantics": context.method_metadata.score_semantics,
        "timing_mode": TIMING_MODE_COLD_PAIR,
        "config": context.config,
        "config_hash": context.spec.config_hash,
        "implementation_hash": context.spec.implementation_hash,
        "implementation_hash_components": context.implementation_hash_components,
        "implementation_provenance": context.implementation_provenance,
        "run_spec": context.spec.as_dict(),
        "manifest": {
            "path": str(context.spec.manifest_path),
            "row_count": len(pairs),
            "sha256": context.spec.manifest_sha256,
            "dedicated_validator_result": manifest_validation,
        },
        "result": {
            "path": str(logical_result_path),
            "relative_path": RESULT_FILENAME,
            "row_count": len(rows),
            "sha256": file_sha256(result_path),
            "score_payload_sha256": score_payload_sha256(rows),
        },
        "success_count": status_counts[OK],
        "failure_counts": {
            PREPARE_A_FAILURE: status_counts[PREPARE_A_FAILURE],
            PREPARE_B_FAILURE: status_counts[PREPARE_B_FAILURE],
            COMPARISON_FAILURE: status_counts[COMPARISON_FAILURE],
        },
        "warm_up": warm_up,
        "execution_wall_ms": float(_format_timing(execution_wall_ms)),
        "external_runtime": context.method_metadata.runtime,
        "startup_validation": startup_validation,
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "python_version": sys.version.split()[0],
            "system": platform.system(),
        },
        "run_isolation": {
            "scope": "one_dataset_protocol",
            "dedicated_managed_runtime": bool(startup_validation.get("managed_by_runner")),
            "warm_state_shared_across_protocol_runs": False,
        },
        "sd300_dependency_rule": {
            "sd300b_sd300c_are_resolution_conditions_not_independent_subject_populations": True,
            "future_splits_must_be_joint_by_subject_across_datasets": True,
            "dataset_and_ppi_condition_recorded_separately": True,
        },
    }


def _validate_result_row(
    pair: PairRecord,
    row: dict[str, str],
    *,
    run_spec: BenchmarkRunSpec,
    score_direction: str,
    score_semantics: str,
) -> None:
    pair_id = pair.pair_id
    expected_identity = {
        "pair_id": pair.pair_id,
        "dataset": pair.dataset,
        "protocol": pair.protocol,
        "subject_id": pair.subject_id,
        "canonical_finger_position": str(pair.canonical_finger_position),
    }
    for key, expected in expected_identity.items():
        if row[key] != expected:
            raise ResultValidationError(
                f"Result {key} mismatch for pair {pair_id}: expected {expected!r}, got {row[key]!r}."
            )
    expected_constants = {
        "method": run_spec.method,
        "method_version": run_spec.method_version,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "config_hash": run_spec.config_hash,
        "implementation_hash": run_spec.implementation_hash,
        "manifest_sha256": run_spec.manifest_sha256,
        "score_direction": score_direction,
        "score_semantics": score_semantics,
    }
    for key, expected in expected_constants.items():
        if row[key] != expected:
            raise ResultValidationError(
                f"Result {key} mismatch for pair {pair_id}: expected {expected!r}, got {row[key]!r}."
            )

    status = row["status"]
    if status not in RESULT_STATUSES:
        raise ResultValidationError(f"Invalid result status {status!r} for pair {pair_id}.")
    if status == OK:
        score = _required_number(row, "raw_score", pair_id)
        if row["raw_score"] != repr(score):
            raise ResultValidationError(
                f"raw_score is not in round-trip-safe canonical form for pair {pair_id}."
            )
        if row["error_code"] or row["error_message"]:
            raise ResultValidationError(f"Successful pair {pair_id} contains error fields.")
    else:
        if row["raw_score"]:
            raise ResultValidationError(f"Failed pair {pair_id} contains a raw_score.")
        if not row["error_code"].strip() or not row["error_message"].strip():
            raise ResultValidationError(f"Failed pair {pair_id} must contain error_code and error_message.")

    executed = {
        "prepare_a": True,
        "prepare_b": status != PREPARE_A_FAILURE,
        "compare": status in (OK, COMPARISON_FAILURE),
    }
    wall_values: dict[str, float] = {}
    component_values: list[float] = []
    for operation, was_executed in executed.items():
        wall_column = f"{operation}_ms"
        method_column = f"method_{operation}_ms"
        diagnostics_column = f"{operation}_diagnostics"
        if not was_executed:
            if row[wall_column] or row[method_column] or row[diagnostics_column]:
                raise ResultValidationError(
                    f"Pair {pair_id} contains {operation} output after an earlier failure."
                )
            continue
        wall = _required_number(row, wall_column, pair_id, nonnegative=True)
        wall_values[operation] = wall
        component_values.append(wall)
        if row[method_column]:
            component_values.append(
                _required_number(row, method_column, pair_id, nonnegative=True)
            )
        _required_diagnostics(row[diagnostics_column], diagnostics_column, pair_id)

    total = _required_number(row, "total_ms", pair_id, nonnegative=True)
    for component in component_values:
        if total + TIMING_TOLERANCE_MS < component:
            raise ResultValidationError(
                f"total_ms is smaller than a timing component for pair {pair_id}."
            )
    if status == OK:
        wall_sum = sum(wall_values.values())
        if total + TIMING_TOLERANCE_MS < wall_sum:
            raise ResultValidationError(
                f"total_ms is smaller than prepare_a_ms + prepare_b_ms + compare_ms for pair {pair_id}."
            )


def _required_number(
    row: dict[str, str],
    column: str,
    pair_id: str,
    *,
    nonnegative: bool = False,
) -> float:
    raw = row[column]
    if raw == "":
        raise ResultValidationError(f"Missing {column} for pair {pair_id}.")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ResultValidationError(f"Non-numeric {column} for pair {pair_id}.") from exc
    if not math.isfinite(value) or (nonnegative and value < 0):
        condition = "finite and non-negative" if nonnegative else "finite"
        raise ResultValidationError(f"{column} must be {condition} for pair {pair_id}.")
    return value


def _required_diagnostics(raw: str, column: str, pair_id: str) -> None:
    if not raw:
        raise ResultValidationError(f"Missing {column} for executed operation on pair {pair_id}.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResultValidationError(f"Invalid JSON in {column} for pair {pair_id}.") from exc
    if not isinstance(value, dict):
        raise ResultValidationError(f"{column} must encode a JSON object for pair {pair_id}.")


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise BundleValidationError(f"Metadata field {key!r} must be an object.")
    return value


def summarize_result_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == OK]
    return {
        "manifest_pair_count": len(rows),
        "result_row_count": len(rows),
        "ok_count": len(ok_rows),
        "failure_count": len(rows) - len(ok_rows),
        "failure_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in (PREPARE_A_FAILURE, PREPARE_B_FAILURE, COMPARISON_FAILURE)
        },
        "raw_score": _numeric_summary(row["raw_score"] for row in ok_rows),
        **{
            column: _timing_summary(row[column] for row in rows)
            for column in (
                "prepare_a_ms",
                "prepare_b_ms",
                "compare_ms",
                "method_prepare_a_ms",
                "method_prepare_b_ms",
                "method_compare_ms",
                "total_ms",
            )
        },
    }


def read_result_rows(result_path: Path) -> list[dict[str, str]]:
    with result_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric_summary(values) -> dict[str, float | None]:
    numbers = [float(value) for value in values if value not in (None, "")]
    if not numbers:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(numbers),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
    }


def _timing_summary(values) -> dict[str, float | None]:
    numbers = [float(value) for value in values if value not in (None, "")]
    if not numbers:
        return {"median": None, "p95": None}
    numbers.sort()
    index = min(len(numbers) - 1, math.ceil(len(numbers) * 0.95) - 1)
    return {"median": statistics.median(numbers), "p95": numbers[index]}
