"""Deterministic, isolated SourceAFIS reproducibility audit workflow.

The ``prepare`` and ``compare`` commands never invoke SourceAFIS.  Only the
explicit ``run`` command starts the managed Java sidecar and executes selected
pairs.  Primary manifests and result bundles are always treated as immutable.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from .cli import DEFAULT_SERVICE_URL, DEFAULT_SIDECAR_JAR
from .contract import BENCHMARK_CONTRACT_VERSION, BenchmarkRunSpec
from .hashing import file_sha256
from .io import write_csv_atomic, write_json_atomic
from .manifest import MANIFEST_COLUMNS, PairRecord, read_pair_manifest
from .preflight import validator_for
from .runner import (
    METADATA_FILENAME,
    RESULT_FILENAME,
    prepare_run_context,
    read_result_rows,
    run_benchmark_manifest,
    score_payload_sha256,
    validate_result_bundle,
)
from .sourceafis_adapter import SourceAfisAdapter
from .sourceafis_client import SourceAfisSidecarClient, validate_health
from .sourceafis_sidecar import ManagedSourceAfisSidecar, SidecarStartup


AUDIT_NAME = "sourceafis_reproducibility_audit_v1"
AUDIT_DESCRIPTION = "SourceAFIS reproducibility audit over pre-specified score strata"
PLAN_SCHEMA_VERSION = "sourceafis-reproducibility-audit-plan-v1"
REPORT_SCHEMA_VERSION = "sourceafis-reproducibility-audit-report-v1"
METHOD = "sourceafis"
DATASETS = ("sd300b", "sd300c")
PROTOCOLS = ("plain_self", "roll_self", "plain_roll")
DEFAULT_AUDIT_ROOT = Path("results") / "reproducibility_audits" / AUDIT_NAME
DEFAULT_SEED = AUDIT_NAME
DEFAULT_LOW_POSITIVE_COUNT = 100
DEFAULT_POSITIVE_SAMPLE_COUNT = 100
SELECTION_COLUMNS = [
    "protocol",
    "subject_id",
    "canonical_finger_position",
    "strata",
    "sd300b_pair_id",
    "sd300c_pair_id",
    "sd300b_primary_raw_score",
    "sd300c_primary_raw_score",
]
PAIR_COMPARISON_COLUMNS = [
    "dataset",
    "protocol",
    "pair_id",
    "subject_id",
    "canonical_finger_position",
    "strata",
    "primary_status",
    "rerun_status",
    "status_equal",
    "primary_error_code",
    "rerun_error_code",
    "error_code_equal",
    "primary_raw_score",
    "rerun_raw_score",
    "raw_score_text_equal",
    "raw_score_abs_delta",
    "raw_score_within_tolerance",
    "prepare_a_diagnostics_equal",
    "prepare_b_diagnostics_equal",
    "compare_diagnostics_equal",
    "primary_method_compare_ms",
    "rerun_method_compare_ms",
    "primary_compare_ms",
    "rerun_compare_ms",
    "primary_total_ms",
    "rerun_total_ms",
    "reproducible",
]
CONDITION_SUMMARY_COLUMNS = [
    "dataset",
    "protocol",
    "selected_pair_count",
    "reproducible_pair_count",
    "nonreproducible_pair_count",
    "status_mismatch_count",
    "error_code_mismatch_count",
    "raw_score_text_equal_count",
    "raw_score_within_tolerance_count",
    "diagnostics_mismatch_count",
    "primary_selected_score_payload_sha256",
    "rerun_score_payload_sha256",
    "score_payload_sha256_equal",
    "score_payload_sha256_required_for_pass",
    "config_hash_equal",
    "implementation_hash_equal",
    "implementation_components_equal_except_sidecar_jar_sha256",
    "primary_sidecar_jar_sha256",
    "rerun_sidecar_jar_sha256",
    "implementation_policy",
    "implementation_accepted",
    "primary_mean_method_compare_ms",
    "rerun_mean_method_compare_ms",
    "timings_used_for_pass_fail",
    "passed",
]


class ReproducibilityAuditError(ValueError):
    """Raised when audit preparation, execution, or comparison is unsafe."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and compare an isolated SourceAFIS reproducibility audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Pre-specify deterministic paired strata without invoking SourceAFIS.",
    )
    prepare.add_argument("--primary-results-root", type=Path, default=Path("results"))
    prepare.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    prepare.add_argument("--seed", default=DEFAULT_SEED)
    prepare.add_argument(
        "--low-positive-count",
        type=_nonnegative_int,
        default=DEFAULT_LOW_POSITIVE_COUNT,
    )
    prepare.add_argument(
        "--positive-sample-count",
        type=_nonnegative_int,
        default=DEFAULT_POSITIVE_SAMPLE_COUNT,
    )

    run = subparsers.add_parser(
        "run",
        help="Explicitly run SourceAFIS on the frozen audit manifests.",
    )
    run.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    run.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    run.add_argument("--sidecar-jar", type=Path, default=DEFAULT_SIDECAR_JAR)
    run.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    run.add_argument("--timeout-seconds", type=float, default=120.0)
    run.add_argument(
        "--skip-existing",
        action="store_true",
        help="Validate and reuse already completed isolated audit bundles.",
    )
    run.add_argument(
        "--allow-jar-hash-variation",
        action="store_true",
        help=(
            "Allow execution when the only implementation-provenance difference "
            "is sidecar_jar_sha256; every other deterministic component must match."
        ),
    )

    compare = subparsers.add_parser(
        "compare",
        help="Compare completed reruns to primary selected rows without invoking SourceAFIS.",
    )
    compare.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    compare.add_argument("--score-abs-tolerance", type=_nonnegative_float, default=0.0)
    compare.add_argument(
        "--allow-jar-hash-variation",
        action="store_true",
        help=(
            "Accept a rerun implementation when every deterministic component except "
            "sidecar_jar_sha256 matches the primary run. The mismatch remains reported."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_audit(
                primary_results_root=args.primary_results_root,
                audit_root=args.audit_root,
                seed=args.seed,
                low_positive_count=args.low_positive_count,
                positive_sample_count=args.positive_sample_count,
            )
            print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            result = run_audit(
                audit_root=args.audit_root,
                data_root=args.data_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
                skip_existing=args.skip_existing,
                allow_jar_hash_variation=args.allow_jar_hash_variation,
            )
            print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        if args.command == "compare":
            result = compare_audit(
                audit_root=args.audit_root,
                score_abs_tolerance=args.score_abs_tolerance,
                allow_jar_hash_variation=args.allow_jar_hash_variation,
            )
            print(json.dumps(_comparison_brief(result), ensure_ascii=True, indent=2, sort_keys=True))
            return 0 if result["passed"] else 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Error: unsupported command {args.command!r}", file=sys.stderr)
    return 1


def prepare_audit(
    *,
    primary_results_root: Path,
    audit_root: Path,
    seed: str,
    low_positive_count: int,
    positive_sample_count: int,
) -> dict[str, Any]:
    """Freeze deterministic paired selections and subset manifests."""

    if not str(seed):
        raise ReproducibilityAuditError("Selection seed must be non-empty.")
    primary_results_root = primary_results_root.resolve()
    audit_root = audit_root.resolve()
    conditions: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, protocol in _all_conditions():
        conditions[(dataset, protocol)] = _load_primary_condition(
            primary_results_root,
            dataset,
            protocol,
        )

    selection_rows: list[dict[str, str]] = []
    selected_by_protocol: dict[str, dict[str, set[str]]] = {}
    for protocol in PROTOCOLS:
        b_rows = conditions[("sd300b", protocol)]["rows"]
        c_rows = conditions[("sd300c", protocol)]["rows"]
        strata = select_protocol_strata(
            b_rows,
            c_rows,
            protocol=protocol,
            seed=seed,
            low_positive_count=low_positive_count,
            positive_sample_count=positive_sample_count,
        )
        selected_by_protocol[protocol] = strata
        b_map = {_identity_from_row(row): row for row in b_rows}
        c_map = {_identity_from_row(row): row for row in c_rows}
        for identity in sorted(strata, key=_identity_sort_key):
            subject_id, finger_position = _split_identity(identity)
            b_row = b_map[identity]
            c_row = c_map[identity]
            selection_rows.append(
                {
                    "protocol": protocol,
                    "subject_id": subject_id,
                    "canonical_finger_position": str(finger_position),
                    "strata": ";".join(sorted(strata[identity])),
                    "sd300b_pair_id": b_row["pair_id"],
                    "sd300c_pair_id": c_row["pair_id"],
                    "sd300b_primary_raw_score": b_row["raw_score"],
                    "sd300c_primary_raw_score": c_row["raw_score"],
                }
            )

    plan_dir = audit_root / "plan"
    manifest_dir = audit_root / "manifests"
    selection_path = plan_dir / "selection.csv"
    _publish_immutable_bytes(selection_path, _csv_bytes(selection_rows, SELECTION_COLUMNS))

    condition_plans: list[dict[str, Any]] = []
    for dataset, protocol in _all_conditions():
        primary = conditions[(dataset, protocol)]
        selected = set(selected_by_protocol[protocol])
        source_manifest_path = Path(primary["metadata"]["manifest"]["path"])
        source_manifest_rows = _read_csv_rows(source_manifest_path, MANIFEST_COLUMNS)
        audit_manifest_rows = [
            row for row in source_manifest_rows if _identity_from_row(row) in selected
        ]
        if len(audit_manifest_rows) != len(selected):
            raise ReproducibilityAuditError(
                f"Selected identity set is not complete in {dataset}/{protocol} source manifest."
            )
        audit_manifest_path = manifest_dir / dataset / f"{protocol}.csv"
        _publish_immutable_bytes(
            audit_manifest_path,
            _csv_bytes(audit_manifest_rows, MANIFEST_COLUMNS),
        )
        expected_bundle = _audit_bundle_path(
            audit_root,
            dataset,
            protocol,
            primary["metadata"]["config_hash"],
        )
        condition_plans.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "selected_pair_count": len(audit_manifest_rows),
                "stratum_counts": _stratum_counts(selected_by_protocol[protocol]),
                "audit_manifest_path": str(audit_manifest_path),
                "audit_manifest_sha256": file_sha256(audit_manifest_path),
                "audit_bundle_path": str(expected_bundle),
                "source_manifest_path": str(source_manifest_path.resolve()),
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "primary_bundle_path": str(primary["bundle"]),
                "primary_result_path": str((primary["bundle"] / RESULT_FILENAME).resolve()),
                "primary_result_sha256": primary["metadata"]["result"]["sha256"],
                "primary_run_metadata_path": str(
                    (primary["bundle"] / METADATA_FILENAME).resolve()
                ),
                "primary_run_metadata_sha256": file_sha256(
                    primary["bundle"] / METADATA_FILENAME
                ),
                "primary_config_hash": primary["metadata"]["config_hash"],
                "primary_implementation_hash": primary["metadata"]["implementation_hash"],
                "method": primary["metadata"]["method"],
                "method_version": primary["metadata"]["method_version"],
            }
        )

    plan = {
        "audit_name": AUDIT_NAME,
        "audit_description": AUDIT_DESCRIPTION,
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "selection_unit": ["subject_id", "canonical_finger_position"],
        "sd300_dependency_note": (
            "SD300b and SD300c are paired resolution conditions over shared identities, "
            "not independent populations."
        ),
        "selection_parameters": {
            "seed": seed,
            "low_positive_count_per_protocol": low_positive_count,
            "deterministic_positive_sample_count_per_protocol": positive_sample_count,
        },
        "selection_rules": [
            "For each protocol, SD300b and SD300c primary identity sets must be identical.",
            "Select every identity with raw_score equal to zero in either resolution and classify zero overlap/discordance explicitly.",
            "From identities positive in both resolutions, select the requested lowest-positive identities by min(score_b, score_c), then max(score_b, score_c), then identity.",
            "From the remaining identities positive in both resolutions, select a deterministic SHA-256-ranked sample using seed, protocol, subject_id, and canonical_finger_position.",
            "Use the same selected anatomical identities in both resolutions and retain source manifest rows without changing pair semantics.",
        ],
        "pass_fail_policy": {
            "raw_score": "Exact numeric equality by default; compare permits an explicit absolute tolerance.",
            "status": "Must be equal.",
            "error_code": "Must be equal.",
            "diagnostics": "All three operation diagnostics fields must be equal.",
            "config_hash": "Must equal the primary run.",
            "implementation_hash": "Must equal the primary run.",
            "timings": "Reported but never used for pass/fail.",
        },
        "primary_results_root": str(primary_results_root),
        "audit_root": str(audit_root),
        "selection_csv_path": str(selection_path),
        "selection_csv_sha256": file_sha256(selection_path),
        "conditions": condition_plans,
        "sourceafis_execution_performed_by_prepare": False,
        "timestamps_in_plan": False,
    }
    plan_path = plan_dir / "audit_plan.json"
    _publish_immutable_bytes(plan_path, _json_bytes(plan))
    return {
        "audit_name": AUDIT_NAME,
        "audit_plan": str(plan_path),
        "selection_csv": str(selection_path),
        "condition_count": len(condition_plans),
        "total_planned_measured_pairs": sum(
            condition["selected_pair_count"] for condition in condition_plans
        ),
        "sourceafis_execution_performed": False,
    }


def select_protocol_strata(
    sd300b_rows: list[dict[str, str]],
    sd300c_rows: list[dict[str, str]],
    *,
    protocol: str,
    seed: str,
    low_positive_count: int,
    positive_sample_count: int,
) -> dict[str, set[str]]:
    """Return deterministic, paired identity-to-strata selections."""

    b_map = {_identity_from_row(row): row for row in sd300b_rows}
    c_map = {_identity_from_row(row): row for row in sd300c_rows}
    if len(b_map) != len(sd300b_rows) or len(c_map) != len(sd300c_rows):
        raise ReproducibilityAuditError(f"Duplicate identity in primary {protocol} results.")
    if set(b_map) != set(c_map):
        raise ReproducibilityAuditError(
            f"SD300b/SD300c identity sets differ for protocol {protocol}."
        )

    selected: dict[str, set[str]] = {}
    positive_both: list[tuple[str, float, float]] = []
    prefix = "plain_roll" if protocol == "plain_roll" else "self"
    for identity in sorted(b_map, key=_identity_sort_key):
        b_score = _finite_score(b_map[identity], f"sd300b/{protocol}")
        c_score = _finite_score(c_map[identity], f"sd300c/{protocol}")
        if b_score < 0 or c_score < 0:
            raise ReproducibilityAuditError(
                f"Negative SourceAFIS score for {identity} in protocol {protocol}."
            )
        labels: set[str] = set()
        if b_score == 0 and c_score == 0:
            labels.add(f"{prefix}_zero_both_resolutions")
        elif b_score == 0:
            labels.add(f"{prefix}_zero_sd300b_only")
        elif c_score == 0:
            labels.add(f"{prefix}_zero_sd300c_only")
        else:
            positive_both.append((identity, b_score, c_score))
        if labels:
            selected[identity] = labels

    low_ranked = sorted(
        positive_both,
        key=lambda item: (min(item[1], item[2]), max(item[1], item[2]), _identity_sort_key(item[0])),
    )
    for identity, _, _ in low_ranked[:low_positive_count]:
        selected.setdefault(identity, set()).add("low_positive_both_resolutions")

    low_identities = {identity for identity, _, _ in low_ranked[:low_positive_count]}
    random_pool = [item for item in positive_both if item[0] not in low_identities]
    random_ranked = sorted(
        random_pool,
        key=lambda item: (_sample_hash(seed, protocol, item[0]), _identity_sort_key(item[0])),
    )
    for identity, _, _ in random_ranked[:positive_sample_count]:
        selected.setdefault(identity, set()).add("deterministic_positive_sample")
    return selected


def run_audit(
    *,
    audit_root: Path,
    data_root: Path,
    sidecar_jar: Path,
    service_url: str,
    timeout_seconds: float,
    skip_existing: bool,
    allow_jar_hash_variation: bool,
) -> dict[str, Any]:
    """Run the six frozen audit manifests in isolated benchmark-v2 bundles."""

    audit_root = audit_root.resolve()
    plan = _load_and_verify_plan(audit_root)
    run_summaries: list[dict[str, Any]] = []
    for condition in plan["conditions"]:
        dataset = condition["dataset"]
        protocol = condition["protocol"]
        manifest_path = Path(condition["audit_manifest_path"])
        expected_bundle = Path(condition["audit_bundle_path"])
        subset_validator = make_subset_validator(
            source_manifest_path=Path(condition["source_manifest_path"]),
            expected_dataset=dataset,
            expected_protocol=protocol,
            expected_subset_sha256=condition["audit_manifest_sha256"],
        )
        with ManagedSourceAfisSidecar(
            sidecar_jar,
            service_url,
            timeout_seconds=timeout_seconds,
        ) as sidecar:
            startup = _startup_dict(sidecar.startup)
            client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
            try:
                health = client.health()
                validate_health(health)
                startup_validation = {
                    **startup,
                    "health": health.raw,
                    "health_requests_before_pair_execution": client.health_request_count,
                    "audit_name": AUDIT_NAME,
                    "audit_plan_path": str((audit_root / "plan" / "audit_plan.json").resolve()),
                }
                adapter = SourceAfisAdapter(client, health=health)
                context = prepare_run_context(
                    manifest_path=manifest_path,
                    expected_dataset=dataset,
                    expected_protocol=protocol,
                    adapter=adapter,
                    results_root=audit_root / "runs",
                    startup_validation=startup_validation,
                    bundle_directory=expected_bundle,
                )
                if context.spec.config_hash != condition["primary_config_hash"]:
                    raise ReproducibilityAuditError(
                        f"Current config hash differs from primary for {dataset}/{protocol}; "
                        "audit pair execution was not started."
                    )
                primary_metadata = json.loads(
                    Path(condition["primary_run_metadata_path"]).read_text(encoding="utf-8")
                )
                implementation = implementation_compatibility(
                    primary_hash=condition["primary_implementation_hash"],
                    current_hash=context.spec.implementation_hash,
                    primary_components=primary_metadata["implementation_hash_components"],
                    current_components=context.implementation_hash_components,
                    allow_jar_hash_variation=allow_jar_hash_variation,
                )
                if not implementation["accepted"]:
                    raise ReproducibilityAuditError(
                        f"Current implementation is not accepted for {dataset}/{protocol}; "
                        "audit pair execution was not started. "
                        f"primary_hash={implementation['primary_hash']}, "
                        f"current_hash={implementation['current_hash']}, "
                        f"equal_except_sidecar_jar_sha256="
                        f"{implementation['components_equal_except_sidecar_jar_sha256']}, "
                        f"primary_jar={implementation['primary_sidecar_jar_sha256']}, "
                        f"current_jar={implementation['current_sidecar_jar_sha256']}."
                    )
                if not implementation["exact_hash_equal"]:
                    print(
                        f"[{dataset}/{protocol}] documented sidecar JAR SHA-256 variation accepted; "
                        "all other deterministic implementation components are identical.",
                        file=sys.stderr,
                        flush=True,
                    )
                metadata = run_benchmark_manifest(
                    manifest_path=manifest_path,
                    adapter=adapter,
                    expected_dataset=dataset,
                    expected_protocol=protocol,
                    results_root=audit_root / "runs",
                    startup_validation=startup_validation,
                    data_root=data_root,
                    dedicated_validator=subset_validator,
                    skip_existing=skip_existing,
                    bundle_directory=expected_bundle,
                    progress_callback=lambda completed, total, d=dataset, p=protocol: print(
                        f"[{d}/{p}] {completed}/{total} measured audit pairs",
                        file=sys.stderr,
                        flush=True,
                    ),
                )
            finally:
                client.close()
        run_summaries.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "row_count": metadata["result"]["row_count"],
                "bundle_path": str(expected_bundle),
                "result_sha256": metadata["result"]["sha256"],
                "score_payload_sha256": metadata["result"]["score_payload_sha256"],
                "implementation_compatibility": implementation,
            }
        )
    return {
        "audit_name": AUDIT_NAME,
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "primary_artifacts_overwritten": False,
    }


def make_subset_validator(
    *,
    source_manifest_path: Path,
    expected_dataset: str,
    expected_protocol: str,
    expected_subset_sha256: str,
    source_validator: Callable[[Path, Path], Any] | None = None,
) -> Callable[[Path, Path], dict[str, Any]]:
    """Validate an audit manifest as an exact subset of a fully validated source manifest."""

    def validate_subset(subset_manifest_path: Path, data_root: Path) -> dict[str, Any]:
        if file_sha256(subset_manifest_path) != expected_subset_sha256:
            raise ReproducibilityAuditError("Audit subset manifest hash changed after planning.")
        validator = source_validator or validator_for(expected_dataset, expected_protocol)
        source_report = validator(source_manifest_path, data_root)
        source_pairs = {pair.pair_id: pair for pair in read_pair_manifest(source_manifest_path)}
        subset_pairs = read_pair_manifest(subset_manifest_path)
        if not subset_pairs:
            raise ReproducibilityAuditError("Audit subset manifest is empty.")
        for pair in subset_pairs:
            source_pair = source_pairs.get(pair.pair_id)
            if source_pair is None or source_pair != pair:
                raise ReproducibilityAuditError(
                    f"Audit pair {pair.pair_id!r} is not an exact source-manifest row."
                )
            if pair.dataset != expected_dataset or pair.protocol != expected_protocol:
                raise ReproducibilityAuditError(
                    f"Audit pair {pair.pair_id!r} has wrong dataset/protocol."
                )
        return {
            "validation_mode": "exact_subset_of_fully_validated_source_manifest",
            "source_manifest_path": str(source_manifest_path.resolve()),
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "source_manifest_validator_result": _report_dict(source_report),
            "subset_manifest_sha256": expected_subset_sha256,
            "subset_pair_count": len(subset_pairs),
        }

    return validate_subset


def compare_audit(
    *,
    audit_root: Path,
    score_abs_tolerance: float,
    allow_jar_hash_variation: bool,
) -> dict[str, Any]:
    """Compare selected primary rows to completed audit reruns."""

    audit_root = audit_root.resolve()
    plan = _load_and_verify_plan(audit_root)
    missing_bundles = [
        f"{condition['dataset']}/{condition['protocol']}"
        for condition in plan["conditions"]
        if not (Path(condition["audit_bundle_path"]) / METADATA_FILENAME).is_file()
        or not (Path(condition["audit_bundle_path"]) / RESULT_FILENAME).is_file()
    ]
    if missing_bundles:
        raise ReproducibilityAuditError(
            "Audit execution is incomplete; comparison was not started. Missing validated "
            f"rerun bundles: {', '.join(missing_bundles)}. Run the audit with "
            "--skip-existing, then compare again."
        )
    selection_lookup = _selection_lookup(Path(plan["selection_csv_path"]))
    pair_comparisons: list[dict[str, str]] = []
    condition_summaries: list[dict[str, Any]] = []

    for condition in plan["conditions"]:
        dataset = condition["dataset"]
        protocol = condition["protocol"]
        primary_rows = read_result_rows(Path(condition["primary_result_path"]))
        primary_by_pair = {row["pair_id"]: row for row in primary_rows}
        manifest_pairs = read_pair_manifest(Path(condition["audit_manifest_path"]))
        rerun_bundle = Path(condition["audit_bundle_path"])
        rerun_metadata = json.loads(
            (rerun_bundle / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        rerun_spec_payload = dict(rerun_metadata["run_spec"])
        rerun_spec_payload["manifest_path"] = Path(rerun_spec_payload["manifest_path"])
        rerun_spec = BenchmarkRunSpec(**rerun_spec_payload)
        validate_result_bundle(
            rerun_bundle,
            manifest_records=manifest_pairs,
            run_spec=rerun_spec,
            score_direction=rerun_metadata["score_direction"],
            score_semantics=rerun_metadata["score_semantics"],
        )
        rerun_rows = read_result_rows(rerun_bundle / RESULT_FILENAME)
        if [row["pair_id"] for row in rerun_rows] != [pair.pair_id for pair in manifest_pairs]:
            raise ReproducibilityAuditError(
                f"Rerun pair sequence differs from audit manifest for {dataset}/{protocol}."
            )
        selected_primary = []
        condition_pairs = []
        for rerun_row in rerun_rows:
            primary_row = primary_by_pair.get(rerun_row["pair_id"])
            if primary_row is None:
                raise ReproducibilityAuditError(
                    f"Rerun pair {rerun_row['pair_id']!r} is absent from the primary result."
                )
            selected_primary.append(primary_row)
            identity = _identity_from_row(rerun_row)
            strata = selection_lookup[(protocol, identity)]
            comparison = compare_pair_rows(
                primary_row,
                rerun_row,
                strata=strata,
                score_abs_tolerance=score_abs_tolerance,
            )
            pair_comparisons.append(comparison)
            condition_pairs.append(comparison)

        config_hash_equal = rerun_metadata["config_hash"] == condition["primary_config_hash"]
        primary_metadata = json.loads(
            Path(condition["primary_run_metadata_path"]).read_text(encoding="utf-8")
        )
        implementation = implementation_compatibility(
            primary_hash=condition["primary_implementation_hash"],
            current_hash=rerun_metadata["implementation_hash"],
            primary_components=primary_metadata["implementation_hash_components"],
            current_components=rerun_metadata["implementation_hash_components"],
            allow_jar_hash_variation=allow_jar_hash_variation,
        )
        primary_payload_hash = score_payload_sha256(selected_primary)
        rerun_payload_hash = score_payload_sha256(rerun_rows)
        score_payload_equal = primary_payload_hash == rerun_payload_hash
        score_payload_required = score_abs_tolerance == 0
        diagnostics_mismatch_count = sum(
            not (
                _as_bool(row["prepare_a_diagnostics_equal"])
                and _as_bool(row["prepare_b_diagnostics_equal"])
                and _as_bool(row["compare_diagnostics_equal"])
            )
            for row in condition_pairs
        )
        passed = (
            all(_as_bool(row["reproducible"]) for row in condition_pairs)
            and config_hash_equal
            and implementation["accepted"]
            and (score_payload_equal or not score_payload_required)
        )
        condition_summaries.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "selected_pair_count": len(condition_pairs),
                "reproducible_pair_count": sum(
                    _as_bool(row["reproducible"]) for row in condition_pairs
                ),
                "nonreproducible_pair_count": sum(
                    not _as_bool(row["reproducible"]) for row in condition_pairs
                ),
                "status_mismatch_count": sum(
                    not _as_bool(row["status_equal"]) for row in condition_pairs
                ),
                "error_code_mismatch_count": sum(
                    not _as_bool(row["error_code_equal"]) for row in condition_pairs
                ),
                "raw_score_text_equal_count": sum(
                    _as_bool(row["raw_score_text_equal"]) for row in condition_pairs
                ),
                "raw_score_within_tolerance_count": sum(
                    _as_bool(row["raw_score_within_tolerance"]) for row in condition_pairs
                ),
                "diagnostics_mismatch_count": diagnostics_mismatch_count,
                "primary_selected_score_payload_sha256": primary_payload_hash,
                "rerun_score_payload_sha256": rerun_payload_hash,
                "score_payload_sha256_equal": score_payload_equal,
                "score_payload_sha256_required_for_pass": score_payload_required,
                "config_hash_equal": config_hash_equal,
                "implementation_hash_equal": implementation["exact_hash_equal"],
                "implementation_components_equal_except_sidecar_jar_sha256": (
                    implementation["components_equal_except_sidecar_jar_sha256"]
                ),
                "primary_sidecar_jar_sha256": implementation["primary_sidecar_jar_sha256"],
                "rerun_sidecar_jar_sha256": implementation["current_sidecar_jar_sha256"],
                "implementation_policy": implementation["policy"],
                "implementation_accepted": implementation["accepted"],
                "primary_mean_method_compare_ms": _mean_timing(
                    row["primary_method_compare_ms"] for row in condition_pairs
                ),
                "rerun_mean_method_compare_ms": _mean_timing(
                    row["rerun_method_compare_ms"] for row in condition_pairs
                ),
                "timings_used_for_pass_fail": False,
                "passed": passed,
            }
        )

    comparison_dir = audit_root / "comparison"
    pair_path = comparison_dir / "pair_comparison.csv"
    summary_path = comparison_dir / "condition_summary.csv"
    write_csv_atomic(pair_comparisons, pair_path, PAIR_COMPARISON_COLUMNS)
    write_csv_atomic(
        [_stringify_row(row) for row in condition_summaries],
        summary_path,
        CONDITION_SUMMARY_COLUMNS,
    )
    report = {
        "audit_name": AUDIT_NAME,
        "audit_description": AUDIT_DESCRIPTION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "score_abs_tolerance": score_abs_tolerance,
        "implementation_policy": (
            "allow_only_sidecar_jar_sha256_variation"
            if allow_jar_hash_variation
            else "require_exact_implementation_hash"
        ),
        "implementation_policy_note": (
            "A JAR hash variation is accepted only when every other deterministic "
            "implementation-hash component is exactly equal; both JAR hashes remain reported."
        ),
        "timings_used_for_pass_fail": False,
        "condition_summaries": condition_summaries,
        "overall": {
            "condition_count": len(condition_summaries),
            "selected_pair_count": len(pair_comparisons),
            "reproducible_pair_count": sum(
                _as_bool(row["reproducible"]) for row in pair_comparisons
            ),
            "nonreproducible_pair_count": sum(
                not _as_bool(row["reproducible"]) for row in pair_comparisons
            ),
            "exact_implementation_hash_condition_count": sum(
                summary["implementation_hash_equal"] for summary in condition_summaries
            ),
            "accepted_jar_hash_variation_condition_count": sum(
                summary["implementation_accepted"]
                and not summary["implementation_hash_equal"]
                for summary in condition_summaries
            ),
        },
        "artifacts": {
            "pair_comparison_csv": str(pair_path),
            "pair_comparison_sha256": file_sha256(pair_path),
            "condition_summary_csv": str(summary_path),
            "condition_summary_sha256": file_sha256(summary_path),
            "audit_plan_path": str((audit_root / "plan" / "audit_plan.json").resolve()),
            "audit_plan_sha256": file_sha256(audit_root / "plan" / "audit_plan.json"),
        },
        "passed": all(summary["passed"] for summary in condition_summaries),
    }
    report_path = comparison_dir / "audit_report.json"
    write_json_atomic(report, report_path)
    return {**report, "report_path": str(report_path)}


def compare_pair_rows(
    primary: dict[str, str],
    rerun: dict[str, str],
    *,
    strata: str,
    score_abs_tolerance: float,
) -> dict[str, str]:
    """Compare one pair while explicitly excluding timing equality from pass/fail."""

    status_equal = primary["status"] == rerun["status"]
    error_code_equal = primary["error_code"] == rerun["error_code"]
    score_text_equal = primary["raw_score"] == rerun["raw_score"]
    if primary["raw_score"] == "" or rerun["raw_score"] == "":
        score_abs_delta = ""
        score_within_tolerance = primary["raw_score"] == rerun["raw_score"]
    else:
        delta = abs(float(primary["raw_score"]) - float(rerun["raw_score"]))
        score_abs_delta = repr(delta)
        score_within_tolerance = math.isfinite(delta) and delta <= score_abs_tolerance
    prepare_a_diagnostics_equal = (
        primary["prepare_a_diagnostics"] == rerun["prepare_a_diagnostics"]
    )
    prepare_b_diagnostics_equal = (
        primary["prepare_b_diagnostics"] == rerun["prepare_b_diagnostics"]
    )
    compare_diagnostics_equal = (
        primary["compare_diagnostics"] == rerun["compare_diagnostics"]
    )
    reproducible = (
        status_equal
        and error_code_equal
        and score_within_tolerance
        and prepare_a_diagnostics_equal
        and prepare_b_diagnostics_equal
        and compare_diagnostics_equal
    )
    return {
        "dataset": rerun["dataset"],
        "protocol": rerun["protocol"],
        "pair_id": rerun["pair_id"],
        "subject_id": rerun["subject_id"],
        "canonical_finger_position": rerun["canonical_finger_position"],
        "strata": strata,
        "primary_status": primary["status"],
        "rerun_status": rerun["status"],
        "status_equal": _bool_text(status_equal),
        "primary_error_code": primary["error_code"],
        "rerun_error_code": rerun["error_code"],
        "error_code_equal": _bool_text(error_code_equal),
        "primary_raw_score": primary["raw_score"],
        "rerun_raw_score": rerun["raw_score"],
        "raw_score_text_equal": _bool_text(score_text_equal),
        "raw_score_abs_delta": score_abs_delta,
        "raw_score_within_tolerance": _bool_text(score_within_tolerance),
        "prepare_a_diagnostics_equal": _bool_text(prepare_a_diagnostics_equal),
        "prepare_b_diagnostics_equal": _bool_text(prepare_b_diagnostics_equal),
        "compare_diagnostics_equal": _bool_text(compare_diagnostics_equal),
        "primary_method_compare_ms": primary["method_compare_ms"],
        "rerun_method_compare_ms": rerun["method_compare_ms"],
        "primary_compare_ms": primary["compare_ms"],
        "rerun_compare_ms": rerun["compare_ms"],
        "primary_total_ms": primary["total_ms"],
        "rerun_total_ms": rerun["total_ms"],
        "reproducible": _bool_text(reproducible),
    }


def implementation_compatibility(
    *,
    primary_hash: str,
    current_hash: str,
    primary_components: dict[str, Any],
    current_components: dict[str, Any],
    allow_jar_hash_variation: bool,
) -> dict[str, Any]:
    """Accept only an explicitly allowed sidecar-JAR hash provenance variation."""

    primary_copy = json.loads(json.dumps(primary_components, sort_keys=True))
    current_copy = json.loads(json.dumps(current_components, sort_keys=True))
    primary_jar = primary_copy.pop("sidecar_jar_sha256", None)
    current_jar = current_copy.pop("sidecar_jar_sha256", None)
    components_equal_except_jar = primary_copy == current_copy
    exact_components_equal = (
        primary_components == current_components and primary_hash == current_hash
    )
    accepted = exact_components_equal or (
        allow_jar_hash_variation and components_equal_except_jar
    )
    return {
        "primary_hash": primary_hash,
        "current_hash": current_hash,
        "exact_hash_equal": exact_components_equal,
        "components_equal_except_sidecar_jar_sha256": components_equal_except_jar,
        "primary_sidecar_jar_sha256": primary_jar,
        "current_sidecar_jar_sha256": current_jar,
        "policy": (
            "allow_only_sidecar_jar_sha256_variation"
            if allow_jar_hash_variation
            else "require_exact_implementation_hash"
        ),
        "accepted": accepted,
    }


def _load_primary_condition(
    primary_results_root: Path,
    dataset: str,
    protocol: str,
) -> dict[str, Any]:
    contract_dir = (
        primary_results_root / dataset / protocol / METHOD / BENCHMARK_CONTRACT_VERSION
    )
    bundles = sorted(
        metadata.parent
        for metadata in contract_dir.glob(f"*/{METADATA_FILENAME}")
        if (metadata.parent / RESULT_FILENAME).is_file()
    )
    if len(bundles) != 1:
        raise ReproducibilityAuditError(
            f"Expected one primary bundle for {dataset}/{protocol}, found {len(bundles)}."
        )
    bundle = bundles[0].resolve()
    metadata = json.loads((bundle / METADATA_FILENAME).read_text(encoding="utf-8"))
    if metadata.get("dataset") != dataset or metadata.get("protocol") != protocol:
        raise ReproducibilityAuditError(f"Primary metadata identity mismatch in {bundle}.")
    raw_spec = dict(metadata["run_spec"])
    raw_spec["manifest_path"] = Path(raw_spec["manifest_path"])
    spec = BenchmarkRunSpec(**raw_spec)
    manifest_pairs = read_pair_manifest(spec.manifest_path)
    validate_result_bundle(
        bundle,
        manifest_records=manifest_pairs,
        run_spec=spec,
        score_direction=metadata["score_direction"],
        score_semantics=metadata["score_semantics"],
    )
    rows = read_result_rows(bundle / RESULT_FILENAME)
    if [row["pair_id"] for row in rows] != [pair.pair_id for pair in manifest_pairs]:
        raise ReproducibilityAuditError(f"Primary result/manifest order mismatch in {bundle}.")
    return {"bundle": bundle, "metadata": metadata, "rows": rows}


def _load_and_verify_plan(audit_root: Path) -> dict[str, Any]:
    plan_path = audit_root / "plan" / "audit_plan.json"
    if not plan_path.is_file():
        raise ReproducibilityAuditError(f"Audit plan does not exist: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_schema_version") != PLAN_SCHEMA_VERSION:
        raise ReproducibilityAuditError("Audit plan schema version mismatch.")
    if Path(plan["audit_root"]).resolve() != audit_root:
        raise ReproducibilityAuditError("Audit plan root does not match --audit-root.")
    if file_sha256(Path(plan["selection_csv_path"])) != plan["selection_csv_sha256"]:
        raise ReproducibilityAuditError("Frozen selection.csv hash mismatch.")
    for condition in plan["conditions"]:
        checks = (
            ("audit_manifest_path", "audit_manifest_sha256"),
            ("source_manifest_path", "source_manifest_sha256"),
            ("primary_result_path", "primary_result_sha256"),
            ("primary_run_metadata_path", "primary_run_metadata_sha256"),
        )
        for path_key, hash_key in checks:
            file_path = Path(condition[path_key])
            if file_sha256(file_path) != condition[hash_key]:
                raise ReproducibilityAuditError(
                    f"Frozen input hash mismatch for {condition['dataset']}/{condition['protocol']}: "
                    f"{file_path}"
                )
    return plan


def _audit_bundle_path(
    audit_root: Path,
    dataset: str,
    protocol: str,
    config_hash: str,
) -> Path:
    return (
        audit_root
        / "runs"
        / dataset
        / protocol
        / METHOD
        / BENCHMARK_CONTRACT_VERSION
        / config_hash
    ).resolve()


def _selection_lookup(selection_path: Path) -> dict[tuple[str, str], str]:
    rows = _read_csv_rows(selection_path, SELECTION_COLUMNS)
    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["protocol"], _identity_from_row(row))
        if key in lookup:
            raise ReproducibilityAuditError(f"Duplicate selection row: {key}")
        lookup[key] = row["strata"]
    return lookup


def _read_csv_rows(file_path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ReproducibilityAuditError(
                f"CSV schema mismatch in {file_path}: {reader.fieldnames}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ReproducibilityAuditError(f"Extra unnamed CSV values in {file_path}.")
    return rows


def _csv_bytes(rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_immutable_bytes(output_path: Path, payload: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise ReproducibilityAuditError(
                f"Frozen audit artifact already exists with different bytes: {output_path}"
            )
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
        temporary.replace(output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _identity_from_row(row: dict[str, str]) -> str:
    return f"{row['subject_id']}\x1f{int(row['canonical_finger_position'])}"


def _split_identity(identity: str) -> tuple[str, int]:
    subject_id, finger_position = identity.split("\x1f", 1)
    return subject_id, int(finger_position)


def _identity_sort_key(identity: str) -> tuple[str, int]:
    return _split_identity(identity)


def _finite_score(row: dict[str, str], label: str) -> float:
    if row.get("status") != "ok" or row.get("raw_score", "") == "":
        raise ReproducibilityAuditError(
            f"Primary row {row.get('pair_id')!r} is not a successful scored row in {label}."
        )
    value = float(row["raw_score"])
    if not math.isfinite(value):
        raise ReproducibilityAuditError(
            f"Primary row {row['pair_id']!r} has non-finite raw_score in {label}."
        )
    return value


def _sample_hash(seed: str, protocol: str, identity: str) -> str:
    subject_id, finger_position = _split_identity(identity)
    payload = f"{seed}|{protocol}|{subject_id}|{finger_position}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stratum_counts(selection: dict[str, set[str]]) -> dict[str, int]:
    labels = sorted({label for identity_labels in selection.values() for label in identity_labels})
    return {
        label: sum(label in identity_labels for identity_labels in selection.values())
        for label in labels
    }


def _all_conditions() -> Iterable[tuple[str, str]]:
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            yield dataset, protocol


def _report_dict(report: Any) -> dict[str, Any]:
    if is_dataclass(report):
        return asdict(report)
    if isinstance(report, dict):
        return dict(report)
    return {"result": str(report)}


def _startup_dict(startup: SidecarStartup | None) -> dict[str, Any]:
    if startup is None:
        return {}
    return {
        "managed_by_runner": startup.managed_by_runner,
        "service_url": startup.service_url,
        "startup_ms": startup.startup_ms,
        "validation_result": startup.validation_result,
        "command": startup.command,
        "jar_path": startup.jar_path,
        "jar_sha256": startup.jar_sha256,
        "java_executable": startup.java_executable,
    }


def _mean_timing(values: Iterable[str]) -> float | None:
    numeric = [float(value) for value in values if value != ""]
    return statistics.fmean(numeric) if numeric else None


def _stringify_row(row: dict[str, Any]) -> dict[str, str]:
    result = {}
    for key, value in row.items():
        if isinstance(value, bool):
            result[key] = _bool_text(value)
        elif value is None:
            result[key] = ""
        else:
            result[key] = str(value)
    return result


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _as_bool(value: str | bool) -> bool:
    return value is True or value == "true"


def _comparison_brief(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_name": report["audit_name"],
        "passed": report["passed"],
        "selected_pair_count": report["overall"]["selected_pair_count"],
        "reproducible_pair_count": report["overall"]["reproducible_pair_count"],
        "nonreproducible_pair_count": report["overall"]["nonreproducible_pair_count"],
        "report_path": report["report_path"],
    }


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative.")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be finite and non-negative.")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
