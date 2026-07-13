"""Validation, supervisor summaries, audits, and the SIFT-specific cohort."""

from __future__ import annotations

import csv
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fingerprint_benchmark.contract import BENCHMARK_CONTRACT_VERSION, OK
from fingerprint_benchmark.hashing import file_sha256, stable_hash
from fingerprint_benchmark.io import write_csv_atomic, write_json_atomic
from fingerprint_benchmark.manifest import MANIFEST_COLUMNS, read_pair_manifest
from fingerprint_benchmark.runner import METADATA_FILENAME, RESULT_FILENAME, read_result_rows

from .config import METHOD_NAME, METHOD_VERSION
from .integrity import compare_inventories, protected_input_inventory


COHORT_NAME = "sift_geometric_joint_self_accept_v1"
REPORT_SCHEMA = "sift-geometric-supervisor-report-v1"


def build_final_reports(repo_root: Path, results_root: Path) -> dict[str, Any]:
    development = results_root / METHOD_NAME / "development"
    decision_rule = json.loads((development / "decision_rule.json").read_text(encoding="utf-8"))
    bundles = _load_primary_bundles(results_root)
    summaries = []
    evaluation_summaries = []
    audit_rows = []
    split_payload = json.loads((development / "subject_split.json").read_text(encoding="utf-8"))
    evaluation_subjects = set(split_payload["evaluation_subjects"])
    for key, bundle in sorted(bundles.items()):
        dataset, protocol = key
        threshold = float(decision_rule["thresholds_by_dataset"][dataset]["primary_threshold"])
        rows = bundle["rows"]
        summary = summarize_run(dataset, protocol, rows, threshold)
        summaries.append(summary)
        evaluation_summaries.append(
            summarize_run(
                dataset,
                protocol,
                [row for row in rows if row["subject_id"] in evaluation_subjects],
                threshold,
            )
        )
        audit_rows.extend(_audit_rows(dataset, protocol, rows, threshold))

    report_root = results_root / METHOD_NAME / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(summaries, report_root / "full_manifest_supervisor_table.csv", list(summaries[0]))
    write_csv_atomic(
        evaluation_summaries,
        report_root / "evaluation_subjects_supervisor_table.csv",
        list(evaluation_summaries[0]),
    )
    if audit_rows:
        write_csv_atomic(audit_rows, report_root / "decision_diagnostics.csv", list(audit_rows[0]))
    _write_special_audits(audit_rows, report_root)

    cohort = build_cohort(repo_root, results_root, bundles, decision_rule)
    cohort_summaries = []
    included = set(cohort["included_identity_keys"])
    for (dataset, protocol), bundle in sorted(bundles.items()):
        filtered = [
            row
            for row in bundle["rows"]
            if _identity_key(row["subject_id"], row["canonical_finger_position"]) in included
        ]
        threshold = float(decision_rule["thresholds_by_dataset"][dataset]["primary_threshold"])
        cohort_summaries.append(summarize_run(dataset, protocol, filtered, threshold))
    write_csv_atomic(
        cohort_summaries,
        report_root / "cohort_supervisor_table.csv",
        list(cohort_summaries[0]),
    )
    markdown_path = report_root / "supervisor_tables.md"
    markdown_path.write_text(
        _render_markdown(summaries, evaluation_summaries, cohort_summaries, cohort, decision_rule),
        encoding="utf-8",
    )

    before = json.loads((development / "protected_inputs_before.json").read_text(encoding="utf-8"))
    after = protected_input_inventory(repo_root)
    write_json_atomic(after, development / "protected_inputs_after.json")
    integrity = compare_inventories(before, after)
    write_json_atomic(integrity, report_root / "protected_input_integrity.json")
    if integrity["status"] != "ok":
        raise RuntimeError(f"Protected inputs changed: {integrity}")

    inventory = _artifact_inventory(results_root / METHOD_NAME)
    write_json_atomic(inventory, report_root / "artifact_hashes.json")
    result = {
        "report_schema": REPORT_SCHEMA,
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "full_run_count": len(summaries),
        "cohort_name": COHORT_NAME,
        "cohort_identity_count": cohort["included_identity_count"],
        "full_manifest_table": str(report_root / "full_manifest_supervisor_table.csv"),
        "cohort_table": str(report_root / "cohort_supervisor_table.csv"),
        "evaluation_subjects_table": str(report_root / "evaluation_subjects_supervisor_table.csv"),
        "supervisor_markdown": str(markdown_path),
        "protected_input_integrity": integrity,
        "artifact_inventory_sha256": inventory["inventory_sha256"],
    }
    write_json_atomic(result, report_root / "report_summary.json")
    return result


def summarize_run(
    dataset: str,
    protocol: str,
    rows: list[dict[str, str]],
    threshold: float,
) -> dict[str, Any]:
    diagnostics = [_diagnostics(row) for row in rows]
    successes = [row for row in rows if row["status"] == OK]
    scores = [float(row["raw_score"]) for row in successes]
    accepted = [row for row in successes if float(row["raw_score"]) >= threshold]
    identities = {_identity_key(row["subject_id"], row["canonical_finger_position"]) for row in rows}
    fingers = Counter(row["canonical_finger_position"] for row in rows)
    geometry_failures = [diag for diag in diagnostics if diag and not bool(diag.get("geometry_success"))]
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "dataset": dataset,
        "protocol": protocol,
        "pairs": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "anatomical_identities": len(identities),
        "finger_type_distribution_json": json.dumps(fingers, sort_keys=True),
        "success_count": len(successes),
        "failure_count": len(rows) - len(successes),
        "decision_threshold": threshold,
        "accepted_count": len(accepted),
        "rejected_count": len(rows) - len(accepted),
        "accepted_percentage": _percentage(len(accepted), len(rows)),
        "raw_score_mean": _mean(scores),
        "raw_score_median": _median(scores),
        "zero_score_percentage": _percentage(sum(score == 0.0 for score in scores), len(successes)),
        "geometry_failure_percentage": _percentage(len(geometry_failures), len(successes)),
        **_timing_fields(rows, "prepare_a_ms", "prepare_a"),
        **_timing_fields(rows, "prepare_b_ms", "prepare_b"),
        **_timing_fields(rows, "compare_ms", "compare"),
        **_timing_fields(rows, "total_ms", "total"),
        **_diagnostic_summary(diagnostics, "keypoint_count_a", "keypoints_a"),
        **_diagnostic_summary(diagnostics, "keypoint_count_b", "keypoints_b"),
        **_diagnostic_summary(diagnostics, "ratio_match_count_a_to_b", "ratio_matches"),
        **_diagnostic_summary(diagnostics, "mutual_match_count", "mutual_matches"),
        **_diagnostic_summary(diagnostics, "geometric_inlier_count", "inliers"),
        **_residual_summary(diagnostics),
    }


def build_cohort(
    repo_root: Path,
    results_root: Path,
    bundles: dict[tuple[str, str], dict[str, Any]],
    decision_rule: dict[str, Any],
) -> dict[str, Any]:
    condition_accepts: dict[tuple[str, str], set[str]] = {}
    condition_status: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in ("sd300b", "sd300c"):
        threshold = float(decision_rule["thresholds_by_dataset"][dataset]["primary_threshold"])
        for protocol in ("plain_self", "roll_self"):
            rows = bundles[(dataset, protocol)]["rows"]
            accepted = {
                _identity_key(row["subject_id"], row["canonical_finger_position"])
                for row in rows
                if row["status"] == OK and float(row["raw_score"]) >= threshold
            }
            condition_accepts[(dataset, protocol)] = accepted
            condition_status[(dataset, protocol)] = {
                _identity_key(row["subject_id"], row["canonical_finger_position"]): (
                    "accepted"
                    if row["status"] == OK and float(row["raw_score"]) >= threshold
                    else (row["status"] if row["status"] != OK else "below_threshold")
                )
                for row in rows
            }
    plain_roll_presence = {
        dataset: {
            _identity_key(row["subject_id"], row["canonical_finger_position"])
            for row in bundles[(dataset, "plain_roll")]["rows"]
        }
        for dataset in ("sd300b", "sd300c")
    }
    universe = set.intersection(*plain_roll_presence.values())
    included = set(universe)
    for accepted in condition_accepts.values():
        included &= accepted
    excluded = sorted(universe - included)
    included_rows = [_identity_record(key) for key in sorted(included)]
    excluded_rows = []
    for key in excluded:
        record = _identity_record(key)
        reasons = {}
        for condition, status_by_identity in condition_status.items():
            status = status_by_identity.get(key, "missing")
            reasons[f"{condition[0]}_{condition[1]}"] = status
        record.update(reasons)
        record["exclusion_reasons_json"] = json.dumps(
            [name for name, status in reasons.items() if status != "accepted"], sort_keys=True
        )
        excluded_rows.append(record)

    cohort_root = results_root / METHOD_NAME / "cohorts" / COHORT_NAME
    cohort_root.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(included_rows, cohort_root / "included_identities.csv", list(included_rows[0]))
    if excluded_rows:
        write_csv_atomic(excluded_rows, cohort_root / "excluded_identities.csv", list(excluded_rows[0]))
    projection_hashes = {}
    for dataset in ("sd300b", "sd300c"):
        for protocol in ("plain_self", "roll_self", "plain_roll"):
            output = cohort_root / dataset / f"{protocol}.csv"
            _project_manifest(repo_root / "protocols" / dataset / f"{protocol}.csv", output, included)
            projection_hashes[f"{dataset}/{protocol}"] = file_sha256(output)
    source_run_hashes = {
        f"{dataset}/{protocol}": bundle["metadata"]["result"]["sha256"]
        for (dataset, protocol), bundle in bundles.items()
    }
    payload = {
        "cohort_schema": "sift-geometric-cohort-v1",
        "cohort_name": COHORT_NAME,
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "membership_rule": (
            "accepted by the frozen SIFT decision rule in all four cross-dataset self conditions and present in both plain-roll manifests; plain-roll outcome is not used"
        ),
        "included_identity_count": len(included),
        "excluded_identity_count": len(excluded),
        "included_subject_count": len({row["subject_id"] for row in included_rows}),
        "included_identity_keys": sorted(included),
        "finger_distribution": dict(Counter(row["canonical_finger_position"] for row in included_rows)),
        "decision_rule_hash": decision_rule["decision_rule_hash"],
        "projection_hashes": projection_hashes,
        "source_run_hashes": source_run_hashes,
    }
    payload["cohort_provenance_hash"] = stable_hash(payload)
    write_json_atomic(payload, cohort_root / "cohort_metadata.json")
    return payload


def _load_primary_bundles(results_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for dataset in ("sd300b", "sd300c"):
        for protocol in ("plain_self", "roll_self", "plain_roll"):
            root = results_root / dataset / protocol / METHOD_NAME / BENCHMARK_CONTRACT_VERSION
            metadata_paths = sorted(root.glob(f"*/{METADATA_FILENAME}"))
            if len(metadata_paths) != 1:
                raise ValueError(f"Expected exactly one primary bundle for {dataset}/{protocol}; found {len(metadata_paths)}.")
            metadata_path = metadata_paths[0]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["run_spec"]["method"] != METHOD_NAME or metadata["run_spec"]["method_version"] != METHOD_VERSION:
                raise ValueError(f"Wrong method identity in {metadata_path}.")
            result_path = metadata_path.parent / RESULT_FILENAME
            rows = read_result_rows(result_path)
            manifest = read_pair_manifest(Path(metadata["run_spec"]["manifest_path"]))
            if [row["pair_id"] for row in rows] != [pair.pair_id for pair in manifest]:
                raise ValueError(f"Pair-ID alignment failure in {result_path}.")
            output[(dataset, protocol)] = {
                "metadata": metadata,
                "rows": rows,
                "result_path": result_path,
            }
    config_hashes = {bundle["metadata"]["run_spec"]["config_hash"] for bundle in output.values()}
    implementation_hashes = {
        bundle["metadata"]["run_spec"]["implementation_hash"] for bundle in output.values()
    }
    if len(config_hashes) != 1 or len(implementation_hashes) != 1:
        raise ValueError("Primary SIFT bundles do not share one config and implementation hash.")
    return output


def _diagnostics(row: dict[str, str]) -> dict[str, Any]:
    raw = row.get("compare_diagnostics", "")
    return json.loads(raw) if raw else {}


def _audit_rows(dataset: str, protocol: str, rows: list[dict[str, str]], threshold: float) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        diagnostics = _diagnostics(row)
        score = float(row["raw_score"]) if row["status"] == OK else None
        output.append(
            {
                "pair_id": row["pair_id"],
                "dataset": dataset,
                "protocol": protocol,
                "subject_id": row["subject_id"],
                "canonical_finger_position": row["canonical_finger_position"],
                "status": row["status"],
                "error_code": row["error_code"],
                "raw_score": "" if score is None else score,
                "threshold": threshold,
                "accepted": bool(score is not None and score >= threshold),
                "zero_score": bool(score == 0.0) if score is not None else False,
                "geometry_success": diagnostics.get("geometry_success", ""),
                "geometry_failure_reason": diagnostics.get("geometry_failure_reason", ""),
                "keypoint_count_a": diagnostics.get("keypoint_count_a", ""),
                "keypoint_count_b": diagnostics.get("keypoint_count_b", ""),
                "ratio_match_count": diagnostics.get("ratio_match_count_a_to_b", ""),
                "mutual_match_count": diagnostics.get("mutual_match_count", ""),
                "matches_submitted": diagnostics.get("matches_submitted_to_geometry", ""),
                "inlier_count": diagnostics.get("geometric_inlier_count", ""),
                "outlier_count": diagnostics.get("geometric_outlier_count", ""),
                "total_ms": row["total_ms"],
            }
        )
    return output


def _write_special_audits(rows: list[dict[str, Any]], report_root: Path) -> None:
    for filename, predicate in (
        ("failure_audit.csv", lambda row: row["status"] != OK),
        ("zero_score_audit.csv", lambda row: bool(row["zero_score"])),
        ("geometry_failure_audit.csv", lambda row: row["geometry_success"] is False),
    ):
        selected = [row for row in rows if predicate(row)]
        if selected:
            write_csv_atomic(selected, report_root / filename, list(selected[0]))


def _timing_fields(rows: list[dict[str, str]], column: str, prefix: str) -> dict[str, float | None]:
    values = [float(row[column]) for row in rows if row.get(column) not in (None, "")]
    return {
        f"{prefix}_ms_mean": _mean(values),
        f"{prefix}_ms_median": _median(values),
        f"{prefix}_ms_p95": _p95(values),
    }


def _diagnostic_summary(
    diagnostics: list[dict[str, Any]], key: str, prefix: str
) -> dict[str, float | None]:
    values = [float(diag[key]) for diag in diagnostics if diag and diag.get(key) is not None]
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_median": _median(values),
        f"{prefix}_p95": _p95(values),
    }


def _residual_summary(diagnostics: list[dict[str, Any]]) -> dict[str, float | None]:
    values = []
    for diag in diagnostics:
        raw = diag.get("residual_reference_pixels") if diag else None
        if isinstance(raw, dict) and raw.get("median") is not None:
            values.append(float(raw["median"]))
    return {
        "residual_median_reference_pixels_mean": _mean(values),
        "residual_median_reference_pixels_median": _median(values),
        "residual_median_reference_pixels_p95": _p95(values),
    }


def _project_manifest(source: Path, output: Path, included: set[str]) -> None:
    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if _identity_key(row["subject_id"], row["canonical_finger_position"]) in included
        ]
    write_csv_atomic(rows, output, MANIFEST_COLUMNS)


def _identity_key(subject_id: str, finger: str | int) -> str:
    return f"{subject_id}|{int(finger):02d}"


def _identity_record(key: str) -> dict[str, Any]:
    subject, finger = key.split("|", 1)
    return {
        "identity_key": key,
        "subject_id": subject,
        "canonical_finger_position": int(finger),
    }


def _artifact_inventory(root: Path) -> dict[str, Any]:
    self_referential_reports = {"artifact_hashes.json", "report_summary.json"}
    files = {
        str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in self_referential_reports
    }
    return {
        "inventory_schema": "sift-artifact-inventory-v1",
        "file_count": len(files),
        "files": files,
        "inventory_sha256": stable_hash(files),
    }


def _render_markdown(
    full: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    cohort: list[dict[str, Any]],
    cohort_metadata: dict[str, Any],
    decision_rule: dict[str, Any],
) -> str:
    lines = [
        "# SIFT Geometric Supervisor Tables",
        "",
        f"Method: `{METHOD_NAME}`",
        f"Method version: `{METHOD_VERSION}`",
        f"Cohort: `{COHORT_NAME}`",
        f"Included anatomical identities: {cohort_metadata['included_identity_count']}",
        f"Decision rule hash: `{decision_rule['decision_rule_hash']}`",
        "",
        "## Full manifests",
        "",
        "| dataset | protocol | pairs | success | failure | accepted | accepted % | median score | zero % | geometry failure % | median total ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in full:
        lines.append(_markdown_row(row))
    lines.extend(
        [
            "",
            "## Frozen evaluation subjects",
            "",
            "These rows exclude every development subject used by pilot selection and threshold calibration.",
            "",
            "| dataset | protocol | pairs | success | failure | accepted | accepted % | median score | zero % | geometry failure % | median total ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in evaluation:
        lines.append(_markdown_row(row))
    lines.extend(
        [
            "",
            "## SIFT-specific cohort",
            "",
            "Membership uses only the four frozen self decisions. Plain-roll outcomes are retained and do not affect membership.",
            "",
            "| dataset | protocol | pairs | success | failure | accepted | accepted % | median score | zero % | geometry failure % | median total ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cohort:
        lines.append(_markdown_row(row))
    return "\n".join(lines) + "\n"


def _markdown_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['dataset']} | {row['protocol']} | {row['pairs']} | {row['success_count']} | "
        f"{row['failure_count']} | {row['accepted_count']} | {row['accepted_percentage']:.2f} | "
        f"{_fmt(row['raw_score_median'])} | {row['zero_score_percentage']:.2f} | "
        f"{row['geometry_failure_percentage']:.2f} | {_fmt(row['total_ms_median'])} |"
    )


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _p95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def _fmt(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.3f}"
