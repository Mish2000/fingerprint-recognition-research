"""Supervisor and technical reporting for the frozen HarrisZ+ joint-500 pilot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..bundle import (
    create_candidate_directory,
    discard_candidate_directory,
    publish_candidate_directory,
)
from ..contract import OK
from ..hashing import canonical_json_bytes, file_sha256
from .pilot import (
    DATASETS,
    METHOD_NAME,
    METHOD_VERSION,
    PILOT_NAMESPACE,
    RUN_CONDITIONS,
    THRESHOLD,
    _artifact_record,
    _publish_immutable_json,
    _read_json,
    accepted_result_row,
    load_bundle,
    project_paths,
)
from .preflight import EXPECTED_SELECTION_SHA256


SUPERVISOR_SCHEMA_VERSION = "harriszplus-joint-500-supervisor-report-v1"
TECHNICAL_SCHEMA_VERSION = "harriszplus-joint-500-technical-provenance-v1"
ARTIFACT_SCHEMA_VERSION = "harriszplus-joint-500-artifact-manifest-v1"
SUPERVISOR_CSV_COLUMNS = ("section_number", "section", "dataset", "metric", "value")


class HarrisZPlusReportingError(ValueError):
    """Raised when the eight immutable bundles cannot support the requested report."""


def summarize_condition(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Summarize decisions, score/timing, detector, matching, and inlier evidence."""

    ok_rows = [row for row in rows if row["status"] == OK]
    accepted = [row for row in rows if accepted_result_row(row)]
    rejected = [row for row in ok_rows if not accepted_result_row(row)]
    failures = [row for row in rows if row["status"] != OK]
    scores = [float(row["raw_score"]) for row in ok_rows]
    compare_diagnostics = [_diagnostics(row, "compare_diagnostics") for row in ok_rows]
    prepare_diagnostics = [
        diagnostics
        for row in rows
        for column in ("prepare_a_diagnostics", "prepare_b_diagnostics")
        if (diagnostics := _diagnostics(row, column))
    ]
    keypoints_a = [_number(diag.get("keypoint_count_a")) for diag in compare_diagnostics]
    keypoints_b = [_number(diag.get("keypoint_count_b")) for diag in compare_diagnostics]
    mutual = [_number(diag.get("mutual_match_count")) for diag in compare_diagnostics]
    tentative = [_number(diag.get("tentative_match_count")) for diag in compare_diagnostics]
    inliers = [_number(diag.get("geometric_inlier_count")) for diag in compare_diagnostics]
    inlier_ratios = [_number(diag.get("inlier_ratio")) for diag in compare_diagnostics]
    descriptors = [_number(diag.get("descriptor_count")) for diag in prepare_diagnostics]
    final_keypoints = [
        _number(diag.get("final_keypoint_count", diag.get("detector_keypoint_count")))
        for diag in prepare_diagnostics
    ]
    peak_allocated = [_number(diag.get("peak_vram_allocated")) for diag in prepare_diagnostics]
    peak_reserved = [_number(diag.get("peak_vram_reserved")) for diag in prepare_diagnostics]
    prepare_timing_fields = (
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
    compare_timing_fields = (
        "matcher_cpu_ms",
        "ransac_cpu_ms",
        "compare_total_ms",
        "end_to_end_wall_ms",
    )
    failure_counts: dict[str, int] = {}
    for row in failures:
        failure_counts[row["status"]] = failure_counts.get(row["status"], 0) + 1
    return {
        "total": len(rows),
        "status_ok": len(ok_rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "failures": len(failures),
        "failure_counts": dict(sorted(failure_counts.items())),
        "acceptance_percentage": _percentage(len(accepted), len(rows)),
        "score_equal_threshold": sum(
            1 for row in ok_rows if int(float(row["raw_score"])) == THRESHOLD
        ),
        "score_zero": sum(1 for score in scores if score == 0),
        "raw_score": _numeric_summary(scores),
        "method_compare_ms": _numeric_summary(_column_numbers(rows, "method_compare_ms")),
        "total_pair_ms": _numeric_summary(_column_numbers(rows, "total_ms")),
        "prepare_a_ms": _numeric_summary(_column_numbers(rows, "method_prepare_a_ms")),
        "prepare_b_ms": _numeric_summary(_column_numbers(rows, "method_prepare_b_ms")),
        "keypoint_count_a": _numeric_summary(keypoints_a),
        "keypoint_count_b": _numeric_summary(keypoints_b),
        "detector_final_keypoint_count": _numeric_summary(final_keypoints),
        "descriptor_count": _numeric_summary(descriptors),
        "tentative_match_count": _numeric_summary(tentative),
        "mutual_match_count": _numeric_summary(mutual),
        "geometric_inlier_count": _numeric_summary(inliers),
        "inlier_ratio": _numeric_summary(inlier_ratios),
        "peak_vram_allocated_bytes": max(_finite(peak_allocated), default=None),
        "peak_vram_reserved_bytes": max(_finite(peak_reserved), default=None),
        "timing_breakdown_ms": {
            **{
                field: _numeric_summary(_number(diag.get(field)) for diag in prepare_diagnostics)
                for field in prepare_timing_fields
            },
            **{
                field: _numeric_summary(_number(diag.get(field)) for diag in compare_diagnostics)
                for field in compare_timing_fields
            },
        },
    }


def build_supervisor_report(*, project_root: Path) -> dict[str, Any]:
    """Publish MD/CSV/JSON with the compact table and exact six-stage structure."""

    paths = project_paths(project_root)
    report_root = paths["pilot_root"] / "report"
    expected_files = tuple(report_root / name for name in (
        "supervisor_report.md",
        "supervisor_report.csv",
        "supervisor_report.json",
    ))
    report_exists = report_root.exists()
    if report_exists:
        if not all(path.is_file() for path in expected_files):
            raise HarrisZPlusReportingError(f"Partial immutable report directory exists: {report_root}")

    _validate_exact_run_namespace(paths["pilot_root"])

    views = {
        (dataset, label): load_bundle(project_root=paths["project_root"], dataset=dataset, label=label)
        for dataset, label, _, _ in RUN_CONDITIONS
    }
    summaries = {
        (dataset, label): summarize_condition(view.rows)
        for (dataset, label), view in views.items()
    }
    survivor_summaries = {
        dataset: _read_json(paths["pilot_root"] / f"survivors/{dataset}/summary.json")
        for dataset in DATASETS
    }
    _validate_report_inputs(views, summaries, survivor_summaries)
    sections = _six_sections(summaries, survivor_summaries)
    compact = _compact_rows(summaries)
    condition_details = {
        dataset: {
            label: summaries[(dataset, label)]
            for label in (
                "plain_self",
                "roll_self",
                "plain_roll_genuine",
                "plain_roll_negative",
            )
        }
        for dataset in DATASETS
    }
    payload = {
        "schema_version": SUPERVISOR_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "operational_threshold": THRESHOLD,
        "threshold_calibrated_for_target_far": False,
        "compact_table": compact,
        "sections": sections,
        "condition_summaries": condition_details,
        "excluded_analyses": [
            "SourceAFIS/SIFT comparison",
            "ROC/AUC/EER",
            "calibration",
            "fusion",
            "GPU benchmarking claims",
            "additional scientific analysis",
        ],
    }
    csv_rows = _supervisor_csv_rows(sections)
    markdown = _supervisor_markdown(sections, compact, summaries)
    expected_bytes = {
        "supervisor_report.json": _pretty_json_bytes(payload),
        "supervisor_report.csv": _csv_bytes(csv_rows, SUPERVISOR_CSV_COLUMNS),
        "supervisor_report.md": markdown.encode("utf-8"),
    }
    if report_exists:
        _validate_exact_immutable_directory(
            report_root,
            expected_bytes,
            label="supervisor report",
        )
        return {
            **payload,
            "report_root": str(report_root),
            "files": {path.name: _artifact_record(path) for path in expected_files},
        }

    candidate = create_candidate_directory(report_root)
    try:
        for filename, content in expected_bytes.items():
            (candidate / filename).write_bytes(content)
        publish_candidate_directory(candidate, report_root)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)
    return {
        **payload,
        "report_root": str(report_root),
        "files": {path.name: _artifact_record(path) for path in expected_files},
    }


def build_technical_provenance(
    *,
    project_root: Path,
    integrity_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish detailed implementation/runtime/run provenance after integrity passes."""

    paths = project_paths(project_root)
    target = paths["pilot_root"] / "technical_provenance.json"
    target_exists = target.exists()
    if integrity_report is None:
        integrity_report = _read_json(paths["pilot_root"] / "integrity/protected_integrity.json")
    if integrity_report.get("protected_artifacts_unchanged") is not True:
        raise HarrisZPlusReportingError("Protected before/after integrity must pass before finalization.")
    normalized_integrity = dict(integrity_report)
    normalized_integrity.pop("path", None)
    normalized_integrity.pop("sha256", None)

    supervisor = build_supervisor_report(project_root=paths["project_root"])
    freeze = _read_json(paths["config"] / "freeze_manifest.json")
    config = _read_json(paths["config"] / "runner_config.json")
    algorithm_config = _read_json(paths["config"] / "algorithm_config.json")
    environment = _read_json(paths["config"] / "environment.json")
    implementation = _read_json(paths["config"] / "implementation_provenance.json")
    implementation_components = _read_json(paths["config"] / "implementation_components.json")
    runtime_identity = _read_json(paths["config"] / "runtime_identity.json")
    preflight = _read_json(paths["preflight"])
    if (
        preflight.get("ppi_coordinate_handling_all_passed") is not True
        or not isinstance(preflight.get("device_binding"), Mapping)
        or preflight["device_binding"].get("passed") is not True
        or not isinstance(preflight.get("memory"), Mapping)
        or preflight["memory"].get("passed") is not True
        or not isinstance(preflight.get("timing_synchronization"), Mapping)
        or preflight["timing_synchronization"].get(
            "required_timing_fields_all_passed"
        )
        is not True
    ):
        raise HarrisZPlusReportingError(
            "Technical provenance requires passing PPI, device, memory, and "
            "synchronized-timing preflight evidence."
        )
    selection_reference = _read_json(paths["pilot_root"] / "selected_identities_reference.json")
    survivors = {
        dataset: _read_json(paths["pilot_root"] / f"survivors/{dataset}/summary.json")
        for dataset in DATASETS
    }

    runs = []
    all_config_hashes = set()
    all_implementation_hashes = set()
    for execution_index, (dataset, label, protocol, role) in enumerate(RUN_CONDITIONS, start=1):
        view = load_bundle(project_root=paths["project_root"], dataset=dataset, label=label)
        summary = summarize_condition(view.rows)
        all_config_hashes.add(view.metadata["config_hash"])
        all_implementation_hashes.add(view.metadata["implementation_hash"])
        runs.append(
            {
                "actual_execution_index": execution_index,
                "dataset": dataset,
                "label": label,
                "protocol": protocol,
                "role": role,
                "bundle_path": str(view.bundle_path),
                "bundle_tree_sha256": _bundle_tree_sha256(view.bundle_path),
                "bundle_tree_contract": (
                    "SHA-256 over ordered canonical JSON {path,size,sha256} records for "
                    "pairs.csv and run_metadata.json, each followed by LF"
                ),
                "manifest": _artifact_record(view.manifest_path),
                "pairs": {
                    **_artifact_record(view.bundle_path / "pairs.csv"),
                    "score_payload_sha256": view.metadata["result"]["score_payload_sha256"],
                },
                "run_metadata": _artifact_record(view.bundle_path / "run_metadata.json"),
                "config_hash": view.metadata["config_hash"],
                "implementation_hash": view.metadata["implementation_hash"],
                "execution_wall_ms": view.metadata["execution_wall_ms"],
                "warm_up": view.metadata["warm_up"],
                "run_isolation": view.metadata.get("run_isolation"),
                "summary": summary,
            }
        )
    _validate_bundle_freeze_identities(
        all_config_hashes,
        all_implementation_hashes,
        freeze,
    )

    total_false_matches = sum(
        summarize_condition(
            load_bundle(
                project_root=paths["project_root"], dataset=dataset, label="plain_roll_negative"
            ).rows
        )["accepted"]
        for dataset in DATASETS
    )
    payload = {
        "schema_version": TECHNICAL_SCHEMA_VERSION,
        "namespace": PILOT_NAMESPACE,
        "status": "complete",
        "method": {
            "name": METHOD_NAME,
            "version": METHOD_VERSION,
            "pipeline": (
                "HarrisZ+ detector -> deterministic scale/orientation -> RootSIFT -> existing mutual "
                "ratio matching -> existing partial-affine RANSAC -> geometric-inlier-count score"
            ),
            "score_direction": "higher_is_more_similar",
            "raw_score": "integer geometric_inlier_count",
            "decision_threshold_in_adapter": None,
            "pilot_decision": "status == ok and geometric_inlier_count >= 4",
        },
        "clean_room_and_reference_provenance": implementation,
        "architecture": {
            "reference_cpu": "clear NumPy/OpenCV dense-response validation backend",
            "cuda": (
                "PyTorch/CUDA dense response, local-maxima, and eigen/subpixel refinement; "
                "deterministic CPU distance/uniform selection, orientation, RootSIFT, matching, "
                "and RANSAC"
            ),
            "native_resolution_grayscale": True,
            "color_grad": False,
            "max_keypoints": 3000,
            "root_sift_reuse": "fingerprint_benchmark.sift.descriptors.rootsift",
            "matcher_reuse": "fingerprint_benchmark.sift.matching.match_descriptors, mutual Lowe 0.75",
            "ransac_reuse": "fingerprint_benchmark.sift.geometry.verify_geometry, partial affine",
            "ppi_geometry_threshold": "3 px at 1000 PPI, linearly 6 px at 2000 PPI",
            "scale_mapping": algorithm_config.get("keypoint_size_formula"),
            "orientation": {
                key: value
                for key, value in algorithm_config.items()
                if key.startswith("orientation_")
            },
        },
        "freeze": {
            **freeze,
            "config_directory": str(paths["config"]),
            "runner_config": config,
            "algorithm_config": algorithm_config,
            "implementation_components": implementation_components,
            "runtime_identity": runtime_identity,
        },
        "environment": environment,
        "preflight": {
            "path": str(paths["preflight"]),
            "sha256": file_sha256(paths["preflight"]),
            "ppi_coordinate_handling_all_passed": True,
            "report": preflight,
        },
        "selection": {
            "path": str(paths["pilot_root"] / "selected_identities_reference.json"),
            "sha256": file_sha256(paths["pilot_root"] / "selected_identities_reference.json"),
            "expected_source_sha256": EXPECTED_SELECTION_SHA256,
            "reference": selection_reference,
        },
        "self_filtering": survivors,
        "execution": {
            "requested_bundle_count": 8,
            "completed_valid_bundle_count": len(runs),
            "full_repeat_count": 0,
            "serial_pair_execution": True,
            "timing_mode": "cold_pair",
            "prepare_operations_per_pair": 2,
            "cross_pair_cache": False,
            "bundle_paths_are_exact_short_requested_paths": True,
            "runs": runs,
            "total_execution_wall_ms": sum(float(run["execution_wall_ms"]) for run in runs),
        },
        "reports": {
            filename: record for filename, record in supervisor["files"].items()
        },
        "integrity": normalized_integrity,
        "total_actual_false_matches": total_false_matches,
        "artifact_manifest_path": str(paths["pilot_root"] / "artifact_manifest.json"),
        "assertions": {
            "datasets_treated_read_only": True,
            "dataset_tree_rescanned": False,
            "sourceafis_code_or_results_changed": False,
            "sift_code_or_results_changed": False,
            "shared_accuracy_artifacts_changed": False,
            "selection_changed": False,
            "threshold_changed_after_results": False,
            "tuning_or_calibration_on_500_results": False,
            "sourceafis_threshold_40_used": False,
            "roc_auc_eer_or_fusion_performed": False,
            "gpu_cpu_efficiency_ranking_claimed": False,
            "extra_research_pilot_performed": False,
            "exactly_eight_bundles": True,
        },
    }
    expected_bytes = _pretty_json_bytes(payload)
    if target_exists:
        _validate_exact_immutable_file(
            target,
            expected_bytes,
            label="technical provenance",
        )
        return {**payload, "path": str(target), "sha256": file_sha256(target)}
    _publish_immutable_json(target, payload)
    return {**payload, "path": str(target), "sha256": file_sha256(target)}


def build_artifact_manifest(*, project_root: Path) -> dict[str, Any]:
    """Seal all pilot files plus references to the external immutable config directory."""

    paths = project_paths(project_root)
    target = paths["pilot_root"] / "artifact_manifest.json"
    if target.exists():
        payload = _read_json(target)
        _validate_artifact_manifest(paths["pilot_root"], payload)
        return {**payload, "path": str(target), "sha256": file_sha256(target)}
    technical = paths["pilot_root"] / "technical_provenance.json"
    if not technical.is_file():
        raise HarrisZPlusReportingError("Technical provenance must be published before artifact sealing.")
    _validate_expected_pilot_artifact_inventory(paths["pilot_root"])
    files = []
    for path in sorted(
        (item for item in paths["pilot_root"].rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(paths["pilot_root"]).as_posix(),
    ):
        if path.resolve() == target.resolve():
            continue
        files.append(
            {
                "path": path.relative_to(paths["pilot_root"]).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    config_files = []
    for path in sorted(item for item in paths["config"].glob("*.json") if item.is_file()):
        config_files.append(
            {
                "path": path.relative_to(paths["project_root"]).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    preflight_files = [_artifact_record(paths["preflight"])]
    preflight_files[0]["path"] = paths["preflight"].relative_to(
        paths["project_root"]
    ).as_posix()
    tree_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in files)
    ).hexdigest()
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "namespace": PILOT_NAMESPACE,
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": sum(record["size"] for record in files),
        "tree_sha256": tree_sha,
        "files": files,
        "external_frozen_config_files": config_files,
        "external_frozen_config_tree_sha256": hashlib.sha256(
            b"".join(canonical_json_bytes(record) + b"\n" for record in config_files)
        ).hexdigest(),
        "external_preflight_files": preflight_files,
        "external_preflight_tree_sha256": hashlib.sha256(
            b"".join(canonical_json_bytes(record) + b"\n" for record in preflight_files)
        ).hexdigest(),
        "immutable": True,
        "overwrite_allowed": False,
    }
    _publish_immutable_json(target, payload)
    _validate_artifact_manifest(paths["pilot_root"], payload)
    return {**payload, "path": str(target), "sha256": file_sha256(target)}


def _six_sections(
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
    survivors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "number": 1,
            "title": "Record counts",
            "datasets": {
                dataset: {"plain_records": 500, "roll_records": 500}
                for dataset in DATASETS
            },
        },
        {
            "number": 2,
            "title": "PLAIN self-comparisons",
            "datasets": {
                dataset: _self_section(summaries[(dataset, "plain_self")])
                for dataset in DATASETS
            },
        },
        {
            "number": 3,
            "title": "ROLL self-comparisons",
            "datasets": {
                dataset: _self_section(summaries[(dataset, "roll_self")])
                for dataset in DATASETS
            },
        },
        {
            "number": 4,
            "title": "PLAIN-to-corresponding-ROLL record matching",
            "datasets": {
                dataset: {
                    "survivors": survivors[dataset]["survivor_count"],
                    "excluded_after_self": survivors[dataset]["excluded_count"],
                    "ground_truth_same_subject_same_finger": summaries[
                        (dataset, "plain_roll_genuine")
                    ]["total"],
                    "ground_truth_wrong": 0,
                    "harriszplus_classified_matching": summaries[
                        (dataset, "plain_roll_genuine")
                    ]["accepted"],
                    "harriszplus_classified_non_matching": summaries[
                        (dataset, "plain_roll_genuine")
                    ]["rejected"],
                    "failures": summaries[(dataset, "plain_roll_genuine")]["failures"],
                }
                for dataset in DATASETS
            },
            "filtering_scope": "independent per dataset",
        },
        {
            "number": 5,
            "title": "PLAIN versus corresponding ROLL",
            "datasets": {
                dataset: _comparison_section(summaries[(dataset, "plain_roll_genuine")])
                for dataset in DATASETS
            },
        },
        {
            "number": 6,
            "title": "PLAIN versus next subject's ROLL",
            "pairing_method": (
                "next surviving subject within the same canonical finger position, "
                "selection_index order, circular shift by one"
            ),
            "datasets": {
                dataset: _negative_section(summaries[(dataset, "plain_roll_negative")])
                for dataset in DATASETS
            },
        },
    ]


def _compact_rows(summaries: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, str]]:
    labels = (
        ("PLAIN מול עצמו", "plain_self", "self"),
        ("ROLL מול עצמו", "roll_self", "self"),
        ("PLAIN מול ROLL המתאים", "plain_roll_genuine", "genuine"),
        ("PLAIN מול ROLL של הנבדק הבא", "plain_roll_negative", "negative"),
    )
    rows = []
    for stage, label, role in labels:
        row = {"stage": stage}
        for dataset in DATASETS:
            summary = summaries[(dataset, label)]
            if role == "negative":
                row[dataset] = (
                    f"incorrectly accepted {summary['accepted']}; correctly rejected "
                    f"{summary['rejected']}; failures {summary['failures']}"
                )
            else:
                row[dataset] = (
                    f"accepted {summary['accepted']}; rejected {summary['rejected']}; "
                    f"failures {summary['failures']}"
                )
        rows.append(row)
    return rows


def _self_section(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accepted": summary["accepted"],
        "rejected": summary["rejected"],
        "failures": summary["failures"],
        "removed": summary["rejected"] + summary["failures"],
        "remaining": summary["accepted"],
    }


def _comparison_section(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "total": summary["total"],
        "accepted": summary["accepted"],
        "rejected": summary["rejected"],
        "failures": summary["failures"],
        "acceptance_percentage": summary["acceptance_percentage"],
        "mean_score": summary["raw_score"]["mean"],
        "median_score": summary["raw_score"]["median"],
        "mean_method_compare_time_ms": summary["method_compare_ms"]["mean"],
        "mean_total_pair_time_ms": summary["total_pair_ms"]["mean"],
        "mean_keypoint_count_a": summary["keypoint_count_a"]["mean"],
        "mean_keypoint_count_b": summary["keypoint_count_b"]["mean"],
        "mean_mutual_match_count": summary["mutual_match_count"]["mean"],
        "mean_geometric_inlier_count": summary["geometric_inlier_count"]["mean"],
    }


def _negative_section(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "total": summary["total"],
        "incorrectly_accepted": summary["accepted"],
        "correctly_rejected": summary["rejected"],
        "failures": summary["failures"],
        "false_match_percentage": summary["acceptance_percentage"],
        "mean_score": summary["raw_score"]["mean"],
        "median_score": summary["raw_score"]["median"],
        "mean_method_compare_time_ms": summary["method_compare_ms"]["mean"],
        "mean_total_pair_time_ms": summary["total_pair_ms"]["mean"],
    }


def _validate_report_inputs(
    views: Mapping[tuple[str, str], Any],
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
    survivors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(views) != 8:
        raise HarrisZPlusReportingError(f"Expected eight bundles, got {len(views)}.")
    for dataset in DATASETS:
        for label in ("plain_self", "roll_self"):
            if summaries[(dataset, label)]["total"] != 500:
                raise HarrisZPlusReportingError(f"{dataset}/{label} must contain exactly 500 rows.")
        expected = int(survivors[dataset]["survivor_count"])
        for label in ("plain_roll_genuine", "plain_roll_negative"):
            if summaries[(dataset, label)]["total"] != expected:
                raise HarrisZPlusReportingError(
                    f"{dataset}/{label} row count does not equal per-dataset survivors."
                )
        if survivors[dataset].get("genuine_or_negative_result_used_for_filtering") is not False:
            raise HarrisZPlusReportingError("Self filtering was contaminated by downstream decisions.")


def _supervisor_csv_rows(sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for section in sections:
        for dataset, metrics in section["datasets"].items():
            for metric, value in metrics.items():
                rows.append(
                    {
                        "section_number": section["number"],
                        "section": section["title"],
                        "dataset": dataset,
                        "metric": metric,
                        "value": _csv_scalar(value),
                    }
                )
    return rows


def _supervisor_markdown(
    sections: Sequence[Mapping[str, Any]],
    compact: Sequence[Mapping[str, str]],
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    section = {item["number"]: item for item in sections}
    lines = [
        "# HarrisZ+ RootSIFT geometric joint 500 pilot supervisor report",
        "",
        "Operational rule: `accepted` only when `status == ok` and integer geometric-inlier score `>= 4`. "
        "This is a frozen pilot rule, not a HarrisZ+ FAR-calibrated threshold.",
        "",
        "| שלב | SD300b | SD300c |",
        "|---|---:|---:|",
    ]
    for row in compact:
        lines.append(f"| {row['stage']} | {row['sd300b']} | {row['sd300c']} |")
    lines.extend(["", "## 1. Record counts", "", "| Dataset | PLAIN | ROLL |", "|---|---:|---:|"])
    for dataset in DATASETS:
        values = section[1]["datasets"][dataset]
        lines.append(
            f"| {_dataset_label(dataset)} | {values['plain_records']} | {values['roll_records']} |"
        )
    for number, title in ((2, "PLAIN self-comparisons"), (3, "ROLL self-comparisons")):
        lines.extend(
            [
                "",
                f"## {number}. {title}",
                "",
                "| Dataset | Accepted | Rejected | Failures | Removed | Remaining |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in DATASETS:
            values = section[number]["datasets"][dataset]
            lines.append(
                f"| {_dataset_label(dataset)} | {values['accepted']} | {values['rejected']} | "
                f"{values['failures']} | {values['removed']} | {values['remaining']} |"
            )
    lines.extend(
        [
            "",
            "## 4. PLAIN-to-corresponding-ROLL record matching",
            "",
            "Self filtering was performed independently for each dataset. No identity was replaced.",
            "",
            "| Dataset | Survivors | Excluded after self | Ground truth same subject/finger | "
            "Ground truth wrong | HarrisZ+ matching | HarrisZ+ non-matching | Failures |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        value = section[4]["datasets"][dataset]
        lines.append(
            f"| {_dataset_label(dataset)} | {value['survivors']} | {value['excluded_after_self']} | "
            f"{value['ground_truth_same_subject_same_finger']} | {value['ground_truth_wrong']} | "
            f"{value['harriszplus_classified_matching']} | "
            f"{value['harriszplus_classified_non_matching']} | {value['failures']} |"
        )
    lines.extend(
        [
            "",
            "## 5. PLAIN versus corresponding ROLL",
            "",
            "| Dataset | Total | Accepted | Rejected | Failures | Acceptance % | Score mean | "
            "Score median | Mean method compare ms | Mean total pair ms | Mean keypoints A/B | "
            "Mean mutual matches | Mean inliers |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        value = section[5]["datasets"][dataset]
        lines.append(
            f"| {_dataset_label(dataset)} | {value['total']} | {value['accepted']} | "
            f"{value['rejected']} | {value['failures']} | {_fmt(value['acceptance_percentage'])} | "
            f"{_fmt(value['mean_score'])} | {_fmt(value['median_score'])} | "
            f"{_fmt(value['mean_method_compare_time_ms'])} | {_fmt(value['mean_total_pair_time_ms'])} | "
            f"{_fmt(value['mean_keypoint_count_a'])}/{_fmt(value['mean_keypoint_count_b'])} | "
            f"{_fmt(value['mean_mutual_match_count'])} | {_fmt(value['mean_geometric_inlier_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 6. PLAIN versus next subject's ROLL",
            "",
            "Pairing method: next survivor within the same canonical finger position, ordered by "
            "`selection_index`, circular shift `1`.",
            "",
            "| Dataset | Total | Incorrectly accepted | Correctly rejected | Failures | False-match % | "
            "Score mean | Score median | Mean method compare ms | Mean total pair ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        value = section[6]["datasets"][dataset]
        lines.append(
            f"| {_dataset_label(dataset)} | {value['total']} | {value['incorrectly_accepted']} | "
            f"{value['correctly_rejected']} | {value['failures']} | "
            f"{_fmt(value['false_match_percentage'])} | {_fmt(value['mean_score'])} | "
            f"{_fmt(value['median_score'])} | {_fmt(value['mean_method_compare_time_ms'])} | "
            f"{_fmt(value['mean_total_pair_time_ms'])} |"
        )
    lines.extend(
        [
            "",
            "Timing note: HarrisZ+ preparation uses a CUDA detector. Earlier SourceAFIS and SIFT pilots "
            "may use different backends, so this report makes no direct CPU-efficiency ranking.",
            "",
        ]
    )
    return "\n".join(lines)


def _bundle_tree_sha256(bundle: Path) -> str:
    records = [
        {
            "path": name,
            "size": (bundle / name).stat().st_size,
            "sha256": file_sha256(bundle / name),
        }
        for name in ("pairs.csv", "run_metadata.json")
    ]
    return hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    ).hexdigest()


def _expected_pilot_artifact_paths() -> set[str]:
    expected = {
        "selected_identities_reference.json",
        "manifests/base_manifest_provenance.json",
        "integrity/protected_before.json",
        "integrity/protected_after.json",
        "integrity/protected_integrity.json",
        "report/supervisor_report.json",
        "report/supervisor_report.csv",
        "report/supervisor_report.md",
        "technical_provenance.json",
    }
    for dataset in DATASETS:
        expected.update(
            {
                f"manifests/{dataset}/plain_self.csv",
                f"manifests/{dataset}/roll_self.csv",
                f"manifests/{dataset}/plain_roll_genuine.csv",
                f"manifests/{dataset}/plain_roll_negative.csv",
                f"manifests/{dataset}/pairing_map.csv",
                f"survivors/{dataset}/included_identities.csv",
                f"survivors/{dataset}/excluded_identities.csv",
                f"survivors/{dataset}/summary.json",
            }
        )
    for dataset, label, _, _ in RUN_CONDITIONS:
        expected.update(
            {
                f"runs/{dataset}/{label}/pairs.csv",
                f"runs/{dataset}/{label}/run_metadata.json",
            }
        )
    return expected


def _validate_expected_pilot_artifact_inventory(pilot_root: Path) -> None:
    actual = {
        path.relative_to(pilot_root).as_posix()
        for path in pilot_root.rglob("*")
        if path.is_file()
        and path.resolve() != (pilot_root / "artifact_manifest.json").resolve()
    }
    expected = _expected_pilot_artifact_paths()
    if actual != expected:
        raise HarrisZPlusReportingError(
            "Pilot namespace does not match the explicit final artifact whitelist; "
            f"added={sorted(actual - expected)[:8]}, missing={sorted(expected - actual)[:8]}."
        )


def _validate_exact_run_namespace(pilot_root: Path) -> None:
    runs_root = pilot_root / "runs"
    expected_bundle_paths = {
        Path(dataset) / label for dataset, label, _, _ in RUN_CONDITIONS
    }
    expected_directories = {
        Path(dataset) for dataset in DATASETS
    } | expected_bundle_paths
    expected_files = {
        bundle / filename
        for bundle in expected_bundle_paths
        for filename in ("pairs.csv", "run_metadata.json")
    }
    actual_directories = {
        path.relative_to(runs_root)
        for path in runs_root.rglob("*")
        if path.is_dir()
    } if runs_root.is_dir() else set()
    actual_files = {
        path.relative_to(runs_root)
        for path in runs_root.rglob("*")
        if path.is_file()
    } if runs_root.is_dir() else set()
    if actual_directories != expected_directories or actual_files != expected_files:
        raise HarrisZPlusReportingError(
            "Run namespace must contain exactly eight bundles with only pairs.csv and "
            "run_metadata.json in each bundle."
        )


def _validate_exact_immutable_file(
    path: Path,
    expected_bytes: bytes,
    *,
    label: str,
) -> None:
    if not path.is_file() or path.read_bytes() != expected_bytes:
        raise HarrisZPlusReportingError(
            f"Existing immutable {label} does not match current validated inputs: {path}"
        )


def _validate_exact_immutable_directory(
    root: Path,
    expected_files: Mapping[str, bytes],
    *,
    label: str,
) -> None:
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected_files):
        raise HarrisZPlusReportingError(
            f"Existing immutable {label} inventory does not match current expected files."
        )
    for logical_path, expected_bytes in expected_files.items():
        _validate_exact_immutable_file(
            root / logical_path,
            expected_bytes,
            label=label,
        )


def _validate_bundle_freeze_identities(
    config_hashes: set[str],
    implementation_hashes: set[str],
    freeze: Mapping[str, Any],
) -> None:
    expected_config = freeze.get("canonical_config_hash")
    expected_implementation = freeze.get("implementation_hash")
    if (
        not isinstance(expected_config, str)
        or not isinstance(expected_implementation, str)
        or config_hashes != {expected_config}
        or implementation_hashes != {expected_implementation}
    ):
        raise HarrisZPlusReportingError(
            "Every pilot bundle must match the current frozen config and implementation identities."
        )


def _validate_artifact_manifest(root: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise HarrisZPlusReportingError("Artifact manifest schema mismatch.")
    if (
        payload.get("namespace") != PILOT_NAMESPACE
        or payload.get("hash_algorithm") != "sha256"
        or payload.get("immutable") is not True
        or payload.get("overwrite_allowed") is not False
    ):
        raise HarrisZPlusReportingError(
            "Artifact manifest immutable namespace/hash semantics changed."
        )
    files = payload.get("files")
    if not isinstance(files, list):
        raise HarrisZPlusReportingError("Artifact manifest lacks files.")
    records = []
    recorded_paths: set[str] = set()
    for record in files:
        logical_path = record["path"]
        if logical_path in recorded_paths:
            raise HarrisZPlusReportingError(f"Duplicate sealed artifact path: {logical_path}")
        recorded_paths.add(logical_path)
        path = (root / logical_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise HarrisZPlusReportingError(
                f"Sealed artifact escapes pilot root: {logical_path}"
            ) from exc
        if (
            not path.is_file()
            or path.stat().st_size != record["size"]
            or file_sha256(path) != record["sha256"]
        ):
            raise HarrisZPlusReportingError(f"Sealed artifact changed: {path}")
        records.append(record)
    tree_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    ).hexdigest()
    if (
        tree_sha != payload.get("tree_sha256")
        or len(records) != payload.get("file_count")
        or sum(record["size"] for record in records) != payload.get("total_bytes")
    ):
        raise HarrisZPlusReportingError("Artifact manifest aggregate identity mismatch.")
    current_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() != (root / "artifact_manifest.json").resolve()
    }
    if current_paths != recorded_paths:
        raise HarrisZPlusReportingError(
            "Pilot artifact inventory has added or removed unsealed files."
        )
    if recorded_paths != _expected_pilot_artifact_paths():
        raise HarrisZPlusReportingError(
            "Sealed pilot artifacts do not match the explicit final artifact whitelist."
        )
    project_root = root.parents[2]
    external_roots = project_paths(project_root)
    for key, tree_key, label, namespace_root in (
        (
            "external_frozen_config_files",
            "external_frozen_config_tree_sha256",
            "External frozen config",
            external_roots["config"],
        ),
        (
            "external_preflight_files",
            "external_preflight_tree_sha256",
            "External preflight",
            external_roots["preflight"].parent,
        ),
    ):
        external_records = payload.get(key)
        if not isinstance(external_records, list) or not external_records:
            raise HarrisZPlusReportingError(f"{label} inventory is missing.")
        recorded_external_paths: set[str] = set()
        for record in external_records:
            logical_path = record["path"]
            if logical_path in recorded_external_paths:
                raise HarrisZPlusReportingError(
                    f"Duplicate {label.lower()} artifact path: {logical_path}"
                )
            recorded_external_paths.add(logical_path)
            path = (project_root / logical_path).resolve()
            try:
                path.relative_to(namespace_root.resolve())
            except ValueError as exc:
                raise HarrisZPlusReportingError(
                    f"{label} artifact escapes its namespace: {logical_path}"
                ) from exc
            if (
                not path.is_file()
                or path.stat().st_size != record["size"]
                or file_sha256(path) != record["sha256"]
            ):
                raise HarrisZPlusReportingError(f"{label} artifact changed: {path}")
        current_external_paths = {
            path.relative_to(project_root).as_posix()
            for path in namespace_root.rglob("*")
            if path.is_file()
        }
        if current_external_paths != recorded_external_paths:
            raise HarrisZPlusReportingError(
                f"{label} inventory has added, removed, or unsealed files."
            )
        external_tree = hashlib.sha256(
            b"".join(
                canonical_json_bytes(record) + b"\n" for record in external_records
            )
        ).hexdigest()
        if external_tree != payload.get(tree_key):
            raise HarrisZPlusReportingError(f"{label} aggregate identity mismatch.")


def _diagnostics(row: Mapping[str, str], column: str) -> dict[str, Any]:
    raw = row.get(column, "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarrisZPlusReportingError(f"Invalid diagnostics JSON in {column}.") from exc
    if not isinstance(value, dict):
        raise HarrisZPlusReportingError(f"Diagnostics in {column} must be an object.")
    return value


def _column_numbers(rows: Sequence[Mapping[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row.get(column) not in (None, "")]


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _numeric_summary(values: Iterable[float | None]) -> dict[str, Any]:
    numbers = sorted(_finite(values))
    if not numbers:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    index = min(len(numbers) - 1, math.ceil(0.95 * len(numbers)) - 1)
    return {
        "count": len(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p95": numbers[index],
    }


def _percentage(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue().encode("utf-8")


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _dataset_label(dataset: str) -> str:
    return "SD300B" if dataset == "sd300b" else "SD300C"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.9f}"
