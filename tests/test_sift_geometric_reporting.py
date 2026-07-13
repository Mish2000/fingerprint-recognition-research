from __future__ import annotations

import csv
from pathlib import Path

from fingerprint_benchmark.manifest import MANIFEST_COLUMNS
from fingerprint_benchmark.sift.reporting import (
    COHORT_NAME,
    _artifact_inventory,
    build_cohort,
    summarize_run,
)


def _result_row(dataset: str, protocol: str, subject: str, finger: int, score: float) -> dict[str, str]:
    return {
        "pair_id": f"{dataset}_{protocol}_{subject}_{finger}",
        "dataset": dataset,
        "protocol": protocol,
        "subject_id": subject,
        "canonical_finger_position": str(finger),
        "raw_score": str(score),
        "prepare_a_ms": "1.0",
        "prepare_b_ms": "1.0",
        "compare_ms": "1.0",
        "total_ms": "3.1",
        "compare_diagnostics": (
            '{"geometry_success":true,"keypoint_count_a":10,"keypoint_count_b":10,'
            '"ratio_match_count_a_to_b":8,"mutual_match_count":6,'
            '"geometric_inlier_count":5,"residual_reference_pixels":{"median":0.5}}'
        ),
        "status": "ok",
        "error_code": "",
    }


def _manifest_row(dataset: str, protocol: str, subject: str, finger: int, image: Path) -> dict[str, str]:
    return {
        "pair_id": f"{dataset}_{protocol}_{subject}_{finger}",
        "dataset": dataset,
        "protocol": protocol,
        "subject_id": subject,
        "canonical_finger_position": str(finger),
        "ppi": "1000" if dataset == "sd300b" else "2000",
        "raw_frgp_a": str(finger),
        "raw_frgp_b": str(finger),
        "path_a": str(image),
        "path_b": str(image),
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_cohort_uses_only_four_self_decisions_and_retains_plain_roll_rejections(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    results_root = tmp_path / "results"
    image = tmp_path / "image.png"
    image.write_bytes(b"x")
    bundles = {}
    for dataset in ("sd300b", "sd300c"):
        for protocol in ("plain_self", "roll_self", "plain_roll"):
            rows = [
                _result_row(dataset, protocol, "00000001", 1, 0.0 if protocol == "plain_roll" else 10.0),
                _result_row(
                    dataset,
                    protocol,
                    "00000002",
                    2,
                    0.0 if (dataset == "sd300c" and protocol == "roll_self") else 10.0,
                ),
            ]
            bundles[(dataset, protocol)] = {
                "rows": rows,
                "metadata": {"result": {"sha256": f"{dataset}-{protocol}"}},
            }
            _write_manifest(
                repo_root / "protocols" / dataset / f"{protocol}.csv",
                [
                    _manifest_row(dataset, protocol, "00000001", 1, image),
                    _manifest_row(dataset, protocol, "00000002", 2, image),
                ],
            )
    decision_rule = {
        "decision_rule_hash": "rule-hash",
        "thresholds_by_dataset": {
            "sd300b": {"primary_threshold": 5.0},
            "sd300c": {"primary_threshold": 5.0},
        },
    }

    cohort = build_cohort(repo_root, results_root, bundles, decision_rule)

    assert cohort["cohort_name"] == COHORT_NAME
    assert cohort["included_identity_keys"] == ["00000001|01"]
    assert cohort["included_identity_count"] == 1
    projected = list(
        csv.DictReader(
            (
                results_root
                / "sift_geometric"
                / "cohorts"
                / COHORT_NAME
                / "sd300b"
                / "plain_roll.csv"
            ).open(newline="", encoding="utf-8")
        )
    )
    assert len(projected) == 1
    assert projected[0]["subject_id"] == "00000001"
    plain_roll_summary = summarize_run(
        "sd300b", "plain_roll", [bundles[("sd300b", "plain_roll")]["rows"][0]], 5.0
    )
    assert plain_roll_summary["accepted_percentage"] == 0.0


def test_self_summary_is_one_hundred_percent_when_all_rows_meet_frozen_rule() -> None:
    rows = [_result_row("sd300b", "plain_self", "00000001", 1, 6.0)]
    summary = summarize_run("sd300b", "plain_self", rows, 5.0)
    assert summary["accepted_percentage"] == 100.0
    assert summary["accepted_count"] == 1


def test_artifact_inventory_excludes_self_referential_reports(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (tmp_path / "stable.csv").write_text("value\n1\n", encoding="utf-8")
    (reports / "artifact_hashes.json").write_text('{"old": true}\n', encoding="utf-8")
    (reports / "report_summary.json").write_text('{"old": true}\n', encoding="utf-8")

    before = _artifact_inventory(tmp_path)
    (reports / "artifact_hashes.json").write_text('{"new": true}\n', encoding="utf-8")
    (reports / "report_summary.json").write_text('{"new": true}\n', encoding="utf-8")
    after = _artifact_inventory(tmp_path)

    assert before == after
    assert set(before["files"]) == {"stable.csv"}
