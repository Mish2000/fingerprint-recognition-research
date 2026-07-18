from __future__ import annotations

import json
from pathlib import Path

import pytest

from fingerprint_benchmark.manifest import PairRecord
from fingerprint_benchmark.shared_accuracy_integrity import capture_snapshot, compare_snapshot
from fingerprint_benchmark.sourceafis_negative_protocol import (
    DATASETS,
    NEGATIVE_MANIFEST_COLUMNS,
    NEGATIVE_PROTOCOL,
    REPORT_SECTION_TITLES,
    SOURCEAFIS_THRESHOLD,
    NegativeProtocolError,
    _csv_bytes,
    benchmark_projection,
    build_next_subject_rows,
    build_report_sections,
    compare_overlap_rows,
    lookup_exact_reuse,
    negative_decision_summary,
    validate_next_subject_rows,
)


def _pair(dataset: str, subject: str, finger: int) -> PairRecord:
    ppi = 1000 if dataset == "sd300b" else 2000
    return PairRecord(
        pair_id=f"{dataset}_plain_roll_{subject}_{finger:02d}",
        dataset=dataset,
        protocol="plain_roll",
        subject_id=subject,
        canonical_finger_position=finger,
        ppi=ppi,
        raw_frgp_a=finger + 10 if finger in {1, 6} else finger,
        raw_frgp_b=finger,
        path_a=Path(f"C:/data/{subject}_plain_{finger:02d}.png"),
        path_b=Path(f"C:/data/{subject}_roll_{finger:02d}.png"),
    )


def _population(dataset: str = "sd300b") -> list[PairRecord]:
    return [
        _pair(dataset, "0003", 1),
        _pair(dataset, "0001", 1),
        _pair(dataset, "0002", 1),
        _pair(dataset, "0002", 2),
        _pair(dataset, "0001", 2),
    ]


def _result_row(pair_id: str, score: str, status: str = "ok") -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "raw_score": score if status == "ok" else "",
        "status": status,
        "error_code": "" if status == "ok" else "synthetic_failure",
        "prepare_a_diagnostics": "{}",
        "prepare_b_diagnostics": "{}",
        "compare_diagnostics": "{}",
        "method_compare_ms": "2.0" if status == "ok" else "",
        "total_ms": "5.0",
    }


def test_only_single_finger_eligible_inputs_are_accepted():
    pairs = _population()
    pairs.append(_pair("sd300b", "0001", 1))
    with pytest.raises(NegativeProtocolError, match="Multiple captures"):
        build_next_subject_rows("sd300b", pairs)


def test_datasets_are_constructed_independently():
    b = build_next_subject_rows("sd300b", _population("sd300b"))
    c = build_next_subject_rows("sd300c", _population("sd300c")[:3])
    assert {row["dataset"] for row in b} == {"sd300b"}
    assert {row["dataset"] for row in c} == {"sd300c"}
    assert len(b) == 5
    assert len(c) == 3


def test_grouping_is_by_canonical_finger_position():
    rows = build_next_subject_rows("sd300b", _population())
    assert [row["canonical_finger_position"] for row in rows] == [1, 1, 1, 2, 2]
    assert all(row["source_plain_pair_id"].endswith(f"_{row['canonical_finger_position']:02d}") for row in rows)
    assert all(row["source_roll_pair_id"].endswith(f"_{row['canonical_finger_position']:02d}") for row in rows)


def test_subject_sort_is_lexicographically_stable():
    rows = build_next_subject_rows("sd300b", _population())
    assert [row["subject_id_a"] for row in rows[:3]] == ["0001", "0002", "0003"]
    assert [row["plain_group_index"] for row in rows[:3]] == [0, 1, 2]


def test_shift_is_exactly_one():
    rows = build_next_subject_rows("sd300b", _population())
    assert all(row["shift"] == 1 for row in rows)
    assert [(row["plain_group_index"], row["roll_group_index"]) for row in rows[:3]] == [
        (0, 1),
        (1, 2),
        (2, 0),
    ]


def test_circular_wrap_around_uses_first_roll():
    rows = build_next_subject_rows("sd300b", _population())
    assert rows[2]["subject_id_a"] == "0003"
    assert rows[2]["subject_id_b"] == "0001"
    assert rows[2]["roll_group_index"] == 0


def test_every_negative_pair_has_different_subjects():
    rows = build_next_subject_rows("sd300b", _population())
    assert all(row["subject_id_a"] != row["subject_id_b"] for row in rows)


def test_every_negative_pair_has_same_canonical_finger():
    pairs = _population()
    rows = build_next_subject_rows("sd300b", pairs)
    by_id = {pair.pair_id: pair for pair in pairs}
    assert all(
        by_id[row["source_plain_pair_id"]].canonical_finger_position
        == by_id[row["source_roll_pair_id"]].canonical_finger_position
        == row["canonical_finger_position"]
        for row in rows
    )


def test_every_plain_is_used_once():
    pairs = _population()
    rows = build_next_subject_rows("sd300b", pairs)
    assert sorted(row["source_plain_pair_id"] for row in rows) == sorted(pair.pair_id for pair in pairs)


def test_every_roll_is_used_once():
    pairs = _population()
    rows = build_next_subject_rows("sd300b", pairs)
    assert sorted(row["source_roll_pair_id"] for row in rows) == sorted(pair.pair_id for pair in pairs)


def test_negative_count_equals_derived_genuine_count():
    pairs = _population()
    rows = build_next_subject_rows("sd300b", pairs)
    report = validate_next_subject_rows("sd300b", rows, pairs)
    assert report["pair_count"] == report["source_genuine_pair_count"] == len(pairs)


def test_no_genuine_contamination_or_same_image():
    pairs = _population()
    report = validate_next_subject_rows(
        "sd300b", build_next_subject_rows("sd300b", pairs), pairs
    )
    assert report["genuine_contamination_count"] == 0
    assert report["same_image_both_sides_count"] == 0


def test_negative_manifest_is_deterministic():
    first = build_next_subject_rows("sd300b", _population())
    second = build_next_subject_rows("sd300b", list(reversed(_population())))
    assert _csv_bytes(first, NEGATIVE_MANIFEST_COLUMNS) == _csv_bytes(second, NEGATIVE_MANIFEST_COLUMNS)


def test_frozen_benchmark_projection_is_exact():
    rows = build_next_subject_rows("sd300b", _population())
    projected = benchmark_projection(rows)
    assert list(projected[0]) == [
        "pair_id",
        "dataset",
        "protocol",
        "subject_id",
        "canonical_finger_position",
        "ppi",
        "raw_frgp_a",
        "raw_frgp_b",
        "path_a",
        "path_b",
    ]
    assert projected[0]["pair_id"] == rows[0]["negative_pair_id"]
    assert projected[0]["protocol"] == NEGATIVE_PROTOCOL


def test_threshold_40_decision_is_inclusive_and_failures_are_separate():
    rows = [
        _result_row("at", str(SOURCEAFIS_THRESHOLD)),
        _result_row("below", "39.999"),
        _result_row("zero", "0.0"),
        _result_row("failure", "", "comparison_failure"),
    ]
    summary = negative_decision_summary(rows, dataset="sd300b")
    assert summary["false_matches"] == 1
    assert summary["correct_non_matches"] == 2
    assert summary["failures"] == 1
    assert summary["score_zero"] == 1
    assert summary["positive_below_threshold"] == 1


def test_exact_reuse_lookup_uses_dataset_ppi_paths_subjects_and_finger():
    negative = build_next_subject_rows("sd300b", _population())[:1]
    shared = [
        {
            **negative[0],
            "split": "evaluation",
            "accuracy_pair_id": "shared-1",
            "raw_score": "0.0",
            "status": "ok",
            "error_code": "",
            "prepare_a_diagnostics_json": "{}",
            "prepare_b_diagnostics_json": "{}",
            "compare_diagnostics_json": "{}",
            "shared_score_file": "scores.csv",
        }
    ]
    overlap, missing = lookup_exact_reuse(negative, shared)
    assert len(overlap) == 1
    assert missing == 0
    changed = [dict(shared[0], subject_id_b="wrong")]
    overlap, missing = lookup_exact_reuse(negative, changed)
    assert overlap == []
    assert missing == 1


def test_exact_score_reproducibility_checks_status_errors_and_diagnostics():
    overlap = [
        {
            "dataset": "sd300b",
            "negative_pair_id": "negative-1",
            "shared_accuracy_pair_id": "shared-1",
            "raw_score": "2.5",
            "status": "ok",
            "error_code": "",
            "prepare_a_diagnostics_json": "{}",
            "prepare_b_diagnostics_json": "{}",
            "compare_diagnostics_json": "{}",
        }
    ]
    details, summary = compare_overlap_rows(overlap, [_result_row("negative-1", "2.5")], dataset="sd300b")
    assert details[0]["exact_reproducibility"] is True
    assert summary["mismatch_count"] == 0
    assert summary["max_absolute_score_delta"] == 0.0


def test_exact_score_reproducibility_rejects_any_score_delta():
    overlap = [
        {
            "dataset": "sd300b",
            "negative_pair_id": "negative-1",
            "shared_accuracy_pair_id": "shared-1",
            "raw_score": "2.5",
            "status": "ok",
            "error_code": "",
            "prepare_a_diagnostics_json": "{}",
            "prepare_b_diagnostics_json": "{}",
            "compare_diagnostics_json": "{}",
        }
    ]
    _, summary = compare_overlap_rows(overlap, [_result_row("negative-1", "2.5001")], dataset="sd300b")
    assert summary["passed"] is False
    assert summary["max_absolute_score_delta"] == pytest.approx(0.0001)


def test_source_artifact_snapshot_detects_no_change(tmp_path):
    source = tmp_path / "protected.txt"
    source.write_text("frozen", encoding="utf-8")
    before = tmp_path / "before.jsonl"
    capture_snapshot({source: "source"}, before)
    report, _, _ = compare_snapshot(before, {source: "source"})
    assert report["protected_artifacts_unchanged"] is True
    assert report["mismatch_count"] == 0


def _report_inputs():
    source = {
        "record_counts": {
            dataset: {"plain_single_finger_records": 10, "roll_single_finger_records": 11}
            for dataset in DATASETS
        },
        "self": {
            dataset: {
                "plain_self": {"total": 10, "matches": 9, "removed_before_derived_protocol": 1, "match_percentage": 90.0, "mean_method_compare_ms": 1.0},
                "roll_self": {"total": 11, "matches": 9, "removed_before_derived_protocol": 2, "match_percentage": 81.8, "mean_method_compare_ms": 1.0},
            }
            for dataset in DATASETS
        },
        "genuine": {
            dataset: {
                "total_pairs": 9,
                "same_subject_same_finger_count": 9,
                "wrong_count": 0,
                "matches": 8,
                "non_matches": 1,
                "match_percentage": 88.8,
                "mean_method_compare_ms": 2.0,
            }
            for dataset in DATASETS
        },
    }
    negative = {
        "negative_results": {
            "datasets": [
                {
                    "dataset": dataset,
                    "total_wrong_pairs": 9,
                    "false_matches": 0,
                    "correct_non_matches": 9,
                    "false_match_percentage": 0.0,
                    "mean_method_compare_ms": 2.0,
                }
                for dataset in DATASETS
            ]
        }
    }
    return source, negative


def test_report_contains_only_requested_sections_in_order():
    sections = build_report_sections(*_report_inputs())
    assert tuple(section["title"] for section in sections) == REPORT_SECTION_TITLES
    assert [section["number"] for section in sections] == [1, 2, 3, 4, 5, 6]
    text = json.dumps(sections).casefold()
    for forbidden in ("roc", "det", "auc", "eer", "confidence", "calibrat", "fusion", "ranking"):
        assert forbidden not in text


def test_sift_is_not_invoked_or_reported():
    module_path = Path(__file__).resolve().parents[1] / "src/fingerprint_benchmark/sourceafis_negative_protocol.py"
    source_text = module_path.read_text(encoding="utf-8")
    assert "from .sift" not in source_text
    assert "import fingerprint_benchmark.sift" not in source_text
    sections = build_report_sections(*_report_inputs())
    assert "sift" not in json.dumps(sections).casefold()
