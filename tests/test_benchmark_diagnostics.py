import csv
import json

import pytest

from fingerprint_benchmark.diagnostics import (
    DiagnosticsError,
    compare_v1_v2_scores,
    paired_sd300_diagnostics,
    score_diagnostics,
    write_diagnostics_json,
    write_paired_diagnostics_csv,
    write_v1_v2_comparison_csv,
)


def test_score_diagnostics_counts_zero_scores_and_canonical_distribution():
    rows = [
        _row("pair-c", dataset="sd300b", subject="0002", position="2", score="0"),
        _row("pair-b", dataset="sd300b", subject="0001", position="2", score="3"),
        _row(
            "pair-f",
            dataset="sd300b",
            subject="0003",
            position="3",
            score="",
            status="comparison_failure",
        ),
        _row("pair-a", dataset="sd300b", subject="0001", position="1", score="0.0"),
    ]

    diagnostics = score_diagnostics(rows)

    assert diagnostics == {
        "pair_count": 4,
        "ok_count": 3,
        "failure_count": 1,
        "failure_counts": {"comparison_failure": 1},
        "zero_score_count": 2,
        "positive_score_count": 1,
        "min": 0.0,
        "max": 3.0,
        "mean": 1.0,
        "median": 0.0,
        "zero_score_pair_ids": ["pair-a", "pair-c"],
        "zero_score_canonical_position_distribution": {"1": 1, "2": 1},
    }


def test_score_diagnostics_returns_null_statistics_when_every_pair_failed():
    diagnostics = score_diagnostics(
        [
            _row(
                "failed",
                dataset="sd300b",
                subject="0001",
                position="1",
                score="",
                status="prepare_a_failure",
            )
        ]
    )

    assert diagnostics["min"] is None
    assert diagnostics["max"] is None
    assert diagnostics["mean"] is None
    assert diagnostics["median"] is None
    assert diagnostics["zero_score_count"] == 0
    assert diagnostics["positive_score_count"] == 0


def test_paired_diagnostics_align_by_identity_not_shuffled_row_order():
    b_rows = [
        _row("b-2", dataset="sd300b", subject="0002", position="2", score="10"),
        _row("b-1", dataset="sd300b", subject="0001", position="1", score="0"),
        _row("b-only", dataset="sd300b", subject="0003", position="3", score="5"),
    ]
    # Deliberately put C identity 0002 before 0001. Positional zip alignment is wrong.
    c_rows = [
        _row("c-2", dataset="sd300c", subject="0002", position="2", score="13"),
        _row("c-only", dataset="sd300c", subject="0004", position="4", score="7"),
        _row("c-1", dataset="sd300c", subject="0001", position="1", score="0"),
    ]

    diagnostics = paired_sd300_diagnostics(b_rows, c_rows)

    assert diagnostics["alignment_key"] == [
        "protocol",
        "subject_id",
        "canonical_finger_position",
    ]
    assert diagnostics["shared_identity_count"] == 2
    assert diagnostics["comparable_identity_count"] == 2
    assert diagnostics["sd300b_only_identity_count"] == 1
    assert diagnostics["sd300c_only_identity_count"] == 1
    assert diagnostics["exact_equality_count"] == 1
    assert diagnostics["mean_absolute_delta"] == 1.5
    assert diagnostics["median_absolute_delta"] == 1.5
    assert diagnostics["pearson_correlation"] == pytest.approx(1.0)
    assert diagnostics["zero_overlap_count"] == 1
    assert diagnostics["zero_overlap_identities"] == [
        {
            "protocol": "plain_self",
            "subject_id": "0001",
            "canonical_finger_position": "1",
        }
    ]
    assert [item["subject_id"] for item in diagnostics["identities"]] == ["0001", "0002"]
    assert [item["delta_c_minus_b"] for item in diagnostics["identities"]] == [0.0, 3.0]
    assert diagnostics["identities"][1]["sd300b_pair_id"] == "b-2"
    assert diagnostics["identities"][1]["sd300c_pair_id"] == "c-2"


def test_paired_diagnostics_keeps_shared_failure_but_excludes_it_from_score_metrics():
    b_rows = [
        _row(
            "b-1",
            dataset="sd300b",
            subject="0001",
            position="1",
            score="",
            status="comparison_failure",
        )
    ]
    c_rows = [
        _row("c-1", dataset="sd300c", subject="0001", position="1", score="4")
    ]

    diagnostics = paired_sd300_diagnostics(b_rows, c_rows)

    assert diagnostics["shared_identity_count"] == 1
    assert diagnostics["comparable_identity_count"] == 0
    assert diagnostics["mean_absolute_delta"] is None
    assert diagnostics["pearson_correlation"] is None
    assert diagnostics["identities"][0]["sd300b_score"] is None
    assert diagnostics["identities"][0]["delta_c_minus_b"] is None


def test_paired_diagnostics_rejects_duplicate_identity():
    b_rows = [
        _row("b-1", dataset="sd300b", subject="0001", position="1", score="1"),
        _row("b-duplicate", dataset="sd300b", subject="0001", position="01", score="2"),
    ]

    with pytest.raises(DiagnosticsError, match="Duplicate paired identity"):
        paired_sd300_diagnostics(b_rows, [])


def test_v1_v2_comparison_aligns_by_pair_id_and_classifies_old_precision():
    v1_rows = [
        _row("precision", dataset="sd300b", subject="0001", position="1", score="1.23456789"),
        _row("exact", dataset="sd300b", subject="0002", position="2", score="2.5"),
        _row("beyond", dataset="sd300b", subject="0003", position="3", score="5"),
        _row("v1-only", dataset="sd300b", subject="0004", position="4", score="8"),
    ]
    v2_rows = [
        _row("beyond", dataset="sd300b", subject="0003", position="3", score="5.0001"),
        _row("exact", dataset="sd300b", subject="0002", position="2", score="2.5"),
        _row(
            "precision",
            dataset="sd300b",
            subject="0001",
            position="1",
            score="1.234567891234",
        ),
        _row("v2-only", dataset="sd300b", subject="0005", position="5", score="9"),
    ]

    comparison = compare_v1_v2_scores(v1_rows, v2_rows)

    assert comparison["alignment_key"] == "pair_id"
    assert comparison["v1_raw_score_serialization"] == "format(float_value, '.9g')"
    assert comparison["shared_pair_count"] == 3
    assert comparison["comparable_pair_count"] == 3
    assert comparison["exact_numeric_equality_count"] == 1
    assert comparison["v1_9g_explained_difference_count"] == 1
    assert comparison["beyond_v1_precision_difference_count"] == 1
    assert comparison["beyond_v1_precision_pair_ids"] == ["beyond"]
    assert comparison["v1_only_pair_ids"] == ["v1-only"]
    assert comparison["v2_only_pair_ids"] == ["v2-only"]
    assert [item["pair_id"] for item in comparison["pairs"]] == [
        "beyond",
        "exact",
        "precision",
    ]
    classes = {item["pair_id"]: item["classification"] for item in comparison["pairs"]}
    assert classes == {
        "beyond": "beyond_v1_precision",
        "exact": "exact_numeric_equality",
        "precision": "explained_by_v1_9g",
    }


def test_diagnostics_writers_are_deterministic(tmp_path):
    b_rows = [
        _row("b-2", dataset="sd300b", subject="0002", position="2", score="4"),
        _row("b-1", dataset="sd300b", subject="0001", position="1", score="0"),
    ]
    c_rows = [
        _row("c-1", dataset="sd300c", subject="0001", position="1", score="0"),
        _row("c-2", dataset="sd300c", subject="0002", position="2", score="5"),
    ]
    paired = paired_sd300_diagnostics(b_rows, c_rows)
    comparison = compare_v1_v2_scores(b_rows, list(reversed(b_rows)))

    json_a = tmp_path / "a.json"
    json_b = tmp_path / "b.json"
    write_diagnostics_json(paired, json_a)
    write_diagnostics_json(paired, json_b)
    assert json_a.read_bytes() == json_b.read_bytes()
    assert json.loads(json_a.read_text(encoding="utf-8"))["zero_overlap_count"] == 1

    paired_a = tmp_path / "paired-a.csv"
    paired_b = tmp_path / "paired-b.csv"
    write_paired_diagnostics_csv(paired, paired_a)
    reversed_report = {**paired, "identities": list(reversed(paired["identities"]))}
    write_paired_diagnostics_csv(reversed_report, paired_b)
    assert paired_a.read_bytes() == paired_b.read_bytes()
    with paired_a.open(newline="", encoding="utf-8") as handle:
        assert [row["subject_id"] for row in csv.DictReader(handle)] == ["0001", "0002"]

    comparison_a = tmp_path / "comparison-a.csv"
    comparison_b = tmp_path / "comparison-b.csv"
    write_v1_v2_comparison_csv(comparison, comparison_a)
    reversed_comparison = {**comparison, "pairs": list(reversed(comparison["pairs"]))}
    write_v1_v2_comparison_csv(reversed_comparison, comparison_b)
    assert comparison_a.read_bytes() == comparison_b.read_bytes()


def _row(
    pair_id,
    *,
    dataset,
    subject,
    position,
    score,
    status="ok",
    protocol="plain_self",
):
    return {
        "pair_id": pair_id,
        "dataset": dataset,
        "protocol": protocol,
        "subject_id": subject,
        "canonical_finger_position": position,
        "raw_score": score,
        "status": status,
    }
