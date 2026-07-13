from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fingerprint_benchmark.hashing import file_sha256
from fingerprint_benchmark.manifest import MANIFEST_COLUMNS
from fingerprint_benchmark.reproducibility_audit import (
    ReproducibilityAuditError,
    compare_pair_rows,
    implementation_compatibility,
    make_subset_validator,
    select_protocol_strata,
)


def test_paired_strata_include_all_zeros_and_disjoint_positive_samples():
    b_rows = [
        _result_row("s1", 1, 0),
        _result_row("s2", 1, 0),
        _result_row("s3", 1, 5),
        _result_row("s4", 1, 1),
        _result_row("s5", 1, 3),
        _result_row("s6", 1, 10),
    ]
    c_rows = [
        _result_row("s1", 1, 0),
        _result_row("s2", 1, 5),
        _result_row("s3", 1, 0),
        _result_row("s4", 1, 2),
        _result_row("s5", 1, 4),
        _result_row("s6", 1, 11),
    ]

    first = select_protocol_strata(
        b_rows,
        c_rows,
        protocol="plain_roll",
        seed="fixed-seed",
        low_positive_count=1,
        positive_sample_count=1,
    )
    second = select_protocol_strata(
        b_rows,
        c_rows,
        protocol="plain_roll",
        seed="fixed-seed",
        low_positive_count=1,
        positive_sample_count=1,
    )

    assert first == second
    assert first["s1\x1f1"] == {"plain_roll_zero_both_resolutions"}
    assert first["s2\x1f1"] == {"plain_roll_zero_sd300b_only"}
    assert first["s3\x1f1"] == {"plain_roll_zero_sd300c_only"}
    assert first["s4\x1f1"] == {"low_positive_both_resolutions"}
    assert len(first) == 5
    deterministic = [
        identity
        for identity, labels in first.items()
        if "deterministic_positive_sample" in labels
    ]
    assert len(deterministic) == 1
    assert deterministic[0] not in {"s1\x1f1", "s2\x1f1", "s3\x1f1", "s4\x1f1"}


def test_paired_strata_reject_resolution_identity_mismatch():
    with pytest.raises(ReproducibilityAuditError, match="identity sets differ"):
        select_protocol_strata(
            [_result_row("s1", 1, 1)],
            [_result_row("s2", 1, 1)],
            protocol="plain_self",
            seed="seed",
            low_positive_count=0,
            positive_sample_count=0,
        )


def test_subset_validator_accepts_exact_rows_and_rejects_tampering(tmp_path):
    source = tmp_path / "source.csv"
    subset = tmp_path / "subset.csv"
    tampered = tmp_path / "tampered.csv"
    rows = [
        _manifest_row("pair-1", "subject-1", 1),
        _manifest_row("pair-2", "subject-2", 2),
    ]
    _write_manifest(source, rows)
    _write_manifest(subset, [rows[1]])
    changed = dict(rows[1])
    changed["path_b"] = "different.png"
    _write_manifest(tampered, [changed])

    exact_validator = make_subset_validator(
        source_manifest_path=source,
        expected_dataset="sd300b",
        expected_protocol="plain_roll",
        expected_subset_sha256=file_sha256(subset),
        source_validator=lambda manifest, data_root: {"status": "ok"},
    )
    report = exact_validator(subset, tmp_path)
    assert report["subset_pair_count"] == 1
    assert report["validation_mode"] == "exact_subset_of_fully_validated_source_manifest"

    tampered_validator = make_subset_validator(
        source_manifest_path=source,
        expected_dataset="sd300b",
        expected_protocol="plain_roll",
        expected_subset_sha256=file_sha256(tampered),
        source_validator=lambda manifest, data_root: {"status": "ok"},
    )
    with pytest.raises(ReproducibilityAuditError, match="not an exact source-manifest row"):
        tampered_validator(tampered, tmp_path)


def test_pair_comparison_ignores_timing_changes_but_not_score_changes():
    primary = _comparison_row("10.5", method_compare_ms="2.0", total_ms="20.0")
    rerun = _comparison_row("10.5", method_compare_ms="8.0", total_ms="80.0")

    equal = compare_pair_rows(
        primary,
        rerun,
        strata="deterministic_positive_sample",
        score_abs_tolerance=0.0,
    )
    assert equal["reproducible"] == "true"
    assert equal["raw_score_text_equal"] == "true"
    assert equal["primary_method_compare_ms"] != equal["rerun_method_compare_ms"]

    changed = dict(rerun)
    changed["raw_score"] = "10.6"
    unequal = compare_pair_rows(
        primary,
        changed,
        strata="deterministic_positive_sample",
        score_abs_tolerance=0.0,
    )
    assert unequal["reproducible"] == "false"
    assert unequal["raw_score_within_tolerance"] == "false"

    tolerated = compare_pair_rows(
        primary,
        changed,
        strata="deterministic_positive_sample",
        score_abs_tolerance=0.1,
    )
    assert tolerated["raw_score_within_tolerance"] == "true"
    assert tolerated["reproducible"] == "true"


def test_implementation_policy_allows_only_explicit_jar_hash_variation():
    primary = {
        "benchmark_runner_source_sha256": "runner",
        "python_adapter_source_sha256": "adapter",
        "sidecar_jar_sha256": "old-jar",
    }
    current = {**primary, "sidecar_jar_sha256": "current-jar"}

    strict = implementation_compatibility(
        primary_hash="primary-hash",
        current_hash="current-hash",
        primary_components=primary,
        current_components=current,
        allow_jar_hash_variation=False,
    )
    assert strict["exact_hash_equal"] is False
    assert strict["components_equal_except_sidecar_jar_sha256"] is True
    assert strict["accepted"] is False

    compatible = implementation_compatibility(
        primary_hash="primary-hash",
        current_hash="current-hash",
        primary_components=primary,
        current_components=current,
        allow_jar_hash_variation=True,
    )
    assert compatible["accepted"] is True
    assert compatible["primary_sidecar_jar_sha256"] == "old-jar"
    assert compatible["current_sidecar_jar_sha256"] == "current-jar"

    changed_runner = {**current, "benchmark_runner_source_sha256": "different-runner"}
    rejected = implementation_compatibility(
        primary_hash="primary-hash",
        current_hash="other-hash",
        primary_components=primary,
        current_components=changed_runner,
        allow_jar_hash_variation=True,
    )
    assert rejected["components_equal_except_sidecar_jar_sha256"] is False
    assert rejected["accepted"] is False


def _result_row(subject: str, finger: int, score: float) -> dict[str, str]:
    return {
        "pair_id": f"pair-{subject}-{finger}",
        "subject_id": subject,
        "canonical_finger_position": str(finger),
        "status": "ok",
        "raw_score": repr(float(score)),
    }


def _manifest_row(pair_id: str, subject: str, finger: int) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "dataset": "sd300b",
        "protocol": "plain_roll",
        "subject_id": subject,
        "canonical_finger_position": str(finger),
        "ppi": "1000",
        "raw_frgp_a": "11",
        "raw_frgp_b": "1",
        "path_a": f"{subject}-plain.png",
        "path_b": f"{subject}-roll.png",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _comparison_row(
    raw_score: str,
    *,
    method_compare_ms: str,
    total_ms: str,
) -> dict[str, str]:
    return {
        "dataset": "sd300b",
        "protocol": "plain_roll",
        "pair_id": "pair-1",
        "subject_id": "subject-1",
        "canonical_finger_position": "1",
        "status": "ok",
        "error_code": "",
        "raw_score": raw_score,
        "prepare_a_diagnostics": "{}",
        "prepare_b_diagnostics": "{}",
        "compare_diagnostics": "{}",
        "method_compare_ms": method_compare_ms,
        "compare_ms": method_compare_ms,
        "total_ms": total_ms,
    }
