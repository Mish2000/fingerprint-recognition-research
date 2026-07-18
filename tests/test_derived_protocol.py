from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fingerprint_benchmark.bundle import create_candidate_directory, discard_candidate_directory
from fingerprint_benchmark.derived_protocol import (
    PROTOCOL_NAMESPACE,
    SOURCEAFIS_THRESHOLD,
    DatasetSelection,
    DerivedProtocolError,
    PrimaryBundle,
    _publish_immutable_candidate,
    compare_derived_to_primary,
    decision_summary,
    derived_implementation_compatibility,
    filter_manifest_bytes,
    finalize_integrity,
    select_dataset_identities,
    validate_exact_manifest_subset,
)
from fingerprint_benchmark.manifest import MANIFEST_COLUMNS, PairRecord


def _pair(dataset: str, protocol: str, subject: str, finger: int) -> PairRecord:
    return PairRecord(
        pair_id=f"{dataset}_{protocol}_{subject}_{finger:02d}",
        dataset=dataset,
        protocol=protocol,
        subject_id=subject,
        canonical_finger_position=finger,
        ppi=1000 if dataset == "sd300b" else 2000,
        raw_frgp_a=finger,
        raw_frgp_b=finger,
        path_a=Path(f"{subject}_plain.png"),
        path_b=Path(f"{subject}_roll.png"),
    )


def _row(pair: PairRecord, score: str, status: str = "ok") -> dict[str, str]:
    return {
        "pair_id": pair.pair_id,
        "dataset": pair.dataset,
        "protocol": pair.protocol,
        "subject_id": pair.subject_id,
        "canonical_finger_position": str(pair.canonical_finger_position),
        "raw_score": score if status == "ok" else "",
        "status": status,
        "error_code": "" if status == "ok" else "synthetic_failure",
        "prepare_a_diagnostics": "{}",
        "prepare_b_diagnostics": "{}",
        "compare_diagnostics": "{}",
        "method_compare_ms": "1.0" if status == "ok" else "",
        "total_ms": "3.0",
    }


def _bundle(dataset: str, protocol: str, scores: dict[tuple[str, int], tuple[str, str]]) -> PrimaryBundle:
    pairs = [_pair(dataset, protocol, subject, finger) for subject, finger in scores]
    rows = [_row(pair, *scores[(pair.subject_id, pair.canonical_finger_position)]) for pair in pairs]
    return PrimaryBundle(dataset, protocol, Path("manifest.csv"), Path("bundle"), pairs, rows, {})


def _selection(dataset: str, plain_scores, roll_scores, pair_identities) -> DatasetSelection:
    plain = _bundle(dataset, "plain_self", plain_scores)
    roll = _bundle(dataset, "roll_self", roll_scores)
    base = _bundle(dataset, "plain_roll", {identity: ("0", "ok") for identity in pair_identities})
    return select_dataset_identities(dataset, plain, roll, base)


def test_per_dataset_filtering_does_not_require_b_and_c_to_match():
    identities = {("0001", 1), ("0002", 2)}
    b = _selection(
        "sd300b",
        {("0001", 1): ("40", "ok"), ("0002", 2): ("41", "ok")},
        {("0001", 1): ("40", "ok"), ("0002", 2): ("41", "ok")},
        identities,
    )
    c = _selection(
        "sd300c",
        {("0001", 1): ("40", "ok"), ("0002", 2): ("39", "ok")},
        {("0001", 1): ("40", "ok"), ("0002", 2): ("41", "ok")},
        identities,
    )
    assert set(b.included_identities) == {("0001", 1), ("0002", 2)}
    assert set(c.included_identities) == {("0001", 1)}


def test_self_threshold_is_inclusive_and_non_ok_is_excluded():
    selection = _selection(
        "sd300b",
        {
            ("at", 1): ("40.0", "ok"),
            ("below", 2): ("39.999", "ok"),
            ("nonok", 3): ("", "comparison_failure"),
        },
        {
            ("at", 1): ("40.0", "ok"),
            ("below", 2): ("100", "ok"),
            ("nonok", 3): ("100", "ok"),
        },
        {("at", 1), ("below", 2), ("nonok", 3)},
    )
    assert selection.included_identities == (("at", 1),)
    reasons = {row["subject_id"]: row["reason_flags"] for row in selection.excluded_rows}
    assert reasons["below"] == "plain_self_below_40"
    assert reasons["nonok"] == "plain_self_non_ok"


def test_plain_roll_score_never_affects_inclusion():
    plain = _bundle("sd300b", "plain_self", {("x", 1): ("50", "ok")})
    roll = _bundle("sd300b", "roll_self", {("x", 1): ("60", "ok")})
    rejected_plain_roll = _bundle("sd300b", "plain_roll", {("x", 1): ("0", "ok")})
    accepted_plain_roll = _bundle("sd300b", "plain_roll", {("x", 1): ("999", "ok")})
    assert select_dataset_identities("sd300b", plain, roll, rejected_plain_roll).included_identities == (("x", 1),)
    assert select_dataset_identities("sd300b", plain, roll, accepted_plain_roll).included_identities == (("x", 1),)


def test_missing_counterparts_are_excluded_with_all_applicable_flags():
    selection = _selection(
        "sd300b",
        {("plain_only", 1): ("50", "ok"), ("no_pair", 2): ("50", "ok")},
        {("roll_only", 3): ("50", "ok"), ("no_pair", 2): ("50", "ok")},
        {("plain_only", 1), ("roll_only", 3)},
    )
    reasons = {row["subject_id"]: set(row["reason_flags"].split(";")) for row in selection.excluded_rows}
    assert reasons["plain_only"] == {"missing_roll_self_identity"}
    assert reasons["roll_only"] == {"missing_plain_self_identity"}
    assert reasons["no_pair"] == {"missing_plain_roll_pair"}


def _manifest_bytes(dataset: str = "sd300b") -> tuple[bytes, bytes, bytes]:
    header = (",".join(MANIFEST_COLUMNS) + "\r\n").encode()
    ppi = "1000" if dataset == "sd300b" else "2000"
    row1 = f"{dataset}_plain_roll_0001_01,{dataset},plain_roll,0001,1,{ppi},1,1,C:\\data\\0001_plain.png,C:\\data\\0001_roll.png\r\n".encode()
    row2 = f"{dataset}_plain_roll_0002_02,{dataset},plain_roll,0002,2,{ppi},2,2,C:\\data\\0002_plain.png,C:\\data\\0002_roll.png\r\n".encode()
    return header, row1, row2


def test_exact_base_row_preservation_and_source_order(tmp_path):
    header, row1, row2 = _manifest_bytes()
    base = tmp_path / "base.csv"
    derived = tmp_path / "derived.csv"
    base.write_bytes(header + row1 + row2)
    derived.write_bytes(filter_manifest_bytes(base, {"sd300b_plain_roll_0002_02"}))
    assert derived.read_bytes() == header + row2
    report = validate_exact_manifest_subset(
        derived,
        base,
        expected_dataset="sd300b",
        expected_pair_ids=["sd300b_plain_roll_0002_02"],
    )
    assert report["source_rows_byte_exact"] is True


def test_derived_manifest_generation_is_deterministic(tmp_path):
    header, row1, row2 = _manifest_bytes()
    base = tmp_path / "base.csv"
    base.write_bytes(header + row1 + row2)
    selected = {"sd300b_plain_roll_0001_01"}
    assert filter_manifest_bytes(base, selected) == filter_manifest_bytes(base, selected)


def test_derived_manifest_completeness_is_enforced(tmp_path):
    header, row1, row2 = _manifest_bytes()
    base = tmp_path / "base.csv"
    derived = tmp_path / "derived.csv"
    base.write_bytes(header + row1 + row2)
    derived.write_bytes(header + row1)
    with pytest.raises(DerivedProtocolError, match="incomplete"):
        validate_exact_manifest_subset(
            derived,
            base,
            expected_dataset="sd300b",
            expected_pair_ids=["sd300b_plain_roll_0001_01", "sd300b_plain_roll_0002_02"],
        )


def test_wrong_dataset_is_rejected():
    b = _bundle("sd300b", "plain_self", {("x", 1): ("50", "ok")})
    c_roll = _bundle("sd300c", "roll_self", {("x", 1): ("50", "ok")})
    b_plain_roll = _bundle("sd300b", "plain_roll", {("x", 1): ("0", "ok")})
    with pytest.raises(DerivedProtocolError, match="Wrong bundle"):
        select_dataset_identities("sd300b", b, c_roll, b_plain_roll)


def test_rerun_alignment_is_by_pair_id_not_row_order():
    pair1 = _pair("sd300b", "plain_roll", "a", 1)
    pair2 = _pair("sd300b", "plain_roll", "b", 2)
    primary = [_row(pair2, "20"), _row(pair1, "10")]
    derived = [_row(pair1, "10"), _row(pair2, "20")]
    details, summary = compare_derived_to_primary(primary, derived, dataset="sd300b")
    assert [row["pair_id"] for row in details] == [pair1.pair_id, pair2.pair_id]
    assert summary["passed"] is True


def test_exact_score_reproducibility_detects_any_delta():
    pair = _pair("sd300b", "plain_roll", "a", 1)
    _, summary = compare_derived_to_primary([_row(pair, "10.0")], [_row(pair, "10.5")], dataset="sd300b")
    assert summary["passed"] is False
    assert summary["max_absolute_score_delta"] == 0.5


def test_atomic_publication_never_overwrites_different_existing_content(tmp_path):
    final = tmp_path / "published"
    final.mkdir()
    (final / "artifact.txt").write_text("original", encoding="utf-8")
    identical = create_candidate_directory(final)
    (identical / "artifact.txt").write_text("original", encoding="utf-8")
    _publish_immutable_candidate(identical, final)
    assert not identical.exists()
    different = create_candidate_directory(final)
    (different / "artifact.txt").write_text("different", encoding="utf-8")
    try:
        with pytest.raises(Exception, match="different content"):
            _publish_immutable_candidate(different, final)
        assert (final / "artifact.txt").read_text(encoding="utf-8") == "original"
    finally:
        discard_candidate_directory(different)


def _snapshot(path: Path, tree_hash: str) -> None:
    path.write_text(
        json.dumps({"header": {"schema_version": "protected-artifact-snapshot-v1"}})
        + "\n"
        + json.dumps(
            {
                "footer": {
                    "file_count": 2,
                    "total_bytes": 3,
                    "tree_sha256": tree_hash,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_protected_artifact_before_after_hashes_must_match(tmp_path):
    protocol_root = tmp_path / "protocol"
    run_root = tmp_path / "run"
    protocol_root.mkdir()
    run_root.mkdir()
    (protocol_root / "x").write_text("x", encoding="utf-8")
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    _snapshot(before, "abc")
    _snapshot(after, "abc")
    report = finalize_integrity(
        project_root=tmp_path,
        protocol_root=protocol_root,
        run_root=run_root,
        before_snapshot=before,
        after_snapshot=after,
    )
    assert report["protected_artifacts_unchanged"] is True


def test_future_protocol_specification_matches_implementation():
    project_root = Path(__file__).resolve().parents[1]
    text = (project_root / "docs" / "per_method_self_accept_then_plain_roll_protocol.md").read_text(encoding="utf-8")
    assert PROTOCOL_NAMESPACE in text
    assert str(SOURCEAFIS_THRESHOLD) in text
    assert "(subject_id, canonical_finger_position)" in text
    assert "plain_roll` score" in text
    assert "Never remove" in text


def test_decision_summary_retains_rejected_plain_roll_pairs():
    accepted = _row(_pair("sd300b", "plain_roll", "a", 1), "40")
    rejected = _row(_pair("sd300b", "plain_roll", "b", 2), "0")
    summary = decision_summary([accepted, rejected], dataset="sd300b")
    assert summary["total_pairs"] == 2
    assert summary["genuine_accepts"] == 1
    assert summary["false_non_matches"] == 1
    assert summary["score_zero"] == 1


def test_implementation_policy_allows_only_jar_and_metadata_provenance_variation():
    primary = {
        "sidecar_jar_sha256": "old-jar",
        "benchmark_runner_source_sha256": "runner",
        "benchmark_support_source_sha256": {"io.py": "io", "provenance.py": "old-provenance"},
    }
    current = {
        "sidecar_jar_sha256": "new-jar",
        "benchmark_runner_source_sha256": "runner",
        "benchmark_support_source_sha256": {"io.py": "io", "provenance.py": "new-provenance"},
    }
    accepted = derived_implementation_compatibility(
        primary_hash="old", current_hash="new", primary_components=primary, current_components=current
    )
    assert accepted["accepted"] is True
    current["benchmark_runner_source_sha256"] = "changed-runner"
    rejected = derived_implementation_compatibility(
        primary_hash="old", current_hash="new", primary_components=primary, current_components=current
    )
    assert rejected["accepted"] is False
