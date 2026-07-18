from __future__ import annotations

import json
from pathlib import Path

import pytest

from fingerprint_benchmark.bundle import create_candidate_directory, discard_candidate_directory
from fingerprint_benchmark.manifest import MANIFEST_COLUMNS, PairRecord
from fingerprint_benchmark.sift_derived_protocol import (
    EXPECTED_CONFIG_FILE_SHA256,
    EXPECTED_DECISION_FILE_SHA256,
    INCLUDED_COLUMNS,
    PrimaryBundle,
    SiftDerivedProtocolError,
    _publish_immutable_candidate,
    compare_derived_to_primary,
    compare_result_rows,
    decision_summary,
    deterministic_sample_indices,
    filter_manifest_bytes,
    frozen_decision,
    select_dataset_identities,
    validate_exact_manifest_subset,
    validate_frozen_artifacts,
)


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
        path_a=Path(fr"C:\data\plain\{subject}_plain.png"),
        path_b=Path(fr"C:\data\roll\{subject}_roll.png"),
    )


def _diagnostics(*, inliers: int = 8, candidate: int = 10, ratio: float = 0.8) -> str:
    return json.dumps(
        {
            "keypoint_count_a": 100,
            "keypoint_count_b": 90,
            "matches_submitted_to_geometry": candidate,
            "geometric_inlier_count": inliers,
            "inlier_ratio": ratio,
            "geometry_success": True,
            "geometry_failure_reason": None,
            "residual_destination_pixels": {"mean": 0.2, "median": 0.1, "p95": 0.4},
            "residual_reference_pixels": {"mean": 0.2, "median": 0.1, "p95": 0.4},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _row(pair: PairRecord, score: str, status: str = "ok", *, diagnostics: str | None = None):
    return {
        "pair_id": pair.pair_id,
        "dataset": pair.dataset,
        "protocol": pair.protocol,
        "subject_id": pair.subject_id,
        "canonical_finger_position": str(pair.canonical_finger_position),
        "raw_score": score if status == "ok" else "",
        "status": status,
        "error_code": "" if status == "ok" else "synthetic_failure",
        "prepare_a_ms": "1",
        "prepare_b_ms": "1",
        "compare_ms": "1",
        "method_compare_ms": "0.5",
        "total_ms": "3",
        "compare_diagnostics": diagnostics if diagnostics is not None else _diagnostics(),
    }


def _bundle(dataset: str, protocol: str, scores):
    pairs = [_pair(dataset, protocol, subject, finger) for subject, finger in scores]
    rows = [_row(pair, *scores[(pair.subject_id, pair.canonical_finger_position)]) for pair in pairs]
    return PrimaryBundle(dataset, protocol, Path("manifest.csv"), Path("bundle"), pairs, rows, {})


def _selection(dataset: str, plain_scores, roll_scores, pair_scores):
    return select_dataset_identities(
        dataset,
        _bundle(dataset, "plain_self", plain_scores),
        _bundle(dataset, "roll_self", roll_scores),
        _bundle(dataset, "plain_roll", pair_scores),
        threshold=4.0,
    )


def test_filtering_is_independent_for_b_and_c():
    identities = {("x", 1): ("100", "ok"), ("y", 2): ("100", "ok")}
    b = _selection("sd300b", identities, identities, identities)
    c = _selection(
        "sd300c",
        {("x", 1): ("100", "ok"), ("y", 2): ("3", "ok")},
        identities,
        identities,
    )
    assert set(b.included_identities) == {("x", 1), ("y", 2)}
    assert set(c.included_identities) == {("x", 1)}


def test_frozen_decision_is_inclusive_at_threshold():
    pair = _pair("sd300b", "plain_self", "x", 1)
    assert frozen_decision(_row(pair, "4.0"), 4.0) == "accepted"
    assert frozen_decision(_row(pair, "3.999"), 4.0) == "rejected"


def test_non_ok_self_row_is_rejected_without_reading_score():
    pair = _pair("sd300b", "plain_self", "x", 1)
    assert frozen_decision(_row(pair, "", "prepare_a_failure"), 4.0) == "rejected"


def test_plain_self_rejection_excludes_identity():
    selection = _selection(
        "sd300b",
        {("x", 1): ("3", "ok")},
        {("x", 1): ("5", "ok")},
        {("x", 1): ("999", "ok")},
    )
    assert not selection.included_identities
    assert selection.excluded_rows[0]["reason_flags"] == "plain_self_rejected"


def test_roll_self_rejection_excludes_identity():
    selection = _selection(
        "sd300b",
        {("x", 1): ("5", "ok")},
        {("x", 1): ("3", "ok")},
        {("x", 1): ("999", "ok")},
    )
    assert selection.excluded_rows[0]["reason_flags"] == "roll_self_rejected"


def test_non_ok_flags_are_distinct_from_threshold_rejection():
    selection = _selection(
        "sd300b",
        {("x", 1): ("", "prepare_a_failure")},
        {("x", 1): ("5", "ok")},
        {("x", 1): ("999", "ok")},
    )
    assert selection.excluded_rows[0]["reason_flags"] == "plain_self_non_ok"


def test_plain_roll_score_and_decision_never_affect_inclusion():
    plain = {("x", 1): ("5", "ok")}
    roll = {("x", 1): ("5", "ok")}
    rejected = _selection("sd300b", plain, roll, {("x", 1): ("0", "ok")})
    accepted = _selection("sd300b", plain, roll, {("x", 1): ("500", "ok")})
    assert rejected.included_identities == accepted.included_identities == (("x", 1),)


def test_missing_counterparts_collect_all_applicable_flags():
    selection = _selection(
        "sd300b",
        {("plain", 1): ("5", "ok"), ("no_pair", 2): ("5", "ok")},
        {("roll", 3): ("5", "ok"), ("no_pair", 2): ("5", "ok")},
        {("plain", 1): ("0", "ok"), ("roll", 3): ("0", "ok")},
    )
    reasons = {row["subject_id"]: set(row["reason_flags"].split(";")) for row in selection.excluded_rows}
    assert reasons["plain"] == {"missing_roll_self_identity"}
    assert reasons["roll"] == {"missing_plain_self_identity"}
    assert reasons["no_pair"] == {"missing_plain_roll_pair"}


def test_inclusion_provenance_has_exact_schema_and_self_pair_ids():
    selection = _selection(
        "sd300b",
        {("x", 1): ("5", "ok")},
        {("x", 1): ("6", "ok")},
        {("x", 1): ("0", "ok")},
    )
    row = selection.included_rows[0]
    assert list(row) == INCLUDED_COLUMNS
    assert row["plain_self_pair_id"].endswith("plain_self_x_01")
    assert row["roll_self_pair_id"].endswith("roll_self_x_01")
    assert row["plain_self_candidate_match_count"] == "10"


def _manifest_bytes(dataset: str = "sd300b"):
    ppi = "1000" if dataset == "sd300b" else "2000"
    header = (",".join(MANIFEST_COLUMNS) + "\r\n").encode()
    row1 = (
        f"{dataset}_plain_roll_a_01,{dataset},plain_roll,a,1,{ppi},1,1,"
        f"C:\\data\\plain\\a_plain.png,C:\\data\\roll\\a_roll.png\r\n"
    ).encode()
    row2 = (
        f"{dataset}_plain_roll_b_02,{dataset},plain_roll,b,2,{ppi},2,2,"
        f"C:\\data\\plain\\b_plain.png,C:\\data\\roll\\b_roll.png\r\n"
    ).encode()
    return header, row1, row2


def test_exact_base_row_preservation(tmp_path):
    header, row1, row2 = _manifest_bytes()
    base = tmp_path / "base.csv"
    derived = tmp_path / "derived.csv"
    base.write_bytes(header + row1 + row2)
    derived.write_bytes(filter_manifest_bytes(base, {"sd300b_plain_roll_b_02"}))
    assert derived.read_bytes() == header + row2
    report = validate_exact_manifest_subset(
        derived,
        base,
        expected_dataset="sd300b",
        expected_pair_ids=["sd300b_plain_roll_b_02"],
    )
    assert report["source_rows_byte_exact"] is True


def test_derived_manifest_is_deterministic(tmp_path):
    header, row1, row2 = _manifest_bytes()
    base = tmp_path / "base.csv"
    base.write_bytes(header + row1 + row2)
    selected = {"sd300b_plain_roll_a_01"}
    assert filter_manifest_bytes(base, selected) == filter_manifest_bytes(base, selected)


def test_eligible_completeness_is_enforced(tmp_path):
    header, row1, row2 = _manifest_bytes()
    base = tmp_path / "base.csv"
    derived = tmp_path / "derived.csv"
    base.write_bytes(header + row1 + row2)
    derived.write_bytes(header + row1)
    with pytest.raises(SiftDerivedProtocolError, match="incomplete"):
        validate_exact_manifest_subset(
            derived,
            base,
            expected_dataset="sd300b",
            expected_pair_ids=["sd300b_plain_roll_a_01", "sd300b_plain_roll_b_02"],
        )


def test_wrong_dataset_is_rejected():
    b = _bundle("sd300b", "plain_self", {("x", 1): ("5", "ok")})
    c = _bundle("sd300c", "roll_self", {("x", 1): ("5", "ok")})
    pair = _bundle("sd300b", "plain_roll", {("x", 1): ("0", "ok")})
    with pytest.raises(SiftDerivedProtocolError, match="Wrong bundle"):
        select_dataset_identities("sd300b", b, c, pair, threshold=4.0)


def test_uniform_sample_indexes_cover_manifest_range():
    indexes = deterministic_sample_indices(8593)
    assert len(indexes) == len(set(indexes)) == 30
    assert indexes[0] == 0
    assert indexes[-1] == 8306


def test_uniform_sample_requires_thirty_available_rows():
    with pytest.raises(SiftDerivedProtocolError):
        deterministic_sample_indices(29)


def test_rerun_alignment_is_by_pair_id_not_primary_row_order():
    first = _row(_pair("sd300b", "plain_roll", "a", 1), "8")
    second = _row(_pair("sd300b", "plain_roll", "b", 2), "9")
    details, summary = compare_derived_to_primary(
        [second, first], [first, second], dataset="sd300b", threshold=4.0
    )
    assert [row["pair_id"] for row in details] == [first["pair_id"], second["pair_id"]]
    assert summary["passed"] is True


def test_raw_score_reproducibility_requires_zero_delta():
    pair = _pair("sd300b", "plain_roll", "a", 1)
    result = compare_result_rows(_row(pair, "8.0"), _row(pair, "8.0001"), threshold=4.0)
    assert result["passed"] is False
    assert result["absolute_score_delta"] == pytest.approx(0.0001)


def test_decision_reproducibility_is_checked():
    pair = _pair("sd300b", "plain_roll", "a", 1)
    result = compare_result_rows(_row(pair, "3.0"), _row(pair, "5.0"), threshold=4.0)
    assert result["decision_equal"] is False


def test_deterministic_diagnostic_reproducibility_is_checked():
    pair = _pair("sd300b", "plain_roll", "a", 1)
    result = compare_result_rows(
        _row(pair, "8", diagnostics=_diagnostics(inliers=8)),
        _row(pair, "8", diagnostics=_diagnostics(inliers=7)),
        threshold=4.0,
    )
    assert result["diagnostics_equal"] is False


def test_full_comparison_checks_residual_summaries():
    pair = _pair("sd300b", "plain_roll", "a", 1)
    altered = json.loads(_diagnostics())
    altered["residual_reference_pixels"]["median"] = 0.2
    result = compare_result_rows(
        _row(pair, "8"),
        _row(pair, "8", diagnostics=json.dumps(altered, sort_keys=True, separators=(",", ":"))),
        threshold=4.0,
        include_residuals=True,
    )
    assert result["diagnostics_equal"] is False


def test_rejected_plain_roll_pairs_remain_in_decision_summary():
    accepted = _row(_pair("sd300b", "plain_roll", "a", 1), "4")
    rejected = _row(_pair("sd300b", "plain_roll", "b", 2), "0")
    summary = decision_summary([accepted, rejected], dataset="sd300b", threshold=4.0)
    assert summary["total_pairs"] == 2
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1


def test_atomic_publication_never_overwrites_different_content(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    (final / "artifact.txt").write_text("original", encoding="utf-8")
    candidate = create_candidate_directory(final)
    (candidate / "artifact.txt").write_text("different", encoding="utf-8")
    try:
        with pytest.raises(Exception, match="different content"):
            _publish_immutable_candidate(candidate, final)
        assert (final / "artifact.txt").read_text(encoding="utf-8") == "original"
    finally:
        discard_candidate_directory(candidate)


def test_actual_frozen_hashes_and_decision_rule_validate():
    project_root = Path(__file__).resolve().parents[1]
    frozen = validate_frozen_artifacts(project_root)
    assert frozen["config_file_sha256"] == EXPECTED_CONFIG_FILE_SHA256
    assert frozen["decision_rule_file_sha256"] == EXPECTED_DECISION_FILE_SHA256
    assert frozen["thresholds"] == {"sd300b": 4.0, "sd300c": 4.0}


def test_changed_decision_artifact_is_rejected(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    changed = tmp_path / "decision_rule.json"
    changed.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SiftDerivedProtocolError, match="decision-rule hash mismatch"):
        validate_frozen_artifacts(
            project_root,
            config_path=project_root / "results/sift_geometric/development/sift_geometric_config.json",
            decision_path=changed,
        )
