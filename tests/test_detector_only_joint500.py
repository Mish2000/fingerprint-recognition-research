import csv
import json
from pathlib import Path
import shutil

import pytest

from fingerprint_benchmark.cli import parse_args
from fingerprint_benchmark.detector_only_joint500 import (
    COHORT_SIZE,
    PROTOCOL_DIRECTORY,
    PROTOCOL_NAME,
    Joint500ProtocolError,
    build_protocol_artifacts,
    report_joint500,
    validate_protocol_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_committed_joint500_cohort_is_balanced_unique_neutral_and_byte_exact():
    report = validate_protocol_artifacts(repository_root=REPOSITORY_ROOT)
    assert report["status"] == "ok"
    assert report["cohort_size"] == COHORT_SIZE
    assert report["unique_subject_count"] == COHORT_SIZE
    assert report["per_position_counts"] == {str(position): 50 for position in range(1, 11)}
    assert report["same_identities_b_c"] is True
    assert report["impostor_bijection"] is True
    assert report["same_impostor_logic_b_c"] is True
    assert report["self_filtering"] is False
    assert report["method_score_result_dependency"] is False
    assert build_protocol_artifacts(repository_root=REPOSITORY_ROOT, check=True)["byte_exact"] is True

    metadata = json.loads(
        (REPOSITORY_ROOT / PROTOCOL_DIRECTORY / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["method_references"] == []
    assert metadata["score_references"] == []
    assert metadata["result_references"] == []
    assert "timestamp" not in metadata


def test_joint500_validator_rejects_artifact_and_base_manifest_tampering(tmp_path):
    root = _copy_protocol_inputs(tmp_path)
    build_protocol_artifacts(repository_root=root)
    artifact_root = root / PROTOCOL_DIRECTORY
    artifacts = sorted(path for path in artifact_root.rglob("*") if path.is_file())
    assert len(artifacts) == 11
    for artifact in artifacts:
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"tamper\n")
        with pytest.raises(Joint500ProtocolError, match="mismatch"):
            validate_protocol_artifacts(repository_root=root)
        artifact.write_bytes(original)
    assert validate_protocol_artifacts(repository_root=root)["status"] == "ok"

    root = _copy_protocol_inputs(tmp_path / "base")
    build_protocol_artifacts(repository_root=root)
    base_manifest = root / "protocols" / "sd300b" / "plain_self.csv"
    base_manifest.write_bytes(base_manifest.read_bytes() + b"\n")
    with pytest.raises(Joint500ProtocolError, match="mismatch"):
        validate_protocol_artifacts(repository_root=root)


def test_joint500_cli_exposes_all_required_phases_and_filters():
    for phase in ("build", "validate", "preflight-sourceafis", "run", "report"):
        args = parse_args(["detector-joint500", phase])
        assert args.joint_phase == phase
    run = parse_args(
        [
            "detector-joint500",
            "run",
            "--dataset",
            "sd300c",
            "--pair-kind",
            "plain_roll_impostor",
            "--method",
            "sourceafis_final_minutiae_rootsift_geometric",
        ]
    )
    assert run.dataset == "sd300c"
    assert run.pair_kind == "plain_roll_impostor"


def test_screening_report_uses_correct_score_direction_far_rule_failures_and_logical_bc_join(tmp_path):
    results = tmp_path / "results"
    method = "synthetic_rootsift_geometric"
    for dataset in ("sd300b", "sd300c"):
        for pair_kind in ("plain_self", "plain_roll_genuine", "plain_roll_impostor"):
            rows = _synthetic_rows(dataset, pair_kind, method)
            if dataset == "sd300c":
                rows.reverse()
            path = results / PROTOCOL_NAME / dataset / pair_kind / method / "pairs.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_rows(path, rows)

    outcome = report_joint500(results_root=results)
    assert outcome["status"] == "ok"
    report = json.loads(
        (results / PROTOCOL_NAME / "report" / "report.json").read_text(encoding="utf-8")
    )
    assert report["screening_only"] is True
    assert report["far_resolution"] == 0.002
    assert report["reported_far_operating_points"] == [0.01]
    assert report["threshold_calibration"] == "none"
    assert all(item["auc"] > 0.5 for item in report["genuine_impostor_screening"])
    assert all(item["screening_eer"] >= 0 for item in report["genuine_impostor_screening"])
    assert all(item["actual_achieved_far"] <= 0.01 for item in report["genuine_impostor_screening"])
    self_summary = next(item for item in report["summaries"] if item["pair_kind"] == "plain_self")
    genuine_summary = next(item for item in report["summaries"] if item["pair_kind"] == "plain_roll_genuine")
    assert self_summary["prepare_failures"] == 1
    assert genuine_summary["requested_rows"] == 4
    assert genuine_summary["successful_rows"] == 4
    assert all(
        item["join_semantics"] == "logical_pair_id_not_row_position"
        for item in report["paired_sd300b_sd300c"]
    )


def _copy_protocol_inputs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for dataset in ("sd300b", "sd300c"):
        shutil.copytree(
            REPOSITORY_ROOT / "protocols" / dataset,
            root / "protocols" / dataset,
        )
    return root


def _synthetic_rows(dataset: str, pair_kind: str, method: str):
    if pair_kind == "plain_roll_genuine":
        scores = [9.0, 8.0, 7.0, 6.0]
    elif pair_kind == "plain_roll_impostor":
        scores = [4.0, 3.0, 2.0, 1.0]
    else:
        scores = [10.0, 10.0, 10.0, 10.0]
    rows = []
    for index, score in enumerate(scores):
        failed = pair_kind == "plain_self" and index == 0
        prepare = json.dumps(
            {
                "detector_point_count": 20 + index,
                "representation_descriptor_count": 18 + index,
                "detector_time_ms": 1.0 + index,
            },
            separators=(",", ":"),
        )
        compare = json.dumps(
            {
                "mutual_match_count": 8 + index,
                "geometric_inlier_count": int(score),
                "inlier_ratio": score / 10,
                "score_components": {"inliers_over_min_keypoints": score / 20},
            },
            separators=(",", ":"),
        )
        rows.append(
            {
                "pair_id": f"{dataset}_logical_{pair_kind}_{index:02d}",
                "dataset": dataset,
                "protocol": f"{PROTOCOL_NAME}_{pair_kind}",
                "method": method,
                "raw_score": "" if failed else repr(score + (0.5 if dataset == "sd300c" else 0.0)),
                "prepare_a_ms": "1.0",
                "prepare_b_ms": "1.1",
                "compare_ms": "0.5" if not failed else "",
                "prepare_a_diagnostics": prepare,
                "prepare_b_diagnostics": "" if failed else prepare,
                "compare_diagnostics": "" if failed else compare,
                "status": "prepare_a_failure" if failed else "ok",
                "error_code": "synthetic_prepare" if failed else "",
                "error_message": "synthetic" if failed else "",
            }
        )
    return rows


def _write_rows(path: Path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
