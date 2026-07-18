from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fingerprint_benchmark.hashing import file_sha256, stable_config_hash
from fingerprint_benchmark.harriszplus.adapter import HarrisZPlusGeometricAdapter
from fingerprint_benchmark.harriszplus.config import HarrisZPlusConfig
from fingerprint_benchmark.harriszplus.preflight import _effective_runner_config
from fingerprint_benchmark.harriszplus.preflight_v2 import (
    EXPECTED_CANDIDATE_CONFIG_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_V1_ALGORITHM_SOURCE_SHA256,
    EXPECTED_V1_FAILURE_SHA256,
    HarrisZPlusPreflightV2Error,
    _cuda_repeat_comparison,
    _response_map_comparison,
    _semantic_pair_comparison,
    allowed_absolute_delta,
    compare_candidate_counts_v2,
    compare_final_keypoints_v2,
    count_equivalence,
    require_pilot_authorization,
)
from fingerprint_benchmark.harriszplus.provenance import implementation_source_hashes
from fingerprint_benchmark.harriszplus.types import DetectedKeypoint, DetectorResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]

V1_ARTIFACT_SHA256 = {
    "results/harriszplus_rootsift_geometric/preflight/engineering_preflight_failure.json": (
        EXPECTED_V1_FAILURE_SHA256
    ),
    "results/harriszplus_rootsift_geometric/preflight/README.md": (
        "ea19481a2d73f0a9e63244b01d01668cccd240f2b639c4d0e67071d20d158fef"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/integrity/protected_after.json": (
        "aecb657527edbba749859a539cddee293c093eac6eb37ce13fc00fba7f42ee35"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/integrity/protected_before.json": (
        "aecb657527edbba749859a539cddee293c093eac6eb37ce13fc00fba7f42ee35"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/integrity/protected_integrity.json": (
        "9f1dcacebb1438c49d06adf8c313d0afe6036581a4cbd22dcb70f2519e534581"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/manifests/base_manifest_provenance.json": (
        "a942a86a6e056a8e7e7e9f7014c74cae8c7a02c84711817e4bdf01071b1253e5"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/manifests/sd300b/plain_self.csv": (
        "b08f663f6018e7d212d9b8034cbf72597e27a69d491e805d8ec6bd894dc8c202"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/manifests/sd300b/roll_self.csv": (
        "9ef3188593140e21be8426fb5065c56530e2c367cbc35c11af4c14b3c23f1988"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/manifests/sd300c/plain_self.csv": (
        "4bdd7485759425135b71461f136bd4f285e046ab68db55f9b2af3bd1f0ffc30d"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/manifests/sd300c/roll_self.csv": (
        "27f33dc5a634369f72577614501a281fd359f6e561db65988dfca0255301282a"
    ),
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1/selected_identities_reference.json": (
        "77b79ddfac90b3621e1f0707e813126e0fef02b620cb276dfc542fb044505670"
    ),
}
EXPECTED_V2_FAILURE_SHA256 = (
    "db18ba8747de4a436fcff78e259ca2d56aa0f639c4f9febe3bee0d2f52d953ce"
)


def _point(
    x: float,
    *,
    y: float = 10.0,
    response: float = 1.0,
    sigma: float = 1.0,
    source_index: int = 0,
) -> DetectedKeypoint:
    return DetectedKeypoint(
        x=x,
        y=y,
        response=response,
        scale_index=0,
        sigma=sigma,
        integration_sigma=sigma * np.sqrt(2.0),
        effective_support_diameter=7.0,
        size=2.0 * sigma * np.sqrt(2.0),
        source_index=source_index,
    )


def _result(points: list[DetectedKeypoint]) -> DetectorResult:
    return DetectorResult(
        backend="test",
        keypoints=tuple(points),
        diagnostics={},
        timings={},
        response_maps={0: np.zeros((2, 2), dtype=np.float32)},
    )


def _pair_payload(
    *,
    score: int,
    decision: bool | None = None,
    descriptors: int = 100,
    status: str = "ok",
    stage: str | None = None,
    payload_hash: str = "same",
) -> dict[str, object]:
    prepare = {"descriptor_count": descriptors}
    return {
        "status": status,
        "failure_stage": stage,
        "raw_score": score,
        "decision_threshold_4": score >= 4 if decision is None else decision,
        "prepare_a": prepare,
        "prepare_b": prepare,
        "payload_sha256": payload_hash,
    }


def test_low_count_candidate_absolute_delta_rule() -> None:
    row = count_equivalence(13, 11)
    assert row["absolute_delta"] == row["allowed_absolute_delta"] == 2
    assert row["minimum_to_maximum_ratio_diagnostic_only"] == pytest.approx(11 / 13)
    assert row["legacy_ratio_is_gate"] is False
    assert row["passed"] is True


def test_high_count_hybrid_tolerance() -> None:
    assert allowed_absolute_delta(10_000, 9_950) == 50
    assert count_equivalence(10_000, 9_950)["passed"] is True
    assert count_equivalence(10_000, 9_949)["passed"] is False


def test_zero_versus_nonzero_candidate_failure() -> None:
    assert count_equivalence(0, 1)["zero_versus_nonzero"] is True
    assert count_equivalence(0, 1)["passed"] is False
    assert count_equivalence(0, 0)["passed"] is True


def test_all_intermediate_candidate_stages_use_v2_rule() -> None:
    keys = (
        "candidates_before_mask",
        "candidates_after_mask",
        "candidates_after_local_maxima",
        "candidates_after_scale_suppression",
        "candidates_after_eigen_ratio",
    )
    cpu_counts = {key: 13 for key in keys}
    cuda_counts = {key: 11 for key in keys}
    cpu = SimpleNamespace(
        diagnostics={
            **cpu_counts,
            "candidates_after_duplicate_removal": 13,
            "scales": {"1": {"counts": cpu_counts}},
        }
    )
    cuda = SimpleNamespace(
        diagnostics={
            **cuda_counts,
            "candidates_after_duplicate_removal": 11,
            "scales": {"1": {"counts": cuda_counts}},
        }
    )
    assert compare_candidate_counts_v2(cpu, cuda)["passed"] is True


def test_bidirectional_keypoint_matching() -> None:
    cpu = _result([_point(10.0, response=2.0), _point(20.0, response=1.0)])
    cuda = _result([_point(10.1, response=2.0), _point(20.1, response=1.0)])
    report = compare_final_keypoints_v2(cpu, cuda)
    assert report["cpu_to_cuda_matched_count"] == 2
    assert report["cuda_to_cpu_matched_count"] == 2
    assert report["bidirectional_matched_fraction"] == 1.0
    assert report["passed"] is True


def test_spatial_tolerance_is_half_pixel() -> None:
    cpu = _result([_point(10.0)])
    assert compare_final_keypoints_v2(cpu, _result([_point(10.5)]))["passed"] is True
    assert compare_final_keypoints_v2(cpu, _result([_point(10.5001)]))["passed"] is False


def test_relative_scale_tolerance_is_one_percent() -> None:
    cpu = _result([_point(10.0, sigma=1.0)])
    assert compare_final_keypoints_v2(cpu, _result([_point(10.0, sigma=1.01)]))[
        "passed"
    ] is True
    assert compare_final_keypoints_v2(cpu, _result([_point(10.0, sigma=1.011)]))[
        "passed"
    ] is False


def test_keypoint_count_tolerance() -> None:
    assert count_equivalence(503, 500)["passed"] is True
    assert count_equivalence(503, 499)["passed"] is False


def test_spearman_gate() -> None:
    cpu = _result(
        [
            _point(10.0, response=4.0),
            _point(20.0, response=3.0),
            _point(30.0, response=2.0),
            _point(40.0, response=1.0),
        ]
    )
    cuda = _result(
        [
            _point(40.0, response=4.0),
            _point(30.0, response=3.0),
            _point(20.0, response=2.0),
            _point(10.0, response=1.0),
        ]
    )
    report = compare_final_keypoints_v2(cpu, cuda)
    assert report["bidirectional_matched_fraction"] == 1.0
    assert report["response_rank_spearman"] == pytest.approx(-1.0)
    assert report["spearman_passed"] is False
    assert report["passed"] is False


def test_response_map_metrics_and_v1_tolerances_are_reported() -> None:
    cpu = SimpleNamespace(response_maps={0: np.array([[-0.1, 0.32]])})
    cuda = SimpleNamespace(response_maps={0: np.array([[0.1, 0.30]])})
    report = _response_map_comparison(cpu, cuda, minimum_pixel_coverage=0.0)
    row = report["per_scale"][0]
    assert report["response_atol"] == 5e-4
    assert report["response_rtol"] == 2e-4
    assert row["sign_disagreement_count_at_0"] == 1
    assert row["mask_disagreement_count_at_0.31"] == 1
    assert "normalized_rmse" in row


def test_downstream_decision_equality_is_mandatory() -> None:
    cpu = _pair_payload(score=4)
    cuda = _pair_payload(score=3)
    report = _semantic_pair_comparison(cpu, cuda)
    assert report["raw_score_within_one_inlier"] is True
    assert report["decision_threshold_4_equal"] is False
    assert report["passed"] is False


def test_raw_score_delta_gate() -> None:
    cpu = _pair_payload(score=8)
    assert _semantic_pair_comparison(cpu, _pair_payload(score=7))["passed"] is True
    assert _semantic_pair_comparison(cpu, _pair_payload(score=6))["passed"] is False


def test_exact_cuda_repeat_hashes() -> None:
    first = _pair_payload(score=4, payload_hash="a")
    same = _pair_payload(score=4, payload_hash="a")
    changed = _pair_payload(score=4, payload_hash="b")
    assert _cuda_repeat_comparison(first, same)["passed"] is True
    assert _cuda_repeat_comparison(first, changed)["passed"] is False


def test_v1_failure_artifacts_are_byte_exact() -> None:
    assert {
        path: file_sha256(PROJECT_ROOT / path)
        for path in V1_ARTIFACT_SHA256
    } == V1_ARTIFACT_SHA256


def test_v2_contract_was_frozen_before_preflight() -> None:
    path = (
        PROJECT_ROOT
        / "results/harriszplus_rootsift_geometric_v2/preflight"
        / "engineering_preflight_contract_v2.json"
    )
    assert file_sha256(path) == EXPECTED_CONTRACT_SHA256
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen"
    assert contract["immutable"] is True
    assert contract["frozen_before_preflight"] is True
    assert contract["frozen_before_any_500_result"] is True


def test_no_algorithm_parameter_or_score_producing_source_changed() -> None:
    assert (
        implementation_source_hashes(strict=True)["required_score_producing_sources"]
        == EXPECTED_V1_ALGORITHM_SOURCE_SHA256
    )
    adapter = HarrisZPlusGeometricAdapter(
        HarrisZPlusConfig(backend="cuda", device="cuda:0")
    )
    try:
        assert (
            stable_config_hash(_effective_runner_config(adapter.metadata()))
            == EXPECTED_CANDIDATE_CONFIG_SHA256
        )
    finally:
        adapter.close()


def test_no_500_run_before_pass_artifact(tmp_path: Path) -> None:
    with pytest.raises(HarrisZPlusPreflightV2Error, match="500 run is forbidden"):
        require_pilot_authorization(project_root=tmp_path)


def test_failed_v2_preflight_forbids_the_500_pilot() -> None:
    failure = (
        PROJECT_ROOT
        / "results/harriszplus_rootsift_geometric_v2/preflight"
        / "engineering_preflight_failure.json"
    )
    assert file_sha256(failure) == EXPECTED_V2_FAILURE_SHA256
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["pilot_500_authorized"] is False
    with pytest.raises(HarrisZPlusPreflightV2Error, match="500 run is forbidden"):
        require_pilot_authorization(project_root=PROJECT_ROOT)
