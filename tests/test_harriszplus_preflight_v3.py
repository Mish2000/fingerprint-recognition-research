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
    EXPECTED_V1_ALGORITHM_SOURCE_SHA256,
    _semantic_pair_comparison,
)
from fingerprint_benchmark.harriszplus.preflight_v3 import (
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PARENT_SHA256,
    FAILURE_RELATIVE,
    MAX_KEYPOINTS,
    MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION,
    NO_V4_RELAXATION_PATH,
    PAIR_CLASSES,
    PASS_RELATIVE,
    HarrisZPlusPreflightV3Error,
    _pilot_identity_keys,
    _synthetic_comparison,
    aggregate_decision_equivalence,
    exact_score_rate,
    load_engineering_identities,
    load_engineering_pairs,
    prepare_engineering_fixtures,
    require_pilot_authorization,
    top_k_equivalence,
)
from fingerprint_benchmark.harriszplus.provenance import (
    implementation_source_hashes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIXTURE_SHA256 = {
    "identities": "8d8e3b74952d08138648b4a9542e2c763ad9ee6ceb32b63948a0f1f6c60c1ae1",
    "pairs": "f475cd22399fbead5cbbcbd89e5b4e92e2088d7d9dd65ffb5dd7d8f95722bc57",
    "provenance": "0c1ec2e34717ba40bbcd0446910194117b3ac54dec9942893d92e2dc20adb9ae",
}
EXPECTED_IDENTITY_KEYS = [
    "00001019|5",
    "00001019|7",
    "00001019|10",
    "00001040|2",
    "00001054|1",
    "00001054|2",
    "00001054|3",
    "00001054|4",
    "00001054|5",
    "00001054|6",
]


def _pair_payload(*, score: int, decision: bool | None = None) -> dict[str, object]:
    return {
        "status": "ok",
        "failure_stage": None,
        "raw_score": score,
        "decision_threshold_4": score >= 4 if decision is None else decision,
        "prepare_a": {"descriptor_count": 100},
        "prepare_b": {"descriptor_count": 100},
        "payload_sha256": "payload",
    }


def _top_k_record(*, candidates: int, duplicate_source: bool = False) -> dict:
    count = MAX_KEYPOINTS
    source_indices = np.arange(count, dtype=np.int64)
    if duplicate_source:
        source_indices[-1] = source_indices[-2]
    payload = SimpleNamespace(
        keypoint_count=count,
        points=np.column_stack(
            (np.arange(count, dtype=np.float32), np.zeros(count, dtype=np.float32))
        ),
        responses=np.arange(count, 0, -1, dtype=np.float32),
        class_ids=np.zeros(count, dtype=np.int32),
        sizes=np.ones(count, dtype=np.float32),
        metadata={"harriszplus_source_indices": source_indices.tolist()},
    )
    outcome = SimpleNamespace(
        representation=SimpleNamespace(payload=payload),
    )
    return {
        "candidates_after_duplicate_removal": candidates,
        "outcome": outcome,
    }


def test_v3_contract_is_frozen_and_spearman_is_diagnostic_only() -> None:
    path = (
        PROJECT_ROOT
        / "results/harriszplus_rootsift_geometric_v3/preflight"
        / "engineering_preflight_contract_v3.json"
    )
    assert file_sha256(path) == EXPECTED_CONTRACT_SHA256
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen"
    assert contract["frozen_before_fixture_materialization"] is True
    assert contract["spearman_diagnostics"]["is_gate"] is False
    assert contract["spearman_diagnostics"]["minimum_pass_threshold"] is None


def test_low_spearman_alone_does_not_fail_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fingerprint_benchmark.harriszplus.preflight_v3._response_map_comparison",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        "fingerprint_benchmark.harriszplus.preflight_v3.compare_candidate_counts_v2",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        "fingerprint_benchmark.harriszplus.preflight_v3.compare_final_keypoints_v2",
        lambda *_args, **_kwargs: {
            "passed": False,
            "count_gate_passed": True,
            "bidirectional_matching_passed": True,
            "spearman_passed": False,
            "response_rank_spearman": -1.0,
        },
    )
    monkeypatch.setattr(
        "fingerprint_benchmark.harriszplus.preflight_v3._keypoints",
        lambda _result: (),
    )
    monkeypatch.setattr(
        "fingerprint_benchmark.harriszplus.preflight_v3._spearman_diagnostics",
        lambda *_args: {"is_gate": False, "global_spearman": -1.0},
    )
    report = _synthetic_comparison(object(), object())
    assert report["spearman_diagnostics"]["global_spearman"] == -1.0
    assert report["spearman_is_gate"] is False
    assert report["passed"] is True


def test_top_k_overlap_gate_and_cutoff_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = iter((2_990, 2_990))
    monkeypatch.setattr(
        "fingerprint_benchmark.harriszplus.preflight_v3._directional_keypoint_matches",
        lambda source, _target: [
            (index, index, 0.0, 0.0)
            for index in range(next(calls))
        ],
    )
    report = top_k_equivalence(
        _top_k_record(candidates=3_100),
        _top_k_record(candidates=3_101),
    )
    assert report["applicable"] is True
    assert report["counts_exact_3000_3000"] is True
    assert report["top_k_entries_unique"] is True
    assert report["bidirectional_overlap"] == pytest.approx(2_990 / 3_000)
    assert report["minimum_bidirectional_overlap"] == (
        MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION
    )
    assert report["cpu_only_count"] == report["cuda_only_count"] == 10
    assert report["cutoff_region_start_rank_zero_based"] == 2_700
    assert report["differences_concentrated_around_cutoff"] is True
    assert report["passed"] is True


def test_top_k_cap_activation_and_inactivation() -> None:
    inactive = top_k_equivalence(
        _top_k_record(candidates=3_000),
        _top_k_record(candidates=3_001),
    )
    assert inactive["applicable"] is False
    assert inactive["passed"] is True


def test_top_k_requires_unique_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fingerprint_benchmark.harriszplus.preflight_v3._directional_keypoint_matches",
        lambda source, _target: [
            (index, index, 0.0, 0.0) for index in range(len(source))
        ],
    )
    report = top_k_equivalence(
        _top_k_record(candidates=3_001, duplicate_source=True),
        _top_k_record(candidates=3_001),
    )
    assert report["top_k_entries_unique"] is False
    assert report["passed"] is False


def test_external_fixture_selection_is_deterministic_and_immutable() -> None:
    first = prepare_engineering_fixtures(project_root=PROJECT_ROOT)
    second = prepare_engineering_fixtures(project_root=PROJECT_ROOT)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["artifact_sha256"] == EXPECTED_FIXTURE_SHA256
    assert [
        row.identity_key for row in load_engineering_identities(PROJECT_ROOT)
    ] == EXPECTED_IDENTITY_KEYS


def test_selected_engineering_identities_are_outside_pilot_500() -> None:
    pilot = _pilot_identity_keys(PROJECT_ROOT)
    selected = {
        (row.subject_id, row.canonical_finger_position)
        for row in load_engineering_identities(PROJECT_ROOT)
    }
    assert len(selected) == 10
    assert selected.isdisjoint(pilot)


def test_self_genuine_and_negative_pair_fixture_semantics() -> None:
    rows = load_engineering_pairs(PROJECT_ROOT)
    assert len(rows) == 60
    for dataset in ("sd300b", "sd300c"):
        assert {
            pair_class: sum(
                row.dataset == dataset and row.pair_class == pair_class
                for row in rows
            )
            for pair_class in PAIR_CLASSES
        } == {
            "plain_self": 5,
            "roll_self": 5,
            "genuine": 10,
            "negative": 10,
        }
    assert all(
        row.path_a == row.path_b and row.subject_id_a == row.subject_id_b
        for row in rows
        if row.pair_class in ("plain_self", "roll_self")
    )
    assert all(
        row.subject_id_a == row.subject_id_b
        for row in rows
        if row.pair_class == "genuine"
    )
    assert all(
        row.subject_id_a != row.subject_id_b and row.negative_shift == 1
        for row in rows
        if row.pair_class == "negative"
    )


def test_exact_score_rate_gate_calculation() -> None:
    rows = [
        {"cpu_cuda_semantic_equivalence": {"raw_score_exact": True}}
        for _ in range(57)
    ] + [
        {"cpu_cuda_semantic_equivalence": {"raw_score_exact": False}}
        for _ in range(3)
    ]
    assert exact_score_rate(rows) == pytest.approx(0.95)


def test_decision_equality_is_a_mandatory_pair_gate() -> None:
    cpu = _pair_payload(score=4)
    cuda = _pair_payload(score=3)
    report = _semantic_pair_comparison(cpu, cuda)
    assert report["raw_score_within_one_inlier"] is True
    assert report["decision_threshold_4_equal"] is False
    assert report["passed"] is False


def test_aggregate_decisions_must_match_by_dataset_and_class() -> None:
    rows = [
        {
            "dataset": "sd300b",
            "pair_class": "genuine",
            "cpu": {"decision_threshold_4": True},
            "cuda_first": {"decision_threshold_4": False},
        }
    ]
    report = aggregate_decision_equivalence(rows)
    assert report["groups"]["sd300b/genuine"]["passed"] is False
    assert report["passed"] is False


def test_v3_has_no_v4_relaxation_path() -> None:
    assert NO_V4_RELAXATION_PATH is True
    contract = json.loads(
        (
            PROJECT_ROOT
            / "results/harriszplus_rootsift_geometric_v3/preflight"
            / "engineering_preflight_contract_v3.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["failure_policy"]["v4_relaxation_path_allowed"] is False


def test_v1_v2_parent_artifacts_remain_byte_exact() -> None:
    assert {
        path: file_sha256(PROJECT_ROOT / path)
        for path in EXPECTED_PARENT_SHA256
    } == EXPECTED_PARENT_SHA256


def test_algorithm_sources_and_config_remain_byte_exact() -> None:
    assert (
        implementation_source_hashes(strict=True)[
            "required_score_producing_sources"
        ]
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


def test_no_500_pilot_before_v3_pass(tmp_path: Path) -> None:
    assert not (tmp_path / PASS_RELATIVE).exists()
    assert not (tmp_path / FAILURE_RELATIVE).exists()
    with pytest.raises(HarrisZPlusPreflightV3Error, match="500 run is forbidden"):
        require_pilot_authorization(project_root=tmp_path)
