from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import numpy as np

from fingerprint_benchmark.contract import (
    BENCHMARK_CONTRACT_VERSION,
    HIGHER_IS_MORE_SIMILAR,
    BenchmarkRunSpec,
    CompareOutcome,
    MethodMetadata,
    PrepareOutcome,
    PreparedRepresentation,
)
from fingerprint_benchmark.hashing import canonical_json_bytes, file_sha256, stable_hash
from fingerprint_benchmark.harriszplus import pilot
from fingerprint_benchmark.harriszplus import preflight
from fingerprint_benchmark.harriszplus import reporting
from fingerprint_benchmark.harriszplus.preflight import (
    EXPECTED_SELECTION_SHA256,
    _ppi_coordinate_handling_evidence,
    compare_detector_results,
    detector_result_sha256,
    load_and_verify_selection,
)
from fingerprint_benchmark.manifest import PairRecord, read_pair_manifest
from fingerprint_benchmark.runner import _execute_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\fingerprint-datasets")
SOURCE_PILOT = PROJECT_ROOT / "results/pilots/sourceafis_joint_500_v1"


def test_exact_500_selection_hash_schema_and_order() -> None:
    path = SOURCE_PILOT / "selected_identities.csv"
    selected = load_and_verify_selection(path)
    assert file_sha256(path) == EXPECTED_SELECTION_SHA256
    assert len(selected) == 500
    assert [row.selection_index for row in selected] == list(range(1, 501))
    assert len({row.identity for row in selected}) == 500
    assert all(row.source_identity_key == f"{row.subject_id}|{row.canonical_finger_position}" for row in selected)


def test_threshold_four_is_inclusive_and_failures_never_accept() -> None:
    assert pilot.accepted_result_row({"status": "ok", "raw_score": "4.0"}) is True
    assert pilot.accepted_result_row({"status": "ok", "raw_score": "3.0"}) is False
    assert pilot.accepted_result_row({"status": "prepare_a_failure", "raw_score": ""}) is False
    with pytest.raises(pilot.HarrisZPlusPilotError, match="non-negative integer"):
        pilot.accepted_result_row({"status": "ok", "raw_score": "4.5"})


def test_empty_flat_keypoint_lists_pass_and_response_mapping_bytes_are_hashed() -> None:
    first = SimpleNamespace(
        response_maps={0: np.zeros((5, 5), dtype=np.float32)},
        keypoints=(),
        diagnostics={},
    )
    second = SimpleNamespace(
        response_maps={0: np.zeros((5, 5), dtype=np.float32)},
        keypoints=(),
        diagnostics={},
    )
    comparison = compare_detector_results(first, second)
    assert comparison["ordered_matched_keypoint_fraction"] == 1.0
    assert comparison["passed"] is True
    original_hash = detector_result_sha256(first)
    second.response_maps[0][2, 2] = 1
    assert detector_result_sha256(second) != original_hash


def test_response_validation_uses_frozen_statistics_and_pixel_coverage() -> None:
    cpu_map = np.zeros((100, 100), dtype=np.float32)
    cpu_map[0, 0] = -10.0
    cpu_map[0, 1] = 10.0
    cuda_map = cpu_map.copy()
    cuda_map[50, 50] = 0.01
    cpu = SimpleNamespace(response_maps={0: cpu_map}, keypoints=(), diagnostics={})
    cuda = SimpleNamespace(response_maps={0: cuda_map}, keypoints=(), diagnostics={})

    real_policy = compare_detector_results(cpu, cuda)
    row = real_policy["response_maps"][0]
    assert row["statistics_passed"] is True
    assert row["outlier_pixel_count"] == 1
    assert row["allclose_pixel_fraction"] == 0.9999
    assert row["coverage_passed"] is True
    assert real_policy["passed"] is True

    synthetic_policy = compare_detector_results(
        cpu,
        cuda,
        minimum_response_pixel_coverage=1.0,
    )
    assert synthetic_policy["response_maps"][0]["coverage_passed"] is False
    assert synthetic_policy["passed"] is False

    cuda_map[50, 51] = 0.01
    below_real_coverage = compare_detector_results(cpu, cuda)
    assert below_real_coverage["response_maps"][0]["outlier_pixel_count"] == 2
    assert below_real_coverage["passed"] is False


def test_response_coverage_cannot_hide_statistic_or_hard_maximum_failure() -> None:
    zero = np.zeros((100, 100), dtype=np.float32)
    statistic_failure = zero.copy()
    statistic_failure[50, 50] = 0.05
    cpu = SimpleNamespace(response_maps={0: zero}, keypoints=(), diagnostics={})
    cuda = SimpleNamespace(
        response_maps={0: statistic_failure}, keypoints=(), diagnostics={}
    )
    comparison = compare_detector_results(cpu, cuda)
    row = comparison["response_maps"][0]
    assert row["allclose_pixel_fraction"] == 0.9999
    assert row["coverage_passed"] is True
    assert row["maximum_delta_passed"] is True
    assert row["statistics_passed"] is False
    assert comparison["passed"] is False

    bounded = zero.copy()
    bounded[0, 0] = -10.0
    bounded[0, 1] = 10.0
    excessive_delta = bounded.copy()
    excessive_delta[50, 50] = 0.2
    cpu.response_maps[0] = bounded
    cuda.response_maps[0] = excessive_delta
    comparison = compare_detector_results(cpu, cuda)
    row = comparison["response_maps"][0]
    assert row["coverage_passed"] is True
    assert row["statistics_passed"] is True
    assert row["maximum_delta_passed"] is False
    assert comparison["passed"] is False


def _candidate_diagnostics(count: int = 10_000) -> dict[str, object]:
    per_scale = {
        "dense_pixels": 20_000,
        "candidates_before_mask": count,
        "candidates_after_mask": count,
        "candidates_after_local_maxima": count,
        "candidates_after_scale_suppression": count,
        "candidates_after_eigen_ratio": count,
    }
    return {
        "candidates_before_mask": count,
        "candidates_after_mask": count,
        "candidates_after_local_maxima": count,
        "candidates_after_scale_suppression": count,
        "candidates_after_eigen_ratio": count,
        "candidates_after_duplicate_removal": count,
        "candidates_after_uniform_selection": 3000,
        "final_keypoint_count": 3000,
        "scales": {"0": {"counts": per_scale}},
    }


def test_candidate_count_ratios_and_uniform_final_exact_gate() -> None:
    response = np.zeros((5, 5), dtype=np.float32)
    cpu = SimpleNamespace(
        response_maps={0: response}, keypoints=(), diagnostics=_candidate_diagnostics()
    )
    cuda = SimpleNamespace(
        response_maps={0: response.copy()}, keypoints=(), diagnostics=_candidate_diagnostics()
    )
    comparison = compare_detector_results(cpu, cuda)
    assert comparison["candidate_counts_passed"] is True

    cuda.diagnostics["scales"]["0"]["counts"]["candidates_after_eigen_ratio"] = 9994
    comparison = compare_detector_results(cpu, cuda)
    assert comparison["candidate_counts"]["per_scale"]["0"][
        "candidates_after_eigen_ratio"
    ]["minimum_to_maximum_ratio"] == 0.9994
    assert comparison["candidate_counts_passed"] is False
    assert comparison["passed"] is False

    cuda.diagnostics = _candidate_diagnostics()
    cuda.diagnostics["final_keypoint_count"] = 2999
    comparison = compare_detector_results(cpu, cuda)
    assert comparison["candidate_counts"]["aggregate"]["final_keypoint_count"][
        "required_exact"
    ] is True
    assert comparison["candidate_counts_passed"] is False


def test_ordered_keypoint_gate_is_separate_from_nearest_unused_agreement() -> None:
    response = np.zeros((5, 5), dtype=np.float32)
    points = [
        {
            "x": float(index),
            "y": 0.0,
            "response": float(20 - index),
            "scale_index": 0,
            "sigma": 1.0,
        }
        for index in range(20)
    ]
    cpu = SimpleNamespace(response_maps={0: response}, keypoints=points, diagnostics={})
    cuda = SimpleNamespace(
        response_maps={0: response.copy()}, keypoints=list(reversed(points)), diagnostics={}
    )
    comparison = compare_detector_results(cpu, cuda)
    assert comparison["matched_keypoint_fraction"] == 1.0
    assert comparison["nearest_keypoints_passed"] is True
    assert comparison["ordered_matched_keypoint_fraction"] == 0.0
    assert comparison["matched_order_spearman_rank_correlation"] == -1.0
    assert comparison["order_correlation_passed"] is False
    assert comparison["ordered_keypoints_passed"] is False
    assert comparison["passed"] is False


def test_order_rank_correlation_tolerates_local_jitter_but_rejects_random_order() -> None:
    response = np.zeros((5, 5), dtype=np.float32)
    points = [
        {
            "x": float(index * 2),
            "y": 0.0,
            "response": float(200 - index),
            "scale_index": 0,
            "sigma": 1.0,
            "source_index": index,
        }
        for index in range(100)
    ]
    cpu = SimpleNamespace(response_maps={0: response}, keypoints=points, diagnostics={})
    locally_jittered = [
        points[index + 1] if index % 2 == 0 else points[index - 1]
        for index in range(len(points))
    ]
    cuda = SimpleNamespace(
        response_maps={0: response.copy()}, keypoints=locally_jittered, diagnostics={}
    )
    comparison = compare_detector_results(cpu, cuda)
    assert comparison["matched_keypoint_fraction"] == 1.0
    assert comparison["ordered_matched_keypoint_fraction"] == 0.0
    assert comparison["canonical_ordered_matched_keypoint_fraction"] == 0.0
    assert comparison["matched_order_spearman_rank_correlation"] > 0.99
    assert comparison["order_correlation_passed"] is True
    assert comparison["passed"] is True

    permutation = np.random.default_rng(0).permutation(len(points))
    cuda.keypoints = [points[int(index)] for index in permutation]
    comparison = compare_detector_results(cpu, cuda)
    assert comparison["matched_keypoint_fraction"] == 1.0
    assert comparison["matched_order_spearman_rank_correlation"] < 0.99
    assert comparison["order_correlation_passed"] is False
    assert comparison["passed"] is False


def test_order_rank_correlation_canonicalizes_only_contiguous_response_ties() -> None:
    response = np.zeros((5, 5), dtype=np.float32)
    tied_points = [
        {
            "x": float(index * 2),
            "y": 0.0,
            "response": 10.0,
            "scale_index": 0,
            "sigma": 1.0,
            "source_index": index,
        }
        for index in range(20)
    ]
    cpu = SimpleNamespace(
        response_maps={0: response}, keypoints=tied_points, diagnostics={}
    )
    cuda = SimpleNamespace(
        response_maps={0: response.copy()},
        keypoints=list(reversed(tied_points)),
        diagnostics={},
    )
    comparison = compare_detector_results(cpu, cuda)
    assert comparison["ordered_matched_keypoint_fraction"] == 0.0
    assert comparison["canonical_ordered_matched_keypoint_fraction"] == 1.0
    assert comparison["matched_order_spearman_rank_correlation"] == 1.0
    assert comparison["order_correlation_passed"] is True
    assert comparison["passed"] is True


def test_detector_repeat_hash_excludes_timing_and_memory_telemetry_only() -> None:
    response = np.zeros((3, 3), dtype=np.float32)
    first = SimpleNamespace(
        response_maps={0: response},
        keypoints=(),
        diagnostics={
            "total_ms": 1.0,
            "peak_vram_allocated_bytes": 10,
            "nested": {"memory_reserved_bytes": 20},
            "stable_count": 3,
        },
    )
    second = SimpleNamespace(
        response_maps={0: response.copy()},
        keypoints=(),
        diagnostics={
            "total_ms": 9.0,
            "peak_vram_allocated_bytes": 999,
            "nested": {"memory_reserved_bytes": 888},
            "stable_count": 3,
        },
    )
    assert detector_result_sha256(first) == detector_result_sha256(second)
    second.diagnostics["stable_count"] = 4
    assert detector_result_sha256(first) != detector_result_sha256(second)


@pytest.mark.parametrize(
    ("dataset", "manifest_ppi", "native_threshold"),
    (("sd300b", 1000, 3.0), ("sd300c", 2000, 6.0)),
)
def test_preflight_gates_and_records_manifest_ppi_coordinate_handling(
    dataset: str,
    manifest_ppi: int,
    native_threshold: float,
) -> None:
    def prepared(ppi: float) -> PrepareOutcome:
        representation = PreparedRepresentation(
            "m",
            "v",
            "f",
            "rv",
            SimpleNamespace(ppi=ppi),
        )
        return PrepareOutcome(representation)

    diagnostics = {
        "coordinate_normalization": "ppi_to_reference",
        "reference_ppi": 1000.0,
        "ransac_threshold_reference_pixels": 3.0,
    }
    evidence = _ppi_coordinate_handling_evidence(
        dataset=dataset,
        manifest_ppi=manifest_ppi,
        prepared_a=prepared(float(manifest_ppi)),
        prepared_b=prepared(float(manifest_ppi)),
        compare_diagnostics=diagnostics,
    )
    assert evidence["both_prepared_payloads_carry_manifest_ppi"] is True
    assert evidence["geometry_diagnostics_passed"] is True
    assert evidence["native_equivalent_threshold_pixels"] == native_threshold
    assert evidence["passed"] is True

    wrong_payload = _ppi_coordinate_handling_evidence(
        dataset=dataset,
        manifest_ppi=manifest_ppi,
        prepared_a=prepared(float(manifest_ppi)),
        prepared_b=prepared(1000.0 if manifest_ppi == 2000 else 2000.0),
        compare_diagnostics=diagnostics,
    )
    assert wrong_payload["both_prepared_payloads_carry_manifest_ppi"] is False
    assert wrong_payload["passed"] is False


@pytest.mark.parametrize("dataset", ("sd300b", "sd300c"))
def test_per_dataset_survivors_genuine_subset_and_negative_bijection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
) -> None:
    source_root = tmp_path / "results/pilots/sourceafis_joint_500_v1"
    source_manifest_root = source_root / "manifests" / dataset
    source_manifest_root.mkdir(parents=True)
    selection_source = SOURCE_PILOT / "selected_identities.csv"
    shutil.copyfile(selection_source, source_root / "selected_identities.csv")
    genuine_source = SOURCE_PILOT / f"manifests/{dataset}/plain_roll_genuine.csv"
    genuine_target = source_manifest_root / "plain_roll_genuine.csv"
    shutil.copyfile(genuine_source, genuine_target)
    artifact_manifest = {
        "files": [
            {
                "path": f"manifests/{dataset}/plain_roll_genuine.csv",
                "size": genuine_target.stat().st_size,
                "sha256": file_sha256(genuine_target),
            }
        ]
    }
    (source_root / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest), encoding="utf-8"
    )

    plain_pairs = read_pair_manifest(SOURCE_PILOT / f"manifests/{dataset}/plain_self.csv")
    roll_pairs = read_pair_manifest(SOURCE_PILOT / f"manifests/{dataset}/roll_self.csv")

    def result_rows(pairs: list[PairRecord], *, plain: bool) -> list[dict[str, str]]:
        rows = []
        for index, pair in enumerate(pairs):
            status = "ok"
            score = "8.0"
            if plain and index == 0:
                score = "3.0"
            if not plain and index == 0:
                score = "3.0"
            if not plain and index == 1:
                status = "prepare_b_failure"
                score = ""
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "subject_id": pair.subject_id,
                    "canonical_finger_position": str(pair.canonical_finger_position),
                    "status": status,
                    "raw_score": score,
                }
            )
        return rows

    bundles = {
        "plain_self": pilot.BundleView(
            dataset,
            "plain_self",
            "plain_self",
            "self",
            Path("plain.csv"),
            Path("plain-bundle"),
            plain_pairs,
            result_rows(plain_pairs, plain=True),
            {},
        ),
        "roll_self": pilot.BundleView(
            dataset,
            "roll_self",
            "roll_self",
            "self",
            Path("roll.csv"),
            Path("roll-bundle"),
            roll_pairs,
            result_rows(roll_pairs, plain=False),
            {},
        ),
    }
    monkeypatch.setattr(
        pilot,
        "load_bundle",
        lambda *, project_root, dataset, label: bundles[label],
    )
    summary = pilot.derive_survivors_and_pair_manifests(
        project_root=tmp_path,
        data_root=DATA_ROOT,
        dataset=dataset,
    )
    assert summary["plain_self_accepted_count"] == 499
    assert summary["roll_self_accepted_count"] == 498
    assert summary["plain_self_rejected_or_failure_count"] == 1
    assert summary["roll_self_rejected_or_failure_count"] == 2
    assert summary["both_self_failure_count"] == 1
    assert summary["both_self_nonaccepted_count"] == 1
    assert summary["both_self_technical_failure_count"] == 0
    assert summary["survivor_count"] == 498
    assert summary["reason_counts"] == {
        "plain_self_score_below_4": 1,
        "roll_self_failure": 1,
        "roll_self_score_below_4": 1,
    }
    output_root = tmp_path / pilot.PILOT_RELATIVE / "manifests" / dataset
    genuine = read_pair_manifest(output_root / "plain_roll_genuine.csv")
    negative = read_pair_manifest(output_root / "plain_roll_negative.csv")
    excluded = {
        (plain_pairs[0].subject_id, plain_pairs[0].canonical_finger_position),
        (roll_pairs[1].subject_id, roll_pairs[1].canonical_finger_position),
    }
    assert len(genuine) == len(negative) == 498
    assert not excluded.intersection(
        (pair.subject_id, pair.canonical_finger_position) for pair in genuine
    )
    roll_owner = {pair.path_b: pair.subject_id for pair in genuine}
    assert len({pair.path_a for pair in negative}) == 498
    assert len({pair.path_b for pair in negative}) == 498
    assert all(roll_owner[pair.path_b] != pair.subject_id for pair in negative)
    assert all(pair.path_a != pair.path_b for pair in negative)


def test_shift_one_negative_is_grouped_by_finger_and_selection_index() -> None:
    identities = [
        pilot.SelectedIdentity(index, f"s{index}", finger, index, index + 1, f"s{index}|{finger}")
        for index, finger in ((1, 1), (2, 2), (3, 1), (4, 2), (5, 1), (6, 2))
    ]
    by_identity = {item.identity: item for item in identities}
    genuine = [
        PairRecord(
            pair_id=f"g{item.selection_index}",
            dataset="sd300b",
            protocol="plain_roll",
            subject_id=item.subject_id,
            canonical_finger_position=item.canonical_finger_position,
            ppi=1000,
            raw_frgp_a=item.selection_index,
            raw_frgp_b=item.selection_index,
            path_a=Path(f"plain-{item.subject_id}-{item.canonical_finger_position}"),
            path_b=Path(f"roll-{item.subject_id}-{item.canonical_finger_position}"),
        )
        for item in reversed(identities)
    ]
    negative, pairing = pilot.build_negative_pairs(
        dataset="sd300b",
        genuine_pairs=genuine,
        selection_by_identity=by_identity,
    )
    assert [row["selection_index_a"] for row in pairing] == [1, 3, 5, 2, 4, 6]
    assert [row["selection_index_b"] for row in pairing] == [3, 5, 1, 4, 6, 2]
    assert all(pairing_row["shift"] == 1 for pairing_row in pairing)
    assert all(pair.canonical_finger_position in (1, 2) for pair in negative)
    assert len({pair.path_a for pair in negative}) == len({pair.path_b for pair in negative}) == 6


def test_immutable_write_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "sealed.bin"
    pilot._publish_immutable_bytes(path, b"first")
    pilot._publish_immutable_bytes(path, b"first")
    with pytest.raises(pilot.HarrisZPlusPilotError, match="different bytes"):
        pilot._publish_immutable_bytes(path, b"second")
    assert path.read_bytes() == b"first"


def test_integrity_before_is_a_hard_gate_and_must_precede_publication(
    tmp_path: Path,
) -> None:
    with pytest.raises(pilot.HarrisZPlusPilotError, match="integrity-before"):
        pilot.run_pilot(project_root=tmp_path, data_root=DATA_ROOT)

    marker = (
        tmp_path
        / pilot.PILOT_RELATIVE
        / "runs/sd300b/plain_self/pairs.csv"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("already-published", encoding="utf-8")
    with pytest.raises(pilot.HarrisZPlusPilotError, match="must be captured before"):
        pilot.capture_integrity_before(project_root=tmp_path)


def test_complete_workflow_propagates_one_validated_integrity_token_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    token = {"validated": True, "before_sha256": "before"}

    def capture(**kwargs):
        events.append("integrity-before")
        return {"validation": token}

    def gated(name: str):
        def implementation(**kwargs):
            assert kwargs["integrity_before_validation"] is token
            events.append(name)
            return {"stage": name}

        return implementation

    def finalize(**kwargs):
        events.append("finalize")
        return {"stage": "finalize"}

    monkeypatch.setattr(pilot, "capture_integrity_before", capture)
    monkeypatch.setattr(pilot, "prepare_base_manifests", gated("prepare"))
    monkeypatch.setattr(pilot, "execute_engineering_preflight", gated("preflight"))
    monkeypatch.setattr(pilot, "freeze_after_preflight", gated("freeze"))
    monkeypatch.setattr(pilot, "run_pilot", gated("run"))
    monkeypatch.setattr(pilot, "finalize_outputs", finalize)
    result = pilot.run_complete_workflow(project_root=tmp_path, data_root=DATA_ROOT)
    assert events == [
        "integrity-before",
        "prepare",
        "preflight",
        "freeze",
        "run",
        "finalize",
    ]
    assert result["integrity_before"]["validation"] is token


def test_required_timing_and_peak_memory_gates_are_explicit_and_aggregate_maxima() -> None:
    timing = preflight._required_timing_evidence(
        {"kernel_ms": 1.5, "wall_ms": 2.5},
        ("kernel_ms", "wall_ms"),
        context="valid",
    )
    assert timing["passed"] is True
    invalid_timing = preflight._required_timing_evidence(
        {"kernel_ms": float("nan"), "wall_ms": -1.0},
        ("kernel_ms", "wall_ms", "transfer_ms"),
        context="invalid",
    )
    assert invalid_timing["missing_fields"] == ["transfer_ms"]
    assert invalid_timing["nonfinite_fields"] == ["kernel_ms"]
    assert invalid_timing["negative_fields"] == ["wall_ms"]
    assert invalid_timing["passed"] is False

    first = preflight._peak_memory_observation(
        {"allocated": 600, "reserved": 700},
        context="first",
        allocated_key="allocated",
        reserved_key="reserved",
    )
    second = preflight._peak_memory_observation(
        {"allocated": 750, "reserved": 850},
        context="second",
        allocated_key="allocated",
        reserved_key="reserved",
    )
    memory = preflight._evaluate_peak_memory((first, second), total_vram_bytes=1000)
    assert memory["peak_vram_allocated_bytes"] == 750
    assert memory["peak_vram_reserved_bytes"] == 850
    assert memory["peak_allocated_fraction_of_device"] == pytest.approx(0.75)
    assert memory["peak_reserved_fraction_of_device"] == pytest.approx(0.85)
    assert memory["maximum_allowed_fraction_of_device"] == 0.90
    assert memory["aggregation"] == "maximum_across_all_detector_and_prepare_calls"
    assert memory["passed"] is True

    over_limit = preflight._peak_memory_observation(
        {"allocated": 850, "reserved": 901},
        context="over-limit",
        allocated_key="allocated",
        reserved_key="reserved",
    )
    assert preflight._evaluate_peak_memory(
        (first, over_limit), total_vram_bytes=1000
    )["passed"] is False
    zero = preflight._peak_memory_observation(
        {"allocated": 0, "reserved": 0},
        context="zero",
        allocated_key="allocated",
        reserved_key="reserved",
    )
    assert zero["values_valid"] is False


def test_canonical_runtime_identity_detects_dependency_drift(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    environment = tmp_path / "environment.yml"
    pyproject.write_text("[project]\nname='fixture'\n", encoding="utf-8")
    environment.write_text("name: fixture\n", encoding="utf-8")
    runtime = {
        "python_version": "3.11.15",
        "python_executable": "C:/fixture/python.exe",
        "opencv_version": "4.12.0",
        "numpy_version": "2.2.6",
        "operating_system": "fixture-os",
        "torch": {
            "installed": True,
            "cuda_available": True,
            "version": "2.11.0+cu130",
            "cuda_build_runtime": "13.0",
            "cudnn_version": 91900,
            "selected_device": "cuda:0",
            "device_index": 0,
            "gpu_model": "fixture-gpu",
            "total_vram_bytes": 16_000_000_000,
            "compute_capability": "12.0",
        },
        "nvidia_smi": [{"index": 0, "driver_version": "592.01"}],
        "dependency_artifact_sha256": {
            "pyproject.toml": file_sha256(pyproject),
            "environment.yml": file_sha256(environment),
        },
    }
    identity = preflight._canonical_runtime_identity(tmp_path, runtime)
    assert identity["nvidia_driver_version"] == "592.01"
    assert identity["runtime_identity_hash"]

    environment.write_text("name: fixture\ndependencies: [changed]\n", encoding="utf-8")
    with pytest.raises(preflight.HarrisZPlusPreflightError, match="dependency hashes"):
        preflight._canonical_runtime_identity(tmp_path, runtime)


@pytest.mark.parametrize("requested", (None, "cuda", "cuda:0"))
def test_pilot_device_aliases_resolve_only_to_logical_cuda_zero(
    requested: str | None,
) -> None:
    assert preflight._canonical_pilot_cuda_device(requested) == "cuda:0"


@pytest.mark.parametrize("requested", ("cuda:1", "cpu", "0", "CUDA:0"))
def test_pilot_rejects_noncanonical_device_overrides(requested: str) -> None:
    with pytest.raises(preflight.HarrisZPlusPreflightError, match="bound to logical device cuda:0"):
        preflight._canonical_pilot_cuda_device(requested)


def test_immutable_binary_publication_preserves_lf_bytes_exactly(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.csv"
    payload = b"header\nfirst\nsecond\n"

    pilot._publish_immutable_bytes(target, payload)
    assert target.read_bytes() == payload
    pilot._publish_immutable_bytes(target, payload)

    with pytest.raises(pilot.HarrisZPlusPilotError, match="different bytes"):
        pilot._publish_immutable_bytes(target, payload.replace(b"first", b"changed"))


def test_final_artifact_manifest_rejects_unsealed_and_external_preflight_changes(
    tmp_path: Path,
) -> None:
    pilot_root = tmp_path / "results/pilots/hz"
    pilot_root.mkdir(parents=True)
    pilot_paths = []
    for logical_path in sorted(reporting._expected_pilot_artifact_paths()):
        path = pilot_root / logical_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sealed:{logical_path}\n", encoding="utf-8")
        pilot_paths.append((logical_path, path))
    config_file = tmp_path / pilot.METHOD_RESULTS_RELATIVE / "config/runtime_identity.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}\n", encoding="utf-8")
    preflight_file = (
        tmp_path
        / pilot.METHOD_RESULTS_RELATIVE
        / "preflight/engineering_preflight.json"
    )
    preflight_file.parent.mkdir(parents=True)
    preflight_file.write_text("{}\n", encoding="utf-8")

    def record(path: Path, logical_path: str) -> dict[str, object]:
        return {
            "path": logical_path,
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    def tree(records: list[dict[str, object]]) -> str:
        return hashlib.sha256(
            b"".join(canonical_json_bytes(item) + b"\n" for item in records)
        ).hexdigest()

    files = [record(path, logical_path) for logical_path, path in pilot_paths]
    config_files = [record(config_file, config_file.relative_to(tmp_path).as_posix())]
    preflight_files = [
        record(preflight_file, preflight_file.relative_to(tmp_path).as_posix())
    ]
    payload = {
        "schema_version": reporting.ARTIFACT_SCHEMA_VERSION,
        "namespace": pilot.PILOT_NAMESPACE,
        "hash_algorithm": "sha256",
        "immutable": True,
        "overwrite_allowed": False,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files),
        "tree_sha256": tree(files),
        "external_frozen_config_files": config_files,
        "external_frozen_config_tree_sha256": tree(config_files),
        "external_preflight_files": preflight_files,
        "external_preflight_tree_sha256": tree(preflight_files),
    }
    reporting._validate_artifact_manifest(pilot_root, payload)
    for key, invalid_value in (
        ("namespace", "wrong"),
        ("hash_algorithm", "sha1"),
        ("immutable", False),
        ("overwrite_allowed", True),
    ):
        invalid_payload = {**payload, key: invalid_value}
        with pytest.raises(
            reporting.HarrisZPlusReportingError,
            match="immutable namespace/hash semantics",
        ):
            reporting._validate_artifact_manifest(pilot_root, invalid_payload)

    extra_config = config_file.parent / "unsealed.txt"
    extra_config.write_text("unsealed\n", encoding="utf-8")
    with pytest.raises(reporting.HarrisZPlusReportingError, match="External frozen config inventory"):
        reporting._validate_artifact_manifest(pilot_root, payload)
    extra_config.unlink()

    unsealed = pilot_root / "nested/artifact_manifest.json"
    unsealed.parent.mkdir()
    unsealed.write_text("unsealed", encoding="utf-8")
    with pytest.raises(reporting.HarrisZPlusReportingError, match="explicit final artifact whitelist"):
        reporting._validate_expected_pilot_artifact_inventory(pilot_root)
    with pytest.raises(reporting.HarrisZPlusReportingError, match="unsealed files"):
        reporting._validate_artifact_manifest(pilot_root, payload)
    unsealed.unlink()
    unsealed.parent.rmdir()

    preflight_file.write_text("changed\n", encoding="utf-8")
    with pytest.raises(reporting.HarrisZPlusReportingError, match="External preflight"):
        reporting._validate_artifact_manifest(pilot_root, payload)


def test_resume_byte_validation_and_exact_eight_bundle_namespace(tmp_path: Path) -> None:
    report_root = tmp_path / "report"
    report_root.mkdir()
    expected = {
        "supervisor_report.json": b"{}\n",
        "supervisor_report.csv": b"header\n",
        "supervisor_report.md": b"# report\n",
    }
    for filename, content in expected.items():
        (report_root / filename).write_bytes(content)
    reporting._validate_exact_immutable_directory(
        report_root, expected, label="supervisor report"
    )
    (report_root / "supervisor_report.md").write_bytes(b"tampered\n")
    with pytest.raises(reporting.HarrisZPlusReportingError, match="current validated inputs"):
        reporting._validate_exact_immutable_directory(
            report_root, expected, label="supervisor report"
        )

    technical = tmp_path / "technical_provenance.json"
    technical.write_bytes(b"sealed\n")
    reporting._validate_exact_immutable_file(
        technical, b"sealed\n", label="technical provenance"
    )
    with pytest.raises(reporting.HarrisZPlusReportingError, match="current validated inputs"):
        reporting._validate_exact_immutable_file(
            technical, b"different\n", label="technical provenance"
        )

    pilot_root = tmp_path / "pilot"
    for dataset, label, _, _ in pilot.RUN_CONDITIONS:
        bundle = pilot_root / "runs" / dataset / label
        bundle.mkdir(parents=True)
        (bundle / "pairs.csv").write_text("pairs\n", encoding="utf-8")
        (bundle / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    reporting._validate_exact_run_namespace(pilot_root)
    extra_directory = pilot_root / "runs/sd300b/unlisted"
    extra_directory.mkdir()
    with pytest.raises(reporting.HarrisZPlusReportingError, match="exactly eight bundles"):
        reporting._validate_exact_run_namespace(pilot_root)


def test_bundle_identities_must_equal_current_validated_freeze() -> None:
    freeze = {
        "canonical_config_hash": "current-config",
        "implementation_hash": "current-implementation",
    }
    reporting._validate_bundle_freeze_identities(
        {"current-config"},
        {"current-implementation"},
        freeze,
    )
    with pytest.raises(reporting.HarrisZPlusReportingError, match="current frozen config"):
        reporting._validate_bundle_freeze_identities(
            {"uniform-but-stale-config"},
            {"uniform-but-stale-implementation"},
            freeze,
        )


def test_bundle_startup_validation_is_bound_to_every_current_artifact_and_policy(
    tmp_path: Path,
) -> None:
    paths = pilot.project_paths(tmp_path)
    paths["preflight"].parent.mkdir(parents=True)
    paths["preflight"].write_text("preflight\n", encoding="utf-8")
    before_path = paths["pilot_root"] / "integrity/protected_before.json"
    before_path.parent.mkdir(parents=True)
    inventory = {
        "schema_version": "harriszplus-manifest-referenced-dataset-inventory-v1",
        "all_protocol_dataset_stat_inventory_sha256": "stats",
        "authoritative_pilot_image_content_inventory_sha256": "content",
    }
    before = {
        "schema_version": preflight.INTEGRITY_SCHEMA_VERSION,
        "tree_sha256": "protected-tree",
        "manifest_referenced_dataset_inventory": inventory,
    }
    before["combined_snapshot_sha256"] = stable_hash(
        {
            "narrow_tree_sha256": before["tree_sha256"],
            "manifest_referenced_dataset_inventory": inventory,
        }
    )
    before_path.write_text(json.dumps(before), encoding="utf-8")
    freeze_path = paths["config"] / "freeze_manifest.json"
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(
        json.dumps(
            {
                "canonical_config_hash": "config",
                "implementation_hash": "implementation",
                "runtime_identity_hash": "runtime",
                "decision_rule_hash": "decision",
            }
        ),
        encoding="utf-8",
    )
    expected = pilot._expected_bundle_startup_validation(paths)
    assert set(expected) == {
        "engineering_preflight_sha256",
        "integrity_before_sha256",
        "integrity_before_combined_snapshot_sha256",
        "config_freeze_manifest_sha256",
        "canonical_config_hash",
        "implementation_hash",
        "runtime_identity_hash",
        "decision_rule_hash",
        "generic_runner",
        "timing_mode",
        "prepare_operations_per_pair",
        "cross_pair_cache",
    }
    metadata = {
        "config_hash": "config",
        "implementation_hash": "implementation",
        "timing_mode": "cold_pair",
        "startup_validation": expected,
    }
    pilot._validate_bundle_startup_validation(metadata, expected)

    for key, value in expected.items():
        mutated = dict(expected)
        if isinstance(value, bool):
            mutated[key] = not value
        elif isinstance(value, int):
            mutated[key] = value + 1
        else:
            mutated[key] = f"{value}-stale"
        with pytest.raises(pilot.HarrisZPlusPilotError, match="startup validation"):
            pilot._validate_bundle_startup_validation(
                {**metadata, "startup_validation": mutated},
                expected,
            )

    with pytest.raises(pilot.HarrisZPlusPilotError, match="startup validation"):
        pilot._validate_bundle_startup_validation(
            {
                **metadata,
                "startup_validation": {**expected, "unexpected": True},
            },
            expected,
        )


def test_finalization_validates_current_freeze_before_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Adapter:
        def close(self) -> None:
            events.append("close-adapter")

    monkeypatch.setattr(pilot, "_new_adapter", Adapter)
    monkeypatch.setattr(pilot, "_read_json", lambda path: {"passed": True})

    def validate(**kwargs):
        events.append("validate-freeze")
        return {"validated": True}

    monkeypatch.setattr(pilot, "validate_frozen_configuration", validate)
    monkeypatch.setattr(
        reporting,
        "build_supervisor_report",
        lambda **kwargs: events.append("report") or {},
    )
    monkeypatch.setattr(
        pilot,
        "capture_integrity_after",
        lambda **kwargs: events.append("integrity")
        or {"protected_artifacts_unchanged": True},
    )
    monkeypatch.setattr(
        reporting,
        "build_technical_provenance",
        lambda **kwargs: events.append("technical") or {},
    )
    monkeypatch.setattr(
        reporting,
        "build_artifact_manifest",
        lambda **kwargs: events.append("seal") or {},
    )
    result = pilot.finalize_outputs(project_root=tmp_path)
    assert events == [
        "validate-freeze",
        "close-adapter",
        "report",
        "integrity",
        "technical",
        "seal",
    ]
    assert result["freeze_validation"] == {"validated": True}


def test_recorded_artifact_manifest_rejects_added_and_changed_files(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "protected"
    namespace.mkdir()
    artifact = namespace / "artifact.txt"
    artifact.write_text("sealed\n", encoding="utf-8")
    record = {
        "path": "artifact.txt",
        "size": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }
    payload = {
        "files": [record],
        "file_count": 1,
        "total_bytes": artifact.stat().st_size,
        "tree_sha256": hashlib.sha256(
            canonical_json_bytes(record) + b"\n"
        ).hexdigest(),
    }
    manifest = namespace / "artifact_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert len(preflight._validate_recorded_artifact_manifest(manifest)) == 1

    extra = namespace / "extra.txt"
    extra.write_text("unsealed\n", encoding="utf-8")
    with pytest.raises(preflight.HarrisZPlusPreflightError, match="inventory changed"):
        preflight._validate_recorded_artifact_manifest(manifest)
    extra.unlink()
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(preflight.HarrisZPlusPreflightError, match="changed"):
        preflight._validate_recorded_artifact_manifest(manifest)


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256", "expected_file_count"),
    (
        (
            "results/pilots/sourceafis_joint_500_v1/artifact_manifest.json",
            "514ef8343d0eff83ad4b05a868ba4d7256836067232f5e87cf167575b11a3d90",
            55,
        ),
        (
            "results/pilots/sift_geometric_joint_500_v1/artifact_manifest.json",
            "e7a92d06736929bb6a59e7be5199f7582f3c570da4a352533d09d63d6575d2a8",
            46,
        ),
        (
            "results/shared_accuracy/sourceafis_sift_v1/artifact_manifest.json",
            "a5253bc1c589ed5b3cd9ba4b787c66ff5251f8cbad2b1ae3839bc17c22b84265",
            63,
        ),
    ),
)
def test_protected_baseline_artifact_manifest_identity_and_exact_inventory(
    relative_path: str,
    expected_sha256: str,
    expected_file_count: int,
) -> None:
    manifest = PROJECT_ROOT / relative_path
    assert file_sha256(manifest) == expected_sha256
    assert len(preflight._validate_recorded_artifact_manifest(manifest)) == expected_file_count


def test_condition_runner_resumes_only_valid_bundle_and_requires_both_cache_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Adapter:
        def metadata(self):
            return SimpleNamespace(
                config={"representation_cache": False, "cross_pair_cache": False}
            )

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(pilot, "_new_adapter", Adapter)
    monkeypatch.setattr(pilot, "_validate_integrity_gate_token", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot, "validate_frozen_configuration", lambda **kwargs: {
        "canonical_config_hash": "c",
        "implementation_hash": "i",
        "runtime_identity_hash": "r",
        "decision_rule_hash": "d",
    })
    monkeypatch.setattr(pilot, "validate_pilot_manifest", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(pilot, "file_sha256", lambda path: "sha")

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "result": {"score_payload_sha256": "score"},
            "config_hash": "c",
            "implementation_hash": "i",
        }

    monkeypatch.setattr(pilot, "run_benchmark_manifest", fake_run)
    monkeypatch.setattr(
        pilot,
        "load_bundle",
        lambda **kwargs: SimpleNamespace(rows=[]),
    )
    selection = load_and_verify_selection(SOURCE_PILOT / "selected_identities.csv")
    result = pilot._run_condition(
        project_root=tmp_path,
        data_root=DATA_ROOT,
        dataset="sd300b",
        label="plain_self",
        protocol="plain_self",
        role="self",
        selection=selection,
        preflight={"passed": True},
        integrity_before_validation={
            "validated": True,
            "before_sha256": "before",
            "combined_snapshot_sha256": "combined",
        },
    )
    assert captured["skip_existing"] is True
    assert captured["bundle_directory"] == (
        tmp_path / pilot.PILOT_RELATIVE / "runs/sd300b/plain_self"
    )
    assert captured["dedicated_validator"] is not None
    assert captured["closed"] is True
    assert result["valid_published_bundle_reused_or_created"] is True


def test_generic_cold_pair_executes_two_prepares_even_for_self() -> None:
    pair = PairRecord(
        pair_id="self",
        dataset="sd300b",
        protocol="plain_self",
        subject_id="s",
        canonical_finger_position=1,
        ppi=1000,
        raw_frgp_a=1,
        raw_frgp_b=1,
        path_a=Path("same.png"),
        path_b=Path("same.png"),
    )

    class Adapter:
        def __init__(self) -> None:
            self.prepared: list[PreparedRepresentation] = []
            self.compared = 0

        def prepare(self, path, metadata):
            representation = PreparedRepresentation("m", "v", "f", "rv", object())
            self.prepared.append(representation)
            return PrepareOutcome(representation, diagnostics={"cache": False})

        def compare(self, a, b):
            self.compared += 1
            assert a is not b
            return CompareOutcome(4, diagnostics={"geometric_inlier_count": 4})

    adapter = Adapter()
    spec = BenchmarkRunSpec(
        "sd300b",
        "plain_self",
        Path("manifest.csv"),
        "manifest",
        "m",
        "v",
        BENCHMARK_CONTRACT_VERSION,
        "config",
        "implementation",
    )
    metadata = MethodMetadata(
        "m",
        "v",
        HIGHER_IS_MORE_SIMILAR,
        "integer inlier count",
        {},
        {"representation_cache": False, "cross_pair_cache": False},
    )
    row = _execute_pair(pair, adapter, run_spec=spec, method_metadata=metadata)
    assert row["status"] == "ok"
    assert len(adapter.prepared) == 2
    assert adapter.prepared[0] is not adapter.prepared[1]
    assert adapter.compared == 1


def test_report_has_exact_four_stage_labels_and_six_sections() -> None:
    numeric = {
        "mean": 1.0,
        "median": 1.0,
    }
    summary = {
        "total": 10,
        "accepted": 6,
        "rejected": 4,
        "failures": 0,
        "acceptance_percentage": 60.0,
        "raw_score": numeric,
        "method_compare_ms": numeric,
        "total_pair_ms": numeric,
        "keypoint_count_a": numeric,
        "keypoint_count_b": numeric,
        "mutual_match_count": numeric,
        "geometric_inlier_count": numeric,
    }
    summaries = {
        (dataset, label): dict(summary)
        for dataset in ("sd300b", "sd300c")
        for label in (
            "plain_self",
            "roll_self",
            "plain_roll_genuine",
            "plain_roll_negative",
        )
    }
    survivors = {
        dataset: {"survivor_count": 10, "excluded_count": 0}
        for dataset in ("sd300b", "sd300c")
    }
    sections = reporting._six_sections(summaries, survivors)
    compact = reporting._compact_rows(summaries)
    markdown = reporting._supervisor_markdown(sections, compact, summaries)
    assert [section["number"] for section in sections] == [1, 2, 3, 4, 5, 6]
    assert [row["stage"] for row in compact] == [
        "PLAIN מול עצמו",
        "ROLL מול עצמו",
        "PLAIN מול ROLL המתאים",
        "PLAIN מול ROLL של הנבדק הבא",
    ]
    assert sections[3]["filtering_scope"] == "independent per dataset"
    assert "| שלב | SD300b | SD300c |" in markdown
