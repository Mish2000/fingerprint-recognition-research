"""Frozen-identity tests for the restored SIFT geometric baseline.

These assert the restoration reproduces the *historical* method, not merely that
it executes.  The anchors are the values recorded in the historical bundles'
``run_metadata.json``: the frozen config, its ``config_hash``, the score
formula, and the failure/zero-score split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fingerprint_benchmark.cli import parse_args
from fingerprint_benchmark.contract import (
    BENCHMARK_CONTRACT_VERSION,
    HIGHER_IS_MORE_SIMILAR,
    TIMING_MODE_COLD_PAIR,
    WARMUP_POLICY,
    MethodExecutionError,
)
from fingerprint_benchmark.hashing import stable_config_hash
from fingerprint_benchmark.provenance import implementation_provenance
from fingerprint_benchmark.sift.config import METHOD_NAME, METHOD_VERSION
from fingerprint_benchmark.sift.matching import MatchRecord
from fingerprint_benchmark.sift.geometry import verify_geometry
from fingerprint_benchmark.sift.parity import (
    deterministic_diagnostics,
    is_deterministic_diagnostic_key,
    select_sample_indices,
)
from fingerprint_benchmark.sift.restored import (
    HISTORICAL_CONFIG_HASH,
    HISTORICAL_MODULE_FILENAMES,
    HISTORICAL_MODULE_SHA256,
    HISTORICAL_SOURCE_COMMIT,
    RestoredSiftGeometricAdapter,
    frozen_config,
    restoration_provenance,
)


SIFT_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "fingerprint_benchmark" / "sift"

#: Exactly the config body every historical bundle recorded, minus the keys the
#: runner injects.  Copied from run_metadata.json of the six historical bundles.
HISTORICAL_CONFIG_BODY = {
    "contrast_threshold": 0.04,
    "descriptor_mode": "rootsift",
    "edge_threshold": 10.0,
    "geometry_model": "affine_partial_2d",
    "image_policy": "native",
    "lowe_ratio": 0.75,
    "mask_mode": "none",
    "matching_mode": "mutual",
    "minimum_descriptors": 2,
    "minimum_geometry_matches": 3,
    "n_octave_layers": 3,
    "nfeatures": 3000,
    "normalize_coordinates_by_ppi": True,
    "opencv_optimized": True,
    "opencv_threads": 16,
    "ransac_confidence": 0.99,
    "ransac_max_iterations": 2000,
    "ransac_refine_iterations": 10,
    "ransac_threshold_at_reference_ppi": 3.0,
    "reference_clahe_clip": 2.0,
    "reference_clahe_grid_x": 8,
    "reference_clahe_grid_y": 8,
    "reference_ppi": 1000.0,
    "reference_target_size": 768,
    "rng_seed": 0,
    "schema_version": "sift-geometric-config-schema-v1",
    "score_mode": "geometric_inlier_count",
    "sigma": 1.6,
    "valid_region_black_threshold": 10,
    "valid_region_close_kernel_at_reference_ppi": 31,
    "valid_region_erode_at_reference_ppi": 5,
    "valid_region_min_coverage": 0.01,
}

HISTORICAL_SCORE_SEMANTICS = (
    "OpenCV SIFT L2 Lowe matches verified by PPI-normalized affine RANSAC; raw score is "
    "geometric_inlier_count with no acceptance thresholding in compare()."
)


def _runner_config(adapter: RestoredSiftGeometricAdapter) -> dict[str, object]:
    """Rebuild exactly the dict runner.prepare_run_context hashes."""

    metadata = adapter.metadata()
    return {
        **metadata.config,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "method": metadata.method,
        "method_version": metadata.method_version,
        "score_direction": metadata.score_direction,
        "score_semantics": metadata.score_semantics,
        "timing_mode": TIMING_MODE_COLD_PAIR,
        "warm_up_policy": WARMUP_POLICY,
    }


def test_restored_modules_are_byte_identical_to_the_historical_source() -> None:
    for name in HISTORICAL_MODULE_FILENAMES:
        digest = hashlib.sha256((SIFT_PACKAGE / name).read_bytes()).hexdigest()
        assert digest == HISTORICAL_MODULE_SHA256[name], f"{name} drifted from the historical source"


def test_frozen_config_matches_the_historical_bundle_config_exactly() -> None:
    assert frozen_config().as_dict() == HISTORICAL_CONFIG_BODY


def test_frozen_config_does_not_rely_on_dataclass_defaults() -> None:
    """The two fields the historical pilot moved away from the defaults."""

    config = frozen_config()
    assert config.descriptor_mode == "rootsift"
    assert config.score_mode == "geometric_inlier_count"


def test_config_hash_reproduces_the_historical_bundle_identity() -> None:
    adapter = RestoredSiftGeometricAdapter()
    try:
        assert stable_config_hash(_runner_config(adapter)) == HISTORICAL_CONFIG_HASH
    finally:
        adapter.close()


def test_method_identity_and_score_direction_are_frozen() -> None:
    adapter = RestoredSiftGeometricAdapter()
    try:
        metadata = adapter.metadata()
        assert (metadata.method, metadata.method_version) == ("sift_geometric", "sift-geometric-v1")
        assert (METHOD_NAME, METHOD_VERSION) == ("sift_geometric", "sift-geometric-v1")
        assert metadata.score_direction == HIGHER_IS_MORE_SIMILAR
        assert metadata.score_semantics == HISTORICAL_SCORE_SEMANTICS
        assert metadata.config["thresholding"] == "none_in_compare"
        assert metadata.config["representation_cache"] is False
    finally:
        adapter.close()


def test_restored_adapter_does_not_override_algorithm_behaviour() -> None:
    """The compatibility layer may only add provenance, never change scoring."""

    for name in ("metadata", "prepare", "compare"):
        assert name not in vars(RestoredSiftGeometricAdapter), (
            f"RestoredSiftGeometricAdapter must not override {name}()"
        )


def test_adapter_declares_every_restored_source_for_provenance() -> None:
    adapter = RestoredSiftGeometricAdapter()
    try:
        declared = {path.name for path in adapter.implementation_source_paths()}
    finally:
        adapter.close()
    assert set(HISTORICAL_MODULE_FILENAMES) | {"restored.py"} == declared


def test_implementation_provenance_records_a_stable_component_hash() -> None:
    adapter = RestoredSiftGeometricAdapter()
    try:
        runner_source = (
            Path(__file__).resolve().parents[1] / "src" / "fingerprint_benchmark" / "runner.py"
        )
        first, components, implementation_hash = implementation_provenance(
            adapter=adapter,
            method_metadata=adapter.metadata(),
            startup_validation={},
            runner_source_path=runner_source,
        )
        _, _, again = implementation_provenance(
            adapter=adapter,
            method_metadata=adapter.metadata(),
            startup_validation={},
            runner_source_path=runner_source,
        )
    finally:
        adapter.close()
    declared = first["adapter_declared_implementation_sources"]
    assert declared["component_sha256"] == components[
        "adapter_declared_implementation_sources"
    ]["component_sha256"]
    assert implementation_hash == again, "implementation hash must be deterministic"
    by_path = {entry["path"]: entry["sha256"] for entry in declared["files"]}
    for name, digest in HISTORICAL_MODULE_SHA256.items():
        assert by_path[f"src/fingerprint_benchmark/sift/{name}"] == digest


def test_restoration_provenance_separates_historical_and_restored_identity() -> None:
    payload = restoration_provenance()
    assert payload["restored_from_commit"] == HISTORICAL_SOURCE_COMMIT
    assert payload["historical_method_name"] == "sift_geometric"
    assert payload["historical_method_version"] == "sift-geometric-v1"
    assert payload["historical_config_hash"] == HISTORICAL_CONFIG_HASH
    assert "improved" not in json.dumps(payload).lower()


def test_score_is_the_raw_geometric_inlier_count_with_no_threshold() -> None:
    """The frozen score mode is the inlier count itself, not the composite."""

    from fingerprint_benchmark.sift.scoring import raw_score, score_components

    components = score_components(inliers=7, matches=19, keypoints_a=500, keypoints_b=400)
    assert raw_score("geometric_inlier_count", components) == 7.0


def test_zero_inliers_is_an_ok_zero_score_not_a_technical_failure() -> None:
    from fingerprint_benchmark.sift.scoring import raw_score, score_components

    components = score_components(inliers=0, matches=12, keypoints_a=300, keypoints_b=300)
    assert raw_score("geometric_inlier_count", components) == 0.0


def test_insufficient_matches_reports_zero_with_an_explicit_reason() -> None:
    from fingerprint_benchmark.sift.extractor import SiftRepresentation

    def representation(points: np.ndarray) -> SiftRepresentation:
        count = len(points)
        return SiftRepresentation(
            points=np.asarray(points, dtype=np.float32),
            sizes=np.ones(count, dtype=np.float32),
            angles=np.zeros(count, dtype=np.float32),
            responses=np.ones(count, dtype=np.float32),
            octaves=np.zeros(count, dtype=np.int32),
            class_ids=np.full(count, -1, dtype=np.int32),
            descriptors=np.zeros((count, 128), dtype=np.float32),
            width=100,
            height=100,
            ppi=1000.0,
            metadata={},
        )

    points = np.asarray([[0, 0], [1, 1]], dtype=np.float32)
    matches = tuple(MatchRecord(i, i, 0.0, 1.0) for i in range(2))
    result = verify_geometry(representation(points), representation(points), matches, frozen_config())
    assert not result.success
    assert result.inlier_count == 0
    assert result.diagnostics["geometry_failure_reason"] == "insufficient_geometry_matches"


def test_unreadable_image_is_a_technical_failure_with_an_error_code(tmp_path: Path) -> None:
    adapter = RestoredSiftGeometricAdapter()
    try:
        with pytest.raises(MethodExecutionError) as excinfo:
            adapter.prepare(tmp_path / "absent.png", {"ppi": 1000})
    finally:
        adapter.close()
    assert excinfo.value.error_code == "image_read_failure"


def test_invalid_ppi_is_a_technical_failure(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "flat.png"
    assert cv2.imwrite(str(path), np.full((64, 64), 128, dtype=np.uint8))
    adapter = RestoredSiftGeometricAdapter()
    try:
        with pytest.raises(MethodExecutionError) as excinfo:
            adapter.prepare(path, {"ppi": 0})
    finally:
        adapter.close()
    assert excinfo.value.error_code == "missing_or_invalid_ppi"


def test_parity_sample_rule_is_the_frozen_stride_plus_outcome_coverage() -> None:
    rows = [
        {"pair_id": f"p{i}", "status": "ok", "raw_score": "5.0"} for i in range(100)
    ]
    rows[42] = {"pair_id": "p42", "status": "ok", "raw_score": "0.0"}
    rows[77] = {"pair_id": "p77", "status": "prepare_a_failure", "raw_score": ""}
    selected = select_sample_indices(rows)
    assert [index for index, reason in sorted(selected.items()) if reason.startswith("stride_")] == [
        (i * 100) // 10 for i in range(10)
    ]
    assert selected[42] == "zero_score_ok"
    assert selected[77] == "technical_failure"


def test_parity_excludes_only_timing_diagnostics() -> None:
    assert is_deterministic_diagnostic_key("geometric_inlier_count")
    assert is_deterministic_diagnostic_key("geometry_failure_reason")
    assert not is_deterministic_diagnostic_key("matching_ms")
    assert not is_deterministic_diagnostic_key("sift_extraction_ms")
    payload = json.dumps({"geometric_inlier_count": 3, "matching_ms": 1.5})
    assert deterministic_diagnostics(payload) == {"geometric_inlier_count": 3}


def _synthetic_fingerprint(path: Path, *, shift: int) -> None:
    import cv2

    image = np.full((256, 256), 235, dtype=np.uint8)
    for radius in range(18, 110, 9):
        cv2.circle(image, (128 + shift, 128), radius, 40 + radius, 2)
    cv2.line(image, (30 + shift, 24), (226, 214), 20, 3)
    assert cv2.imwrite(str(path), image)


def test_restored_adapter_produces_a_valid_benchmark_v2_bundle(tmp_path: Path) -> None:
    """End-to-end through the current runner, contract and bundle validator."""

    from fingerprint_benchmark.contract import BenchmarkRunSpec
    from fingerprint_benchmark.manifest import MANIFEST_COLUMNS, read_pair_manifest
    from fingerprint_benchmark.runner import run_benchmark_manifest, validate_result_bundle

    image_a, image_b = tmp_path / "a.png", tmp_path / "b.png"
    _synthetic_fingerprint(image_a, shift=0)
    _synthetic_fingerprint(image_b, shift=3)

    manifest_path = tmp_path / "manifest.csv"
    rows = [
        ",".join(MANIFEST_COLUMNS),
        f"sd300b_plain_self_00001000_01,sd300b,plain_self,00001000,1,1000,11,11,{image_a},{image_b}",
        f"sd300b_plain_self_00001001_02,sd300b,plain_self,00001001,2,1000,12,12,{image_a},{image_a}",
    ]
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")

    adapter = RestoredSiftGeometricAdapter()
    try:
        metadata = run_benchmark_manifest(
            manifest_path=manifest_path,
            adapter=adapter,
            expected_dataset="sd300b",
            expected_protocol="plain_self",
            results_root=tmp_path / "results",
            data_root=tmp_path,
            dedicated_validator=lambda _manifest, _data_root: {"status": "ok"},
        )
    finally:
        adapter.close()

    assert metadata["method"] == "sift_geometric"
    assert metadata["method_version"] == "sift-geometric-v1"
    assert metadata["config_hash"] == HISTORICAL_CONFIG_HASH
    assert metadata["score_direction"] == HIGHER_IS_MORE_SIMILAR
    assert metadata["score_semantics"] == HISTORICAL_SCORE_SEMANTICS
    assert metadata["success_count"] == 2

    bundle = Path(metadata["result"]["path"]).parent
    run_spec = BenchmarkRunSpec(
        expected_dataset="sd300b",
        expected_protocol="plain_self",
        manifest_path=manifest_path.resolve(),
        manifest_sha256=metadata["manifest"]["sha256"],
        method=metadata["method"],
        method_version=metadata["method_version"],
        benchmark_contract_version=BENCHMARK_CONTRACT_VERSION,
        config_hash=metadata["config_hash"],
        implementation_hash=metadata["implementation_hash"],
    )
    validated = validate_result_bundle(
        bundle,
        manifest_records=read_pair_manifest(manifest_path),
        run_spec=run_spec,
        score_direction=HIGHER_IS_MORE_SIMILAR,
        score_semantics=HISTORICAL_SCORE_SEMANTICS,
    )
    assert validated["result"]["sha256"] == metadata["result"]["sha256"]


def test_cli_registers_the_restored_method_without_touching_detector_only() -> None:
    smoke = parse_args(["sift-geometric-smoke", "--manifest", "protocols/sd300b/plain_self.csv"])
    assert smoke.command == "sift-geometric-smoke"
    run = parse_args(["run-sift-geometric", "--dataset", "sd300b", "--protocol", "plain_roll"])
    assert (run.command, run.dataset, run.protocol) == ("run-sift-geometric", "sd300b", "plain_roll")
    parity = parse_args(["sift-geometric-parity", "--historical-results-root", "hist/results"])
    assert parity.command == "sift-geometric-parity"
    joint = parse_args(["detector-joint500", "validate"])
    assert (joint.command, joint.joint_phase) == ("detector-joint500", "validate")
