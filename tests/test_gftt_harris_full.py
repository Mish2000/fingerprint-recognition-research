from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fingerprint_benchmark.cli import parse_args
from fingerprint_benchmark.contract import (
    BenchmarkRunSpec,
    MethodExecutionError,
    PreparedRepresentation,
)
from fingerprint_benchmark.detectors.opencv_gftt_harris import (
    OpenCVGFTTHarrisRootSIFTGeometricAdapter,
)
from fingerprint_benchmark.gftt_harris_full import (
    METHOD_NAME,
    METHOD_VERSION,
    REPRESENTATION_VERSION,
    GFTTHarrisRootSIFTGeometricAdapter,
    GFTTHarrisRootSIFTGeometricConfig,
    frozen_config,
)
from fingerprint_benchmark.gftt_harris_full.parity import (
    compare_result_rows,
    select_sample_indices,
)
from fingerprint_benchmark.gftt_harris_full.restored_equivalence import (
    HISTORICAL_DETECTOR_CONFIG,
    HISTORICAL_PIPELINE_CONFIG,
    PARENT_METHOD_NAME,
    assert_v1_equivalence,
)
from fingerprint_benchmark.local_features.detector_only import representation_sha256
from fingerprint_benchmark.local_features.types import LocalFeatureRepresentation
from fingerprint_benchmark.manifest import read_pair_manifest
from fingerprint_benchmark.runner import (
    prepare_run_context,
    run_benchmark_manifest,
    validate_result_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ALGORITHM_CONFIG_HASH = (
    "5765065c1f3f5238b47fa43746c3f6b2a2b271c6764834bdb9d28f0bcb8fd282"
)


def _checkerboard(size: int = 192, block: int = 8) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return (((xx // block + yy // block) % 2) * 255).astype(np.uint8)


def _write_checkerboard(path: Path) -> Path:
    assert cv2.imwrite(str(path), _checkerboard())
    return path


def _without_timings(value):
    if isinstance(value, dict):
        return {
            key: _without_timings(item)
            for key, item in value.items()
            if not key.endswith("_ms")
        }
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def test_identity_frozen_config_and_hash_are_stable() -> None:
    first = frozen_config()
    second = frozen_config()
    adapter = GFTTHarrisRootSIFTGeometricAdapter()
    metadata = adapter.metadata()

    assert metadata.method == METHOD_NAME == "gftt_harris_rootsift_geometric"
    assert metadata.method_version == METHOD_VERSION == "gftt-harris-rootsift-geometric-v1"
    assert metadata.score_direction == "higher_is_more_similar"
    assert first == second
    assert first.config_hash == second.config_hash == EXPECTED_ALGORITHM_CONFIG_HASH
    assert metadata.config["config_hash"] == EXPECTED_ALGORITHM_CONFIG_HASH
    assert metadata.config["decision_threshold"] is None
    with pytest.raises(FrozenInstanceError):
        first.max_corners = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quality_level", 0.02),
        ("support_size_reference_px", 17.0),
        ("lowe_ratio", 0.70),
        ("ransac_max_iterations", 1999),
        ("opencv_optimized", False),
    ],
)
def test_every_algorithm_change_changes_hash_and_is_rejected_by_v1(field: str, value) -> None:
    changed = replace(frozen_config(), **{field: value})
    assert changed.config_hash != frozen_config().config_hash
    with pytest.raises(ValueError, match="only accepts the frozen"):
        assert_v1_equivalence(changed)
    with pytest.raises(ValueError, match="only accepts the frozen"):
        GFTTHarrisRootSIFTGeometricAdapter(changed)


def test_nonalgorithmic_metadata_does_not_enter_algorithm_hash() -> None:
    config = frozen_config()
    metadata = GFTTHarrisRootSIFTGeometricAdapter(config).metadata()
    mutated_metadata = dict(metadata.implementation_provenance)
    mutated_metadata["research_note"] = "descriptive only"
    assert config.config_hash == EXPECTED_ALGORITHM_CONFIG_HASH


def test_explicit_conversion_matches_historical_joint500_values() -> None:
    config = frozen_config()
    assert config.to_detector_config().as_dict() == HISTORICAL_DETECTOR_CONFIG
    assert config.to_pipeline_config().as_dict() == HISTORICAL_PIPELINE_CONFIG
    assert config.algorithm_config()["detector"]["use_harris_detector"] is True
    assert config.algorithm_config()["detector"]["mask"] is None
    assert config.algorithm_config()["pipeline"]["score_mode"] == "geometric_inlier_count"


def test_full_provenance_distinguishes_components_and_identity() -> None:
    adapter = GFTTHarrisRootSIFTGeometricAdapter()
    metadata = adapter.metadata()
    components = metadata.implementation_provenance["components"]
    paths = {path.relative_to(PROJECT_ROOT).as_posix() for path in adapter.implementation_source_paths()}

    assert metadata.method != PARENT_METHOD_NAME
    assert set(components) == {
        "gftt_harris",
        "orientation",
        "descriptor_extraction",
        "rootsift",
        "matching",
        "geometry",
        "scoring",
        "full_adapter",
        "full_config",
    }
    assert "src/fingerprint_benchmark/gftt_harris_full/adapter.py" in paths
    assert "src/fingerprint_benchmark/detectors/opencv_gftt_harris.py" in paths
    assert "src/fingerprint_benchmark/local_features/geometry.py" in paths


def test_complete_pair_is_exactly_equivalent_to_detector_only(tmp_path: Path) -> None:
    image = _write_checkerboard(tmp_path / "finger.png")
    old = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
    full = GFTTHarrisRootSIFTGeometricAdapter()
    metadata = {"ppi": 1000.0, "pair_id": "pair", "side": "same"}

    old_prepared = old.prepare(image, metadata)
    full_prepared = full.prepare(image, metadata)
    old_comparison = old.compare(old_prepared.representation, old_prepared.representation)
    full_comparison = full.compare(full_prepared.representation, full_prepared.representation)

    assert full_prepared.representation.method == METHOD_NAME
    assert full_prepared.representation.representation_version == REPRESENTATION_VERSION
    assert full_comparison.raw_score > 0.0
    assert full_comparison.raw_score == old_comparison.raw_score
    assert _without_timings(full_prepared.diagnostics) == _without_timings(old_prepared.diagnostics)
    assert _without_timings(full_comparison.diagnostics) == _without_timings(
        old_comparison.diagnostics
    )
    assert full_comparison.diagnostics["decision_threshold_applied"] is False
    assert full_comparison.diagnostics["score_direction"] == "higher_is_more_similar"


def test_single_scale_support_is_ppi_aware_and_border_safe(tmp_path: Path) -> None:
    image = _write_checkerboard(tmp_path / "finger.png")
    adapter = GFTTHarrisRootSIFTGeometricAdapter()
    at_1000 = adapter.prepare(image, {"ppi": 1000.0})
    at_2000 = adapter.prepare(image, {"ppi": 2000.0})

    assert at_1000.diagnostics["native_support_size_px"] == 16.0
    assert at_2000.diagnostics["native_support_size_px"] == 32.0
    assert at_1000.diagnostics["canonical_point_count"] <= 3000
    assert at_1000.diagnostics["detector_diagnostics"]["corner_count"] <= 3000
    assert at_1000.diagnostics["orientation_bins"] == 36
    assert at_1000.diagnostics["gradient_border"] == "BORDER_REFLECT"
    assert at_1000.diagnostics["sift_detector_invoked"] is False
    assert at_1000.representation.payload.descriptors.shape[1] == 128
    assert at_1000.representation.payload.descriptors.dtype == np.float32


def test_valid_zero_score_has_geometry_reason_and_no_threshold() -> None:
    def representation(offset: int) -> PreparedRepresentation:
        descriptors = np.zeros((2, 128), dtype=np.float32)
        descriptors[:, offset : offset + 2] = 1.0
        payload = LocalFeatureRepresentation(
            points=np.asarray([[10, 10], [20, 20]], dtype=np.float32),
            sizes=np.full(2, 16.0, dtype=np.float32),
            angles=np.zeros(2, dtype=np.float32),
            responses=np.zeros(2, dtype=np.float32),
            octaves=np.zeros(2, dtype=np.int32),
            class_ids=np.arange(2, dtype=np.int32),
            descriptors=descriptors,
            width=32,
            height=32,
            ppi=1000.0,
            metadata={},
        )
        return PreparedRepresentation(
            method=METHOD_NAME,
            method_version=METHOD_VERSION,
            representation_format="detector-only-local-features",
            representation_version=REPRESENTATION_VERSION,
            payload=payload,
            metadata={"representation_sha256": representation_sha256(payload)},
        )

    outcome = GFTTHarrisRootSIFTGeometricAdapter().compare(
        representation(0), representation(4)
    )
    assert outcome.raw_score == 0.0
    assert outcome.diagnostics["geometry_failure_reason"] == "insufficient_geometry_matches"
    assert outcome.diagnostics["decision_threshold_applied"] is False
    assert outcome.diagnostics["decision_threshold"] is None


def test_technical_failures_preserve_explicit_error_codes(tmp_path: Path) -> None:
    adapter = GFTTHarrisRootSIFTGeometricAdapter()
    with pytest.raises(MethodExecutionError) as unreadable:
        adapter.prepare(tmp_path / "missing.png", {"ppi": 1000})
    assert unreadable.value.error_code == "image_read_failure"

    image = _write_checkerboard(tmp_path / "finger.png")
    with pytest.raises(MethodExecutionError) as invalid_ppi:
        adapter.prepare(image, {"ppi": 0})
    assert invalid_ppi.value.error_code == "detector_only_preparation_failure"
    assert invalid_ppi.value.method_internal_ms is not None


def test_representation_identity_is_authorized_by_full_adapter(tmp_path: Path) -> None:
    image = _write_checkerboard(tmp_path / "finger.png")
    adapter = GFTTHarrisRootSIFTGeometricAdapter()
    prepared = adapter.prepare(image, {"ppi": 1000})
    foreign = PreparedRepresentation(
        method="foreign",
        method_version=METHOD_VERSION,
        representation_format=prepared.representation.representation_format,
        representation_version=REPRESENTATION_VERSION,
        payload=prepared.representation.payload,
    )
    with pytest.raises(MethodExecutionError) as error:
        adapter.compare(foreign, prepared.representation)
    assert error.value.error_code == "representation_identity_mismatch"


def test_repeatability_of_representations_scores_and_diagnostics(tmp_path: Path) -> None:
    image = _write_checkerboard(tmp_path / "finger.png")
    outcomes = []
    for _ in range(3):
        adapter = GFTTHarrisRootSIFTGeometricAdapter()
        prepared = adapter.prepare(image, {"ppi": 1000})
        comparison = adapter.compare(prepared.representation, prepared.representation)
        outcomes.append(
            (
                prepared.diagnostics["representation_sha256"],
                comparison.raw_score,
                _without_timings(prepared.diagnostics),
                _without_timings(comparison.diagnostics),
            )
        )
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_full_adapter_publishes_and_validates_pairwise_v2_bundle(tmp_path: Path) -> None:
    image = _write_checkerboard(tmp_path / "finger.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "full-pair",
                "dataset": "synthetic",
                "protocol": "self",
                "subject_id": "00000001",
                "canonical_finger_position": 1,
                "ppi": 1000,
                "raw_frgp_a": 11,
                "raw_frgp_b": 11,
                "path_a": image,
                "path_b": image,
            }
        )
    adapter = GFTTHarrisRootSIFTGeometricAdapter()
    results_root = tmp_path / "results"
    metadata = run_benchmark_manifest(
        manifest_path=manifest,
        adapter=adapter,
        expected_dataset="synthetic",
        expected_protocol="self",
        results_root=results_root,
        data_root=tmp_path,
        dedicated_validator=lambda _manifest, _data: {"status": "ok"},
    )
    context = prepare_run_context(
        manifest_path=manifest,
        expected_dataset="synthetic",
        expected_protocol="self",
        adapter=adapter,
        results_root=results_root,
    )
    validated = validate_result_bundle(
        Path(metadata["result"]["path"]).parent,
        manifest_records=read_pair_manifest(manifest),
        run_spec=context.spec,
        score_direction=context.method_metadata.score_direction,
        score_semantics=context.method_metadata.score_semantics,
    )
    assert metadata["method"] == METHOD_NAME
    assert metadata["success_count"] == 1
    assert validated["result"]["row_count"] == 1


def test_cli_registers_smoke_run_and_parity_commands(tmp_path: Path) -> None:
    smoke = parse_args(
        [
            "gftt-harris-smoke",
            "--image-a",
            str(tmp_path / "a.png"),
            "--image-b",
            str(tmp_path / "b.png"),
            "--ppi-a",
            "1000",
            "--ppi-b",
            "2000",
        ]
    )
    run = parse_args(["run-gftt-harris", "--dataset", "sd300b", "--protocol", "plain_self"])
    parity = parse_args(["gftt-harris-parity"])
    assert smoke.command == "gftt-harris-smoke"
    assert run.command == "run-gftt-harris"
    assert parity.command == "gftt-harris-parity"


def test_parity_selection_has_ten_stride_pairs_and_edge_coverage() -> None:
    rows = []
    for index in range(100):
        diagnostics = {
            "matches_submitted_to_geometry": 2 if index == 7 else 5,
            "geometry_failure_reason": "insufficient_geometry_matches" if index == 7 else None,
        }
        rows.append(
            {
                "status": "ok",
                "raw_score": "0.0" if index == 7 else repr(float(index + 1)),
                "error_code": "",
                "compare_diagnostics": json.dumps(diagnostics),
            }
        )
    selected = select_sample_indices(rows, pair_kind="plain_roll_impostor")
    assert all((i * 100) // 10 in selected for i in range(10))
    reasons = {reason for values in selected.values() for reason in values}
    assert {"valid_zero_score", "few_matches", "geometry_failure"}.issubset(reasons)
    assert "highest_historical_score" in reasons
    assert "historical_positive_impostor" in reasons


def test_parity_comparison_is_exact_and_ignores_only_timings() -> None:
    base = {
        "status": "ok",
        "raw_score": "4.0",
        "error_code": "",
        "error_message": "",
        "prepare_a_diagnostics": json.dumps({"count": 3, "prepare_total_ms": 1.0}),
        "prepare_b_diagnostics": json.dumps({"count": 3, "prepare_total_ms": 2.0}),
        "compare_diagnostics": json.dumps(
            {"estimated_transform": [[1.0, 0.0, 2.0]], "compare_total_ms": 1.0}
        ),
    }
    timing_only = dict(base)
    timing_only["compare_diagnostics"] = json.dumps(
        {"estimated_transform": [[1.0, 0.0, 2.0]], "compare_total_ms": 99.0}
    )
    changed = dict(timing_only)
    changed["compare_diagnostics"] = json.dumps(
        {"estimated_transform": [[1.0, 0.0, 3.0]], "compare_total_ms": 99.0}
    )
    assert compare_result_rows(base, timing_only) == []
    assert compare_result_rows(base, changed)[0]["field"] == "compare_diagnostics"
