from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from fingerprint_benchmark.contract import MethodExecutionError, PreparedRepresentation
from fingerprint_benchmark.sift.adapter import SiftGeometricAdapter
from fingerprint_benchmark.sift.config import (
    METHOD_NAME,
    METHOD_VERSION,
    SiftGeometricConfig,
)
from fingerprint_benchmark.sift.descriptors import rootsift, validate_descriptors
from fingerprint_benchmark.sift.extractor import SiftRepresentation
from fingerprint_benchmark.sift.geometry import verify_geometry
from fingerprint_benchmark.sift.matching import MatchRecord, match_descriptors
from fingerprint_benchmark.sift.preprocessing import prepare_image, valid_region_mask
from fingerprint_benchmark.sift.scoring import raw_score, score_components


def _descriptors(rows: list[tuple[int, float]]) -> np.ndarray:
    output = np.zeros((len(rows), 128), dtype=np.float32)
    for row, (column, value) in enumerate(rows):
        output[row, column] = value
    return output


def _representation(points: np.ndarray, *, ppi: float = 1000.0) -> SiftRepresentation:
    count = len(points)
    return SiftRepresentation(
        points=np.asarray(points, dtype=np.float32),
        sizes=np.ones(count, dtype=np.float32),
        angles=np.zeros(count, dtype=np.float32),
        responses=np.ones(count, dtype=np.float32),
        octaves=np.zeros(count, dtype=np.int32),
        class_ids=np.full(count, -1, dtype=np.int32),
        descriptors=np.zeros((count, 128), dtype=np.float32),
        width=1000,
        height=1000,
        ppi=ppi,
        metadata={},
    )


def test_public_method_identity_is_single_and_exact() -> None:
    assert METHOD_NAME == "sift_geometric"
    assert METHOD_VERSION == "sift-geometric-v1"
    metadata = SiftGeometricAdapter().metadata()
    assert metadata.method == METHOD_NAME
    assert metadata.method_version == METHOD_VERSION
    assert metadata.score_direction == "higher_is_more_similar"
    assert metadata.config["thresholding"] == "none_in_compare"
    assert metadata.config["representation_cache"] is False


def test_config_rejects_non_normalized_native_geometry() -> None:
    with pytest.raises(ValueError, match="normalize"):
        SiftGeometricConfig(normalize_coordinates_by_ppi=False)


def test_rootsift_is_l1_then_square_root_and_handles_zero_norm() -> None:
    descriptors = np.zeros((2, 128), dtype=np.float32)
    descriptors[0, :2] = [1.0, 3.0]
    transformed = rootsift(descriptors)
    assert transformed.dtype == np.float32
    assert transformed[0, 0] == pytest.approx(math.sqrt(0.25))
    assert transformed[0, 1] == pytest.approx(math.sqrt(0.75))
    assert np.array_equal(transformed[1], np.zeros(128, dtype=np.float32))
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 64), dtype=np.float32),
        np.zeros((2, 128), dtype=np.int32),
        np.full((2, 128), np.nan, dtype=np.float32),
    ],
)
def test_descriptor_validation_rejects_shape_dtype_and_nonfinite(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_descriptors(bad)


def test_ratio_matching_and_mutual_consistency() -> None:
    a = _descriptors([(0, 1.0), (1, 1.0), (2, 1.0)])
    b = _descriptors([(0, 1.0), (1, 1.0), (3, 1.0), (4, 1.0)])
    one_way = match_descriptors(a, b, lowe_ratio=0.75, matching_mode="one_way")
    mutual = match_descriptors(a, b, lowe_ratio=0.75, matching_mode="mutual")
    assert [(m.index_a, m.index_b) for m in one_way.submitted] == [(0, 0), (1, 1)]
    assert [(m.index_a, m.index_b) for m in mutual.submitted] == [(0, 0), (1, 1)]
    assert mutual.diagnostics["raw_knn_count_a_to_b"] == 3
    assert mutual.diagnostics["raw_knn_count_b_to_a"] == 4
    assert mutual.diagnostics["mutual_match_count"] == 2


def test_bidirectional_union_deduplicates_pairs() -> None:
    a = _descriptors([(0, 1.0), (1, 1.0)])
    b = _descriptors([(0, 1.0), (1, 1.0), (2, 1.0)])
    result = match_descriptors(a, b, lowe_ratio=0.75, matching_mode="bidirectional_union")
    pairs = [(match.index_a, match.index_b) for match in result.submitted]
    assert pairs == sorted(set(pairs))


def test_partial_affine_geometry_counts_inliers_outliers_and_residuals() -> None:
    points_a = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10], [40, 40]], dtype=np.float32)
    points_b = np.asarray([[5, 7], [15, 7], [5, 17], [15, 17], [5, 50]], dtype=np.float32)
    matches = tuple(MatchRecord(i, i, 0.0, 1.0) for i in range(5))
    config = SiftGeometricConfig(
        geometry_model="affine_partial_2d",
        matching_mode="one_way",
        ransac_threshold_at_reference_ppi=1.0,
    )
    result = verify_geometry(_representation(points_a), _representation(points_b), matches, config)
    assert result.success
    assert result.inlier_count == 4
    assert result.outlier_count == 1
    assert result.inlier_ratio == pytest.approx(0.8)
    assert result.diagnostics["residual_reference_pixels"]["maximum"] < 1e-4


def test_ppi_normalization_produces_same_geometry_for_doubled_coordinates() -> None:
    points = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    shifted = points + np.asarray([3, 4], dtype=np.float32)
    matches = tuple(MatchRecord(i, i, 0.0, 1.0) for i in range(4))
    config = SiftGeometricConfig(geometry_model="affine_partial_2d", matching_mode="one_way")
    first = verify_geometry(_representation(points, ppi=1000), _representation(shifted, ppi=1000), matches, config)
    second = verify_geometry(
        _representation(points * 2, ppi=2000),
        _representation(shifted * 2, ppi=2000),
        matches,
        config,
    )
    assert first.inlier_count == second.inlier_count == 4
    assert np.allclose(first.transform, second.transform)


def test_geometry_failure_retains_submitted_match_accounting() -> None:
    points = np.asarray([[0, 0], [1, 1]], dtype=np.float32)
    matches = tuple(MatchRecord(i, i, 0.0, 1.0) for i in range(2))
    result = verify_geometry(_representation(points), _representation(points), matches, SiftGeometricConfig())
    assert not result.success
    assert result.inlier_count == 0
    assert result.outlier_count == 2
    assert result.diagnostics["geometry_failure_reason"] == "insufficient_geometry_matches"


def test_score_formula_components_and_direction() -> None:
    components = score_components(inliers=12, matches=20, keypoints_a=100, keypoints_b=80)
    expected = 12.0 * (12.0 / 20.0) * np.log1p(20.0)
    assert components["geometric_inlier_count"] == 12.0
    assert components["geometric_inlier_ratio"] == pytest.approx(0.6)
    assert components["inliers_over_min_keypoints"] == pytest.approx(0.15)
    assert components["inliers_times_inlier_ratio_times_log1p_matches"] == pytest.approx(expected)
    assert raw_score("inliers_times_inlier_ratio_times_log1p_matches", components) == pytest.approx(expected)
    better = score_components(inliers=13, matches=20, keypoints_a=100, keypoints_b=80)
    assert better["inliers_times_inlier_ratio_times_log1p_matches"] > expected


def test_zero_score_semantics_are_finite() -> None:
    components = score_components(inliers=0, matches=0, keypoints_a=0, keypoints_b=0)
    assert all(value == 0.0 and math.isfinite(value) for value in components.values())


def test_valid_region_mask_excludes_black_frame() -> None:
    image = np.full((200, 160), 220, dtype=np.uint8)
    image[:20, :] = 0
    image[-10:, :] = 0
    config = SiftGeometricConfig(mask_mode="valid_region")
    mask, metadata = valid_region_mask(image, 1000, config)
    assert metadata["mask_status"] == "ok"
    assert np.all(mask[:10] == 0)
    assert mask[100, 80] == 255
    assert 0.0 < metadata["mask_coverage_ratio"] < 1.0


def test_native_image_policy_does_not_resize_or_enhance() -> None:
    image = np.arange(120 * 80, dtype=np.uint8).reshape(120, 80)
    prepared = prepare_image(image, 1000, SiftGeometricConfig(mask_mode="none"))
    assert prepared.image.shape == image.shape
    assert np.array_equal(prepared.image, image)
    assert prepared.metadata["enhancement"] == "none"
    assert prepared.metadata["resize_scale"] == 1.0


def test_adapter_fails_explicitly_when_image_cannot_be_loaded(tmp_path: Path) -> None:
    with pytest.raises(MethodExecutionError) as exc_info:
        SiftGeometricAdapter().prepare(tmp_path / "missing.png", {"ppi": 1000})
    assert exc_info.value.error_code == "image_read_failure"


def test_adapter_rejects_representation_from_other_method() -> None:
    representation = PreparedRepresentation(
        method="other",
        method_version="other-v1",
        representation_format="x",
        representation_version="x",
        payload={},
    )
    with pytest.raises(MethodExecutionError, match="another method"):
        SiftGeometricAdapter().compare(representation, representation)


def test_real_sift_representation_is_deterministic_and_preserves_keypoint_fields(tmp_path: Path) -> None:
    image = np.full((256, 256), 240, dtype=np.uint8)
    for radius in range(20, 100, 10):
        cv2.circle(image, (128, 128), radius, 40 + radius, 2)
    cv2.line(image, (40, 30), (220, 210), 20, 3)
    path = tmp_path / "finger.png"
    assert cv2.imwrite(str(path), image)
    adapter = SiftGeometricAdapter(SiftGeometricConfig(nfeatures=300, opencv_threads=1))
    first = adapter.prepare(path, {"ppi": 1000, "pair_id": "x", "side": "a"})
    second = adapter.prepare(path, {"ppi": 1000, "pair_id": "x", "side": "a"})
    a = first.representation.payload
    b = second.representation.payload
    assert a.keypoint_count > 2
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.sizes, b.sizes)
    assert np.array_equal(a.angles, b.angles)
    assert np.array_equal(a.responses, b.responses)
    assert np.array_equal(a.octaves, b.octaves)
    assert np.array_equal(a.class_ids, b.class_ids)
    assert np.array_equal(a.descriptors, b.descriptors)
    comparison = adapter.compare(first.representation, second.representation)
    assert comparison.raw_score > 0.0
    assert comparison.diagnostics["geometric_inlier_count"] <= comparison.diagnostics[
        "matches_submitted_to_geometry"
    ]
    assert comparison.diagnostics["geometric_outlier_count"] == (
        comparison.diagnostics["matches_submitted_to_geometry"]
        - comparison.diagnostics["geometric_inlier_count"]
    )
    assert comparison.method_internal_ms is not None and comparison.method_internal_ms >= 0.0


def test_opencv_runtime_provenance_is_explicit() -> None:
    metadata = SiftGeometricAdapter().metadata()
    assert metadata.implementation_provenance["opencv_version"] == cv2.__version__
    assert metadata.implementation_provenance["opencv_distribution"]["name"] == "opencv-python"
    assert metadata.runtime["opencv_thread_count"] == 16
    assert isinstance(metadata.runtime["opencv_build_information"], str)
    assert metadata.implementation_provenance["sift_constructor_parameters"] == {
        "nfeatures": 3000,
        "nOctaveLayers": 3,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10.0,
        "sigma": 1.6,
    }
