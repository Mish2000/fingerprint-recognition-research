from __future__ import annotations

import math

import numpy as np
import pytest

from fingerprint_benchmark.local_features import (
    LocalFeatureRepresentation,
    MatchRecord,
    match_descriptors,
    raw_score,
    score_components,
    verify_geometry,
)
from fingerprint_benchmark.local_features.descriptors import rootsift, validate_descriptors
from fingerprint_benchmark.local_features.detector_only import DetectorOnlyProtocolConfig


def _descriptors(rows: list[tuple[int, float]]) -> np.ndarray:
    output = np.zeros((len(rows), 128), dtype=np.float32)
    for row, (column, value) in enumerate(rows):
        output[row, column] = value
    return output


def _representation(
    points: np.ndarray,
    *,
    ppi: float = 1000.0,
) -> LocalFeatureRepresentation:
    count = len(points)
    return LocalFeatureRepresentation(
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


def test_rootsift_is_l1_then_square_root() -> None:
    descriptors = np.zeros((1, 128), dtype=np.float32)
    descriptors[0, :2] = [1.0, 3.0]
    transformed = rootsift(descriptors)
    assert transformed.dtype == np.float32
    assert transformed[0, 0] == pytest.approx(math.sqrt(0.25))
    assert transformed[0, 1] == pytest.approx(math.sqrt(0.75))


def test_rootsift_handles_zero_norm() -> None:
    transformed = rootsift(np.zeros((1, 128), dtype=np.float32))
    assert np.array_equal(transformed[0], np.zeros(128, dtype=np.float32))
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


def test_lowe_matching_and_mutual_consistency() -> None:
    descriptors_a = _descriptors([(0, 1.0), (1, 1.0), (2, 1.0)])
    descriptors_b = _descriptors([(0, 1.0), (1, 1.0), (3, 1.0), (4, 1.0)])
    one_way = match_descriptors(
        descriptors_a,
        descriptors_b,
        lowe_ratio=0.75,
        matching_mode="one_way",
    )
    mutual = match_descriptors(
        descriptors_a,
        descriptors_b,
        lowe_ratio=0.75,
        matching_mode="mutual",
    )
    expected = [(0, 0), (1, 1)]
    assert [(match.index_a, match.index_b) for match in one_way.submitted] == expected
    assert [(match.index_a, match.index_b) for match in mutual.submitted] == expected
    assert mutual.diagnostics["mutual_match_count"] == 2


def test_bidirectional_union_deduplicates_pairs() -> None:
    descriptors_a = _descriptors([(0, 1.0), (1, 1.0)])
    descriptors_b = _descriptors([(0, 1.0), (1, 1.0), (2, 1.0)])
    result = match_descriptors(
        descriptors_a,
        descriptors_b,
        lowe_ratio=0.75,
        matching_mode="bidirectional_union",
    )
    pairs = [(match.index_a, match.index_b) for match in result.submitted]
    assert pairs == sorted(set(pairs))


def test_partial_affine_geometry_counts_inliers_and_outliers() -> None:
    points_a = np.asarray(
        [[0, 0], [10, 0], [0, 10], [10, 10], [40, 40]],
        dtype=np.float32,
    )
    points_b = np.asarray(
        [[5, 7], [15, 7], [5, 17], [15, 17], [5, 50]],
        dtype=np.float32,
    )
    matches = tuple(MatchRecord(index, index, 0.0, 1.0) for index in range(5))
    config = DetectorOnlyProtocolConfig(
        geometry_model="affine_partial_2d",
        matching_mode="one_way",
        ransac_threshold_reference_px=1.0,
    )
    result = verify_geometry(_representation(points_a), _representation(points_b), matches, config)
    assert result.success
    assert result.inlier_count == 4
    assert result.outlier_count == 1
    assert result.inlier_ratio == pytest.approx(0.8)
    assert result.diagnostics["residual_reference_pixels"]["maximum"] < 1e-4


def test_ppi_normalization_preserves_geometry() -> None:
    points = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    shifted = points + np.asarray([3, 4], dtype=np.float32)
    matches = tuple(MatchRecord(index, index, 0.0, 1.0) for index in range(4))
    config = DetectorOnlyProtocolConfig(matching_mode="one_way")
    first = verify_geometry(
        _representation(points, ppi=1000),
        _representation(shifted, ppi=1000),
        matches,
        config,
    )
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
    matches = tuple(MatchRecord(index, index, 0.0, 1.0) for index in range(2))
    result = verify_geometry(
        _representation(points),
        _representation(points),
        matches,
        DetectorOnlyProtocolConfig(),
    )
    assert not result.success
    assert result.inlier_count == 0
    assert result.outlier_count == 2
    assert result.diagnostics["geometry_failure_reason"] == "insufficient_geometry_matches"


def test_raw_geometric_inlier_score() -> None:
    components = score_components(inliers=12, matches=20, keypoints_a=100, keypoints_b=80)
    assert components["geometric_inlier_count"] == 12.0
    assert raw_score("geometric_inlier_count", components) == 12.0
    assert components["geometric_inlier_ratio"] == pytest.approx(0.6)
    assert components["inliers_over_min_keypoints"] == pytest.approx(0.15)


def test_zero_score_semantics_are_finite() -> None:
    components = score_components(inliers=0, matches=0, keypoints_a=0, keypoints_b=0)
    assert all(value == 0.0 and math.isfinite(value) for value in components.values())
