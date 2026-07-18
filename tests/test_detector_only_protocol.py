from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fingerprint_benchmark.local_features import scoring as common_scoring
from fingerprint_benchmark.contract import PreparedRepresentation
from fingerprint_benchmark.detectors import DetectedPoint, DetectorResult
from fingerprint_benchmark.local_features import geometry as common_geometry
from fingerprint_benchmark.local_features import matching as common_matching
from fingerprint_benchmark.local_features.descriptors import rootsift as common_rootsift
from fingerprint_benchmark.local_features.detector_only import (
    DetectorOnlyAdapter,
    DetectorOnlyProtocolConfig,
    REPRESENTATION_VERSION,
    build_representation,
)


class FakeDetector:
    def __init__(
        self,
        points: tuple[DetectedPoint, ...],
        *,
        name: str = "fake",
        version: str = "fake-v1",
    ) -> None:
        self.points = points
        self.detector_name = name
        self.detector_version = version
        self.config = {"fixture": True}

    def detect(
        self,
        image: np.ndarray,
        image_metadata: dict[str, object],
        mask: np.ndarray | None = None,
    ) -> DetectorResult:
        return DetectorResult(
            points=self.points,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            detector_config=self.config,
            diagnostics={"mask": mask is not None},
            detector_time_ms=0.0,
            metadata={"image_metadata": dict(image_metadata)},
        )


def _textured_image(size: int = 128) -> np.ndarray:
    yy, xx = np.indices((size, size))
    checker = ((xx // 8 + yy // 8) % 2) * 180
    ramp = (3 * xx + 5 * yy) % 75
    return np.asarray(checker + ramp, dtype=np.uint8)


def _locations() -> tuple[tuple[float, float], ...]:
    return (
        (24.0, 24.0),
        (48.0, 24.0),
        (72.0, 24.0),
        (24.0, 48.0),
        (48.0, 48.0),
        (72.0, 48.0),
    )


def _result(points: tuple[DetectedPoint, ...], name: str) -> DetectorResult:
    return FakeDetector(points, name=name).detect(
        _textured_image(),
        {"ppi": 1000.0},
    )


def test_native_scale_angle_and_response_do_not_change_the_representation() -> None:
    first_points = tuple(
        DetectedPoint(x, y, response=float(index), detector_scale=1.0, detector_angle=0.0)
        for index, (x, y) in enumerate(_locations())
    )
    second_points = tuple(
        DetectedPoint(
            x,
            y,
            response=float(100 - index),
            detector_scale=20.0 + index,
            detector_angle=173.0,
        )
        for index, (x, y) in enumerate(_locations())
    )
    config = DetectorOnlyProtocolConfig(maximum_keypoints=20)
    first, first_diagnostics, _ = build_representation(
        _textured_image(), {"ppi": 1000.0}, _result(first_points, "first"), config
    )
    second, second_diagnostics, _ = build_representation(
        _textured_image(), {"ppi": 1000.0}, _result(second_points, "second"), config
    )
    for field in (
        "points",
        "sizes",
        "angles",
        "responses",
        "octaves",
        "class_ids",
        "descriptors",
    ):
        assert np.array_equal(getattr(first, field), getattr(second, field))
    assert first_diagnostics["representation_sha256"] == second_diagnostics[
        "representation_sha256"
    ]
    assert first_diagnostics["detector_scale_used"] is False
    assert second_diagnostics["detector_angle_used"] is False


def test_changing_a_location_changes_the_common_representation() -> None:
    original = tuple(DetectedPoint(x, y, 1.0) for x, y in _locations())
    moved = (DetectedPoint(28.0, 24.0, 1.0), *original[1:])
    config = DetectorOnlyProtocolConfig(maximum_keypoints=20)
    first, first_diagnostics, _ = build_representation(
        _textured_image(), {"ppi": 1000.0}, _result(original, "first"), config
    )
    second, second_diagnostics, _ = build_representation(
        _textured_image(), {"ppi": 1000.0}, _result(tuple(moved), "second"), config
    )
    assert not np.array_equal(first.points, second.points)
    assert first_diagnostics["representation_sha256"] != second_diagnostics[
        "representation_sha256"
    ]


def test_every_point_uses_one_physical_support_and_orientation_policy() -> None:
    points = tuple(DetectedPoint(x, y, 1.0) for x, y in _locations())
    config = DetectorOnlyProtocolConfig(
        reference_ppi=1000.0,
        support_size_reference_px=16.0,
        maximum_keypoints=20,
    )
    representation, diagnostics, _ = build_representation(
        _textured_image(), {"ppi": 2000.0}, _result(points, "fake"), config
    )
    assert np.array_equal(
        representation.sizes,
        np.full(representation.keypoint_count, 32.0, dtype=np.float32),
    )
    assert diagnostics["orientation_count"] == representation.keypoint_count
    assert diagnostics["orientation_policy"] == "common_dominant_gradient_v1"
    assert representation.metadata["orientation_policy"] == "common_dominant_gradient_v1"
    assert diagnostics["support_policy"] == "fixed_physical_diameter_scaled_by_manifest_ppi"


def test_adapter_compare_exposes_raw_score_without_decision_threshold() -> None:
    image = _textured_image()
    points = tuple(DetectedPoint(x, y, 1.0) for x, y in _locations())
    detector = FakeDetector(points)
    config = DetectorOnlyProtocolConfig(maximum_keypoints=20)
    representation, _, _ = build_representation(
        image,
        {"ppi": 1000.0},
        detector.detect(image, {"ppi": 1000.0}),
        config,
    )
    adapter = DetectorOnlyAdapter(detector, config)
    prepared = PreparedRepresentation(
        method=adapter.method_name,
        method_version=adapter.method_version,
        representation_format="detector-only-local-features",
        representation_version=REPRESENTATION_VERSION,
        payload=representation,
    )
    comparison = adapter.compare(prepared, prepared)
    assert comparison.raw_score == float(
        comparison.diagnostics["geometric_inlier_count"]
    )
    assert comparison.diagnostics["score_mode"] == "geometric_inlier_count"
    assert comparison.diagnostics["score_components"]["geometric_inlier_count"] == (
        comparison.raw_score
    )
    assert comparison.diagnostics["decision_threshold_applied"] is False
    assert comparison.diagnostics["decision_threshold"] is None
    assert adapter.metadata().config["decision_threshold"] is None
    assert adapter.metadata().config["score_mode"] == "geometric_inlier_count"


def test_adapter_compare_executes_scoring_helpers(monkeypatch) -> None:
    image = _textured_image()
    points = tuple(DetectedPoint(x, y, 1.0) for x, y in _locations())
    detector = FakeDetector(points)
    config = DetectorOnlyProtocolConfig(maximum_keypoints=20)
    representation, _, _ = build_representation(
        image,
        {"ppi": 1000.0},
        detector.detect(image, {"ppi": 1000.0}),
        config,
    )
    adapter = DetectorOnlyAdapter(detector, config)
    prepared = PreparedRepresentation(
        method=adapter.method_name,
        method_version=adapter.method_version,
        representation_format="detector-only-local-features",
        representation_version=REPRESENTATION_VERSION,
        payload=representation,
    )
    real_score_components = common_scoring.score_components
    real_raw_score = common_scoring.raw_score
    calls: list[str] = []

    def shifted_components(**kwargs):
        calls.append("score_components")
        components = real_score_components(**kwargs)
        components["geometric_inlier_count"] += 100.0
        return components

    def shifted_raw_score(mode, components):
        calls.append("raw_score")
        return real_raw_score(mode, components) + 7.0

    monkeypatch.setattr(common_scoring, "score_components", shifted_components)
    monkeypatch.setattr(common_scoring, "raw_score", shifted_raw_score)

    comparison = adapter.compare(prepared, prepared)
    assert calls == ["score_components", "raw_score"]
    assert comparison.raw_score == (
        float(comparison.diagnostics["geometric_inlier_count"]) + 107.0
    )


def test_generic_imports_are_repository_native_implementations() -> None:
    assert common_rootsift.__module__ == (
        "fingerprint_benchmark.local_features.descriptors.rootsift"
    )
    assert common_matching.match_descriptors.__module__ == (
        "fingerprint_benchmark.local_features.matching"
    )
    assert common_geometry.verify_geometry.__module__ == (
        "fingerprint_benchmark.local_features.geometry"
    )
