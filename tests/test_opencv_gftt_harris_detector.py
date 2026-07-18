from __future__ import annotations

import inspect

import cv2
import numpy as np

from fingerprint_benchmark.detectors import Detector
from fingerprint_benchmark.detectors import opencv_gftt_harris as harris_module
from fingerprint_benchmark.detectors.opencv_gftt_harris import (
    DETECTOR_NAME,
    DETECTOR_VERSION,
    METHOD_NAME,
    METHOD_VERSION,
    OpenCVGFTTHarrisDetector,
    OpenCVGFTTHarrisRootSIFTGeometricAdapter,
    OpenCVHarrisConfig,
)


def _checkerboard(size: int = 96, block: int = 12) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return (((xx // block + yy // block) % 2) * 255).astype(np.uint8)


def _direct(
    image: np.ndarray,
    config: OpenCVHarrisConfig,
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    return cv2.goodFeaturesToTrack(
        image=image,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        mask=mask,
        blockSize=config.block_size,
        gradientSize=config.gradient_size,
        useHarrisDetector=True,
        k=config.harris_k,
    )


def test_wrapper_coordinates_are_exactly_the_direct_opencv_result() -> None:
    image = _checkerboard()
    config = OpenCVHarrisConfig(max_corners=80)
    detector = OpenCVGFTTHarrisDetector(config)
    direct = _direct(image, config)
    result = detector.detect(image, {"ppi": 1000.0})

    assert isinstance(detector, Detector)
    assert direct is not None
    expected = np.asarray(direct, dtype=np.float32).reshape(-1, 2)
    actual = np.asarray([(point.x, point.y) for point in result.points], dtype=np.float32)
    assert np.array_equal(actual, expected)
    assert result.detector_name == DETECTOR_NAME
    assert result.detector_version == DETECTOR_VERSION


def test_uniform_image_returns_an_empty_ranked_tuple() -> None:
    result = OpenCVGFTTHarrisDetector().detect(
        np.full((64, 64), 127, dtype=np.uint8),
        {"ppi": 1000.0},
    )
    assert result.points == ()
    assert result.diagnostics["corners_is_none"] is True


def test_mask_excludes_every_coordinate_outside_the_allowed_region() -> None:
    image = _checkerboard()
    mask = np.zeros_like(image)
    mask[:, : image.shape[1] // 2] = 255
    config = OpenCVHarrisConfig(max_corners=100, min_distance=2.0)
    result = OpenCVGFTTHarrisDetector(config).detect(
        image,
        {"ppi": 1000.0},
        mask=mask,
    )
    direct = _direct(image, config, mask)
    assert direct is not None
    assert all(mask[int(round(point.y)), int(round(point.x))] > 0 for point in result.points)
    assert np.array_equal(
        np.asarray([(point.x, point.y) for point in result.points], dtype=np.float32),
        np.asarray(direct, dtype=np.float32).reshape(-1, 2),
    )


def test_repeat_detection_is_exact_and_metadata_is_complete() -> None:
    image = _checkerboard()
    config = OpenCVHarrisConfig(max_corners=60)
    detector = OpenCVGFTTHarrisDetector(config)
    first = detector.detect(image, {"ppi": 1000.0})
    second = detector.detect(image, {"ppi": 1000.0})
    assert first.points == second.points
    assert dict(first.detector_config) == config.as_dict()
    assert first.metadata["opencv_parameters"] == config.as_dict()
    assert first.metadata["opencv_version"] == cv2.__version__
    assert first.detector_time_ms >= 0.0


def test_harris_wrapper_contains_no_local_corner_selection_implementation() -> None:
    source = inspect.getsource(harris_module)
    for forbidden in ("corner" + "Harris", "di" + "late", "N" + "MS"):
        assert forbidden not in source
    assert "fingerprint_benchmark.sift" not in source
    assert "fingerprint_benchmark.harriszplus" not in source


def test_public_harris_method_runs_the_complete_common_pipeline(tmp_path) -> None:
    image_path = tmp_path / "checkerboard.png"
    assert cv2.imwrite(str(image_path), _checkerboard(size=128, block=8))
    adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter(
        detector_config=OpenCVHarrisConfig(max_corners=100, min_distance=3.0)
    )
    prepared = adapter.prepare(image_path, {"ppi": 1000.0})
    comparison = adapter.compare(prepared.representation, prepared.representation)
    metadata = adapter.metadata()

    assert metadata.method == METHOD_NAME
    assert metadata.method_version == METHOD_VERSION
    assert prepared.diagnostics["detector_name"] == DETECTOR_NAME
    assert prepared.diagnostics["detector_version"] == DETECTOR_VERSION
    assert comparison.raw_score >= 0.0
    assert comparison.diagnostics["decision_threshold_applied"] is False
