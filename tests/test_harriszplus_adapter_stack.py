from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from fingerprint_benchmark.contract import PreparedRepresentation
import fingerprint_benchmark.harriszplus.adapter as adapter_module
from fingerprint_benchmark.harriszplus.config import HarrisZPlusConfig
import fingerprint_benchmark.harriszplus.extractor as extractor_module
from fingerprint_benchmark.harriszplus.types import DetectedKeypoint, DetectorResult
from fingerprint_benchmark.sift import descriptors as sift_descriptors
from fingerprint_benchmark.sift import geometry as sift_geometry
from fingerprint_benchmark.sift import matching as sift_matching
from fingerprint_benchmark.sift.extractor import SiftRepresentation
from fingerprint_benchmark.sift.matching import MatchRecord


def _representation(
    points: np.ndarray,
    descriptors: np.ndarray,
    *,
    ppi: float,
    prepare_total_ms: float = 1.0,
) -> SiftRepresentation:
    count = int(points.shape[0])
    return SiftRepresentation(
        points=np.ascontiguousarray(points, dtype=np.float32),
        sizes=np.full(count, 4.0, dtype=np.float32),
        angles=np.zeros(count, dtype=np.float32),
        responses=np.ones(count, dtype=np.float32),
        octaves=np.zeros(count, dtype=np.int32),
        class_ids=np.arange(count, dtype=np.int32),
        descriptors=np.ascontiguousarray(descriptors, dtype=np.float32),
        width=512,
        height=512,
        ppi=float(ppi),
        metadata={"prepare_total_ms": float(prepare_total_ms)},
    )


def _prepared(payload: SiftRepresentation, digest: str) -> PreparedRepresentation:
    return PreparedRepresentation(
        method=adapter_module.METHOD_NAME,
        method_version=adapter_module.METHOD_VERSION,
        representation_format="harriszplus-keypoints-rootsift-descriptors",
        representation_version=adapter_module.REPRESENTATION_VERSION,
        payload=payload,
        metadata={
            "representation_sha256": digest,
            "prepare_total_ms": payload.metadata["prepare_total_ms"],
        },
    )


def _identity_descriptors(count: int) -> np.ndarray:
    descriptors = np.zeros((count, 128), dtype=np.float32)
    descriptors[np.arange(count), np.arange(count)] = 1.0
    return descriptors


def test_rootsift_matcher_and_geometry_are_imported_unchanged() -> None:
    assert extractor_module.rootsift is sift_descriptors.rootsift
    assert adapter_module.match_descriptors is sift_matching.match_descriptors
    assert adapter_module.verify_geometry is sift_geometry.verify_geometry

    raw = np.arange(1, 1 + 4 * 128, dtype=np.float32).reshape(4, 128)
    np.testing.assert_array_equal(
        extractor_module.rootsift(raw),
        sift_descriptors.rootsift(raw),
    )

    descriptors = _identity_descriptors(4)
    points = np.asarray(
        [[20.0, 20.0], [80.0, 20.0], [20.0, 80.0], [80.0, 80.0]],
        dtype=np.float32,
    )
    representation = _representation(points, descriptors, ppi=1000.0)
    actual = adapter_module.match_descriptors(
        representation.descriptors,
        representation.descriptors,
        lowe_ratio=0.75,
        matching_mode="mutual",
    )
    expected = sift_matching.match_descriptors(
        representation.descriptors,
        representation.descriptors,
        lowe_ratio=0.75,
        matching_mode="mutual",
    )
    assert actual == expected
    assert len(actual.submitted) == 4


def test_existing_geometry_reuse_normalizes_3px_at_1000_to_6px_at_2000() -> None:
    config = HarrisZPlusConfig(backend="reference_cpu")
    geometry_config = adapter_module._sift_geometry_config(config)
    descriptors = _identity_descriptors(4)
    points_1000 = np.asarray(
        [[20.0, 20.0], [80.0, 20.0], [20.0, 80.0], [80.0, 80.0]],
        dtype=np.float32,
    )
    points_2000 = points_1000 * 2.0
    matches = tuple(MatchRecord(index, index, 0.0, 1.0) for index in range(4))

    result_1000 = adapter_module.verify_geometry(
        _representation(points_1000, descriptors, ppi=1000.0),
        _representation(points_1000 + (2.0, 0.0), descriptors, ppi=1000.0),
        matches,
        geometry_config,
    )
    result_2000 = adapter_module.verify_geometry(
        _representation(points_2000, descriptors, ppi=2000.0),
        _representation(points_2000 + (4.0, 0.0), descriptors, ppi=2000.0),
        matches,
        geometry_config,
    )

    assert result_1000.success and result_2000.success
    assert result_1000.inlier_count == result_2000.inlier_count == 4
    assert result_1000.diagnostics["ransac_threshold_reference_pixels"] == 3.0
    assert result_2000.diagnostics["ransac_threshold_reference_pixels"] == 3.0
    assert result_2000.diagnostics["coordinate_normalization"] == "ppi_to_reference"
    native_threshold_2000 = (
        result_2000.diagnostics["ransac_threshold_reference_pixels"]
        * 2000.0
        / result_2000.diagnostics["reference_ppi"]
    )
    assert native_threshold_2000 == 6.0


def test_adapter_score_is_nonnegative_integer_and_metadata_has_no_threshold_or_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter_module,
        "implementation_source_hashes",
        lambda *, strict=True: {"strict": strict},
    )
    monkeypatch.setattr(adapter_module, "runtime_metadata", lambda config: {})
    adapter = adapter_module.HarrisZPlusGeometricAdapter(
        HarrisZPlusConfig(backend="reference_cpu")
    )
    points = np.asarray(
        [[20.0, 20.0], [80.0, 20.0], [20.0, 80.0], [80.0, 80.0]],
        dtype=np.float32,
    )
    descriptors = _identity_descriptors(4)
    first = _prepared(_representation(points, descriptors, ppi=1000.0), "a")
    second = _prepared(_representation(points, descriptors, ppi=1000.0), "b")

    outcome = adapter.compare(first, second)
    metadata = adapter.metadata()

    assert isinstance(outcome.raw_score, int)
    assert outcome.raw_score == 4
    assert outcome.raw_score >= 0
    assert outcome.diagnostics["decision_threshold_applied"] is False
    assert metadata.config["decision_threshold"] is None
    assert metadata.config["thresholding"] == "none_in_adapter_compare"
    assert metadata.config["representation_cache"] is False
    assert metadata.config["cross_pair_cache"] is False


def test_extractor_uses_one_exact_lanczos4_double_and_supplies_explicit_angles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HarrisZPlusConfig(backend="reference_cpu", max_keypoints=8)
    image = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
    image_path = tmp_path / "native.png"
    assert cv2.imwrite(str(image_path), image)

    detected = tuple(
        DetectedKeypoint(
            x=float(x),
            y=float(y),
            response=10.0 - source_index,
            scale_index=scale_index,
            sigma=config.output_sigma(scale_index),
            integration_sigma=config.output_integration_sigma(scale_index),
            effective_support_diameter=config.effective_gaussian_support_diameter(
                scale_index
            ),
            size=config.keypoint_size(scale_index),
            source_index=source_index,
        )
        for source_index, (x, y, scale_index) in enumerate(
            ((10.0, 10.0, 0), (20.0, 14.0, 1))
        )
    )
    detector_calls: list[dict[str, Any]] = []

    def fake_detector(
        source: np.ndarray,
        active_config: HarrisZPlusConfig,
        *,
        doubled_image: np.ndarray,
        return_response_maps: bool,
    ) -> DetectorResult:
        detector_calls.append(
            {
                "source_shape": source.shape,
                "source_dtype": source.dtype,
                "doubled_shape": doubled_image.shape,
                "doubled_dtype": doubled_image.dtype,
                "return_response_maps": return_response_maps,
                "config": active_config,
            }
        )
        return DetectorResult(
            backend="reference_cpu",
            keypoints=detected,
            diagnostics={"final_keypoint_count": len(detected)},
            timings={},
        )

    monkeypatch.setattr(extractor_module, "detect_harriszplus_cpu", fake_detector)
    resize_calls: list[dict[str, Any]] = []
    real_resize = cv2.resize

    def recording_resize(
        source: np.ndarray,
        dsize: tuple[int, int],
        *,
        interpolation: int,
    ) -> np.ndarray:
        resize_calls.append(
            {
                "source_shape": source.shape,
                "source_dtype": source.dtype,
                "dsize": dsize,
                "interpolation": interpolation,
            }
        )
        return real_resize(source, dsize, interpolation=interpolation)

    monkeypatch.setattr(extractor_module.cv2, "resize", recording_resize)
    supplied: list[cv2.KeyPoint] = []

    class FakeSift:
        def compute(
            self,
            descriptor_image: np.ndarray,
            keypoints: list[cv2.KeyPoint],
        ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
            assert descriptor_image.shape == image.shape
            assert descriptor_image.dtype == np.uint8
            supplied.extend(keypoints)
            raw = np.tile(np.arange(1, 129, dtype=np.float32), (len(keypoints), 1))
            return keypoints, raw

    monkeypatch.setattr(extractor_module.cv2, "SIFT_create", lambda **kwargs: FakeSift())
    representation, diagnostics, _ = extractor_module.extract_representation(
        image_path,
        {"ppi": 1000},
        config,
    )

    assert len(resize_calls) == 1
    assert resize_calls[0] == {
        "source_shape": (24, 32),
        "source_dtype": np.dtype(np.uint8),
        "dsize": (64, 48),
        "interpolation": cv2.INTER_LANCZOS4,
    }
    assert len(detector_calls) == 1
    assert detector_calls[0]["source_shape"] == (24, 32)
    assert detector_calls[0]["doubled_shape"] == (48, 64)
    assert detector_calls[0]["source_dtype"] == np.dtype(np.float32)
    assert detector_calls[0]["doubled_dtype"] == np.dtype(np.float32)
    assert all(0.0 <= keypoint.angle < 360.0 for keypoint in supplied)
    assert all(keypoint.angle != -1.0 for keypoint in supplied)
    assert [keypoint.octave for keypoint in supplied] == [0, 0]
    assert [keypoint.class_id for keypoint in supplied] == [0, 1]
    np.testing.assert_allclose(
        [keypoint.size for keypoint in supplied],
        [point.size for point in detected],
        rtol=0.0,
        atol=1e-6,
    )
    assert representation.width == 32 and representation.height == 24
    np.testing.assert_array_equal(representation.octaves, np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(representation.class_ids, np.asarray([0, 1], dtype=np.int32))
    assert representation.metadata["harriszplus_scale_indices"] == [0, 1]
    assert representation.metadata["harriszplus_source_indices"] == [0, 1]
    assert diagnostics["lanczos_call_count"] == 1
    assert diagnostics["lanczos_scale_indices"] == [0, 1]
    assert diagnostics["keypoint_size_mapping"] == "size = 2 * output_integration_sigma"


def test_adapter_prepares_the_same_path_twice_without_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = adapter_module.HarrisZPlusGeometricAdapter(
        HarrisZPlusConfig(backend="reference_cpu")
    )
    calls: list[Path] = []

    def fake_extract(
        image_path: Path,
        image_metadata: dict[str, Any],
        config: HarrisZPlusConfig,
    ) -> tuple[SiftRepresentation, dict[str, Any], float]:
        calls.append(image_path)
        payload = _representation(
            np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
            _identity_descriptors(2),
            ppi=float(image_metadata["ppi"]),
        )
        return payload, {"representation_sha256": f"call-{len(calls)}"}, 1.0

    monkeypatch.setattr(adapter_module, "extract_representation", fake_extract)
    image_path = tmp_path / "same.png"
    first = adapter.prepare(image_path, {"ppi": 1000})
    second = adapter.prepare(image_path, {"ppi": 1000})

    assert calls == [image_path, image_path]
    assert first.representation.payload is not second.representation.payload
    assert first.representation.metadata["representation_sha256"] == "call-1"
    assert second.representation.metadata["representation_sha256"] == "call-2"
