import base64
from dataclasses import replace
import math

import cv2
import numpy as np
import pytest

from fingerprint_benchmark.contract import MethodExecutionError
from fingerprint_benchmark.detectors.sourceafis_final_minutiae import (
    METHOD_NAME,
    METHOD_VERSION,
    SourceAfisFinalMinutiaeDetector,
    SourceAfisFinalMinutiaeRootSIFTGeometricAdapter,
    scaled_to_native_pixel_center,
)
from fingerprint_benchmark.local_features.detector_only import (
    DetectorOnlyProtocolConfig,
    canonical_features,
)
from fingerprint_benchmark.provenance import ProvenanceError
from fingerprint_benchmark.runner import prepare_run_context
from fingerprint_benchmark.sourceafis_client import (
    EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE,
    EXTRACT_RAW_TEMPLATE_INTERNAL_TIMING_SCOPE,
    EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE,
    VERIFY_INTERNAL_TIMING_SCOPE,
    SourceAfisContractError,
    SourceAfisFinalMinutiae,
    SourceAfisHealth,
    SourceAfisMinutia,
    SourceAfisSidecarClient,
    _parse_final_minutiae_response,
    validate_health,
)


def test_client_sends_exact_row_major_bytes_and_parses_strict_response(monkeypatch):
    pixels = bytes(range(12))
    captured = {}
    client = SourceAfisSidecarClient("http://127.0.0.1:8765")

    def fake_request(method, path, payload=None):
        captured.update(method=method, path=path, payload=payload)
        return _response(width=4, height=3, scaled_width=2, scaled_height=2)

    monkeypatch.setattr(client, "_request_json", fake_request)
    try:
        parsed = client.extract_final_minutiae(pixels, 4, 3, 1000)
    finally:
        client.close()

    assert captured["method"] == "POST"
    assert captured["path"] == "/extract-final-minutiae"
    assert base64.b64decode(captured["payload"]["pixels_base64"], validate=True) == pixels
    assert parsed.minutia_count == 2
    assert parsed.minutiae[1].minutia_type == "BIFURCATION"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("template_sha256"),
        lambda payload: payload.__setitem__("native_width", "4"),
        lambda payload: payload["minutiae"][0].__setitem__("direction_radians", math.nan),
        lambda payload: payload.__setitem__("minutia_count", 3),
        lambda payload: payload["minutiae"][0].__setitem__("x_scaled", 2),
        lambda payload: payload["minutiae"][0].__setitem__("type", "OTHER"),
        lambda payload: payload.__setitem__("template_sha256", "ABC"),
        lambda payload: payload.__setitem__("method_internal_ms", -1),
    ],
)
def test_client_rejects_every_invalid_final_minutia_schema(mutation):
    payload = _response(width=4, height=3, scaled_width=2, scaled_height=2)
    mutation(payload)
    with pytest.raises(SourceAfisContractError):
        _parse_final_minutiae_response(payload, expected_width=4, expected_height=3, expected_dpi=1000)


def test_health_requires_final_minutiae_capability_and_v22_contract():
    health = SourceAfisHealth(_health())
    validate_health(health)
    broken = _health()
    broken["supports_final_minutiae_extraction"] = False
    with pytest.raises(SourceAfisContractError, match="supports_final_minutiae_extraction"):
        validate_health(SourceAfisHealth(broken))


def test_pixel_center_coordinate_mapping_covers_identity_scaling_and_bounds():
    assert scaled_to_native_pixel_center(0, native_size=500, scaled_size=500) == 0.0
    assert scaled_to_native_pixel_center(499, native_size=500, scaled_size=500) == 499.0
    assert scaled_to_native_pixel_center(0, native_size=1000, scaled_size=500) == 0.5
    assert scaled_to_native_pixel_center(499, native_size=1000, scaled_size=500) == 998.5
    assert scaled_to_native_pixel_center(0, native_size=2000, scaled_size=500) == 1.5
    assert scaled_to_native_pixel_center(499, native_size=2000, scaled_size=500) == 1997.5
    assert scaled_to_native_pixel_center(2, native_size=11, scaled_size=5) == pytest.approx(5.0)
    assert scaled_to_native_pixel_center(2, native_size=7, scaled_size=5) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="outside"):
        scaled_to_native_pixel_center(5, native_size=10, scaled_size=5)
    with pytest.raises(ValueError, match="out-of-bounds"):
        scaled_to_native_pixel_center(0, native_size=5, scaled_size=10)


def test_detector_preserves_pixels_rejects_mask_and_is_deterministic(monkeypatch):
    import fingerprint_benchmark.detectors.sourceafis_final_minutiae as module

    base = np.arange(48, dtype=np.uint8).reshape(6, 8)
    image = base[:, ::2]
    original = image.copy()
    client = FakeFinalClient(_final(width=4, height=6, scaled_width=2, scaled_height=3))
    detector = SourceAfisFinalMinutiaeDetector(client, health=SourceAfisHealth(_health()))
    ticks = iter([*range(0, 80, 10), *range(100, 180, 10)])
    monkeypatch.setattr(module, "perf_counter_ns", lambda: next(ticks))

    first = detector.detect(image, {"ppi": 1000})
    second = detector.detect(image, {"ppi": 1000})

    assert client.pixels == [image.tobytes(order="C"), image.tobytes(order="C")]
    assert np.array_equal(image, original)
    assert first == second
    assert all(0 <= point.x < 4 and 0 <= point.y < 6 for point in first.points)
    assert set(first.diagnostics) >= {
        "pixel_serialization_ms",
        "sidecar_request_wall_ms",
        "sourceafis_method_internal_ms",
        "coordinate_mapping_ms",
        "detector_total_ms",
    }
    with pytest.raises(ValueError, match="does not support masks"):
        detector.detect(image, {"ppi": 1000}, mask=np.ones_like(image))


def test_sourceafis_metadata_direction_type_and_response_do_not_change_common_features():
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    first = FakeFinalClient(_final(width=64, height=64, scaled_width=64, scaled_height=64))
    second_payload = _final(width=64, height=64, scaled_width=64, scaled_height=64)
    second_payload = replace(
        second_payload,
        minutiae=tuple(
            SourceAfisMinutia(
                item.source_index,
                item.x_scaled,
                item.y_scaled,
                item.direction_radians + 1.0,
                "ENDING" if item.minutia_type == "BIFURCATION" else "BIFURCATION",
            )
            for item in second_payload.minutiae
        ),
    )
    second = FakeFinalClient(second_payload)
    detector_a = SourceAfisFinalMinutiaeDetector(first, health=SourceAfisHealth(_health()))
    detector_b = SourceAfisFinalMinutiaeDetector(second, health=SourceAfisHealth(_health()))
    result_a = detector_a.detect(image, {"ppi": 1000})
    result_b = detector_b.detect(image, {"ppi": 1000})
    canonical_a, diagnostics_a = canonical_features(
        image, {"ppi": 1000}, result_a, DetectorOnlyProtocolConfig()
    )
    canonical_b, diagnostics_b = canonical_features(
        image, {"ppi": 1000}, result_b, DetectorOnlyProtocolConfig()
    )
    assert canonical_a == canonical_b
    assert diagnostics_a["detector_fields_used"] == ["x", "y"]
    assert diagnostics_b["detector_response_used"] is False


def test_public_adapter_identity_threshold_point_limit_and_provenance(tmp_path):
    client = FakeFinalClient(_final(width=32, height=32, scaled_width=32, scaled_height=32))
    adapter = SourceAfisFinalMinutiaeRootSIFTGeometricAdapter(
        client,
        health=SourceAfisHealth(_health()),
        protocol_config=DetectorOnlyProtocolConfig(maximum_keypoints=1),
    )
    metadata = adapter.metadata()
    assert metadata.method == METHOD_NAME
    assert metadata.method_version == METHOD_VERSION
    assert metadata.config["decision_threshold"] is None
    assert metadata.config["score_mode"] == "geometric_inlier_count"
    assert metadata.config["required_runtime_artifacts"] == ["sidecar_jar_sha256"]
    declared = {str(path).replace("\\", "/") for path in adapter.implementation_source_paths()}
    assert any(path.endswith("sourceafis_client.py") for path in declared)
    assert any(path.endswith("SourceAfisV2Engine.java") for path in declared)
    assert any(path.endswith("pom.xml") for path in declared)

    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(MethodExecutionError) as error:
        adapter.prepare(image_path, {"ppi": 1000})
    assert error.value.error_code == "too_many_detector_points"

    with pytest.raises(ValueError, match="rootsift"):
        SourceAfisFinalMinutiaeRootSIFTGeometricAdapter(
            client,
            health=SourceAfisHealth(_health()),
            protocol_config=DetectorOnlyProtocolConfig(descriptor="sift"),
        )


def test_jar_sha_is_enforced_before_run_context_and_enters_hash(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("placeholder", encoding="utf-8")
    adapter = SourceAfisFinalMinutiaeRootSIFTGeometricAdapter(
        FakeFinalClient(_final(width=32, height=32, scaled_width=32, scaled_height=32)),
        health=SourceAfisHealth(_health()),
    )
    with pytest.raises(ProvenanceError, match="before warm-up"):
        prepare_run_context(
            manifest_path=manifest,
            expected_dataset="sd300b",
            expected_protocol="test",
            adapter=adapter,
            results_root=tmp_path,
            startup_validation={},
        )
    context = prepare_run_context(
        manifest_path=manifest,
        expected_dataset="sd300b",
        expected_protocol="test",
        adapter=adapter,
        results_root=tmp_path,
        startup_validation={"jar_sha256": "a" * 64},
    )
    assert context.implementation_hash_components["sidecar_jar_sha256"] == "a" * 64


class FakeFinalClient:
    def __init__(self, result):
        self.result = result
        self.pixels = []

    def extract_final_minutiae(self, pixels, width, height, dpi):
        self.pixels.append(pixels)
        assert (width, height, dpi) == (
            self.result.native_width,
            self.result.native_height,
            self.result.effective_dpi,
        )
        return self.result


def _response(*, width, height, scaled_width, scaled_height):
    return {
        "sourceafis_version": "3.18.1",
        "template_version": "3.18.1-java",
        "effective_dpi": 1000.0,
        "native_width": width,
        "native_height": height,
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "coordinate_space": "sourceafis_500_dpi_scaled_image",
        "selection_stage": "sourceafis_final_template_minutiae",
        "selection_semantics": "sourceafis_final_selected_minutia_set",
        "source_order_semantics": "deterministic_sourceafis_template_order_not_quality_ranking",
        "template_sha256": "a" * 64,
        "minutia_count": 2,
        "minutiae": [
            {"source_index": 0, "x_scaled": 0, "y_scaled": 0, "direction_radians": 0.5, "type": "ENDING"},
            {
                "source_index": 1,
                "x_scaled": scaled_width - 1,
                "y_scaled": scaled_height - 1,
                "direction_radians": 1.5,
                "type": "BIFURCATION",
            },
        ],
        "method_internal_ms": 1.25,
    }


def _final(**dimensions):
    response = _response(**dimensions)
    return _parse_final_minutiae_response(
        response,
        expected_width=dimensions["width"],
        expected_height=dimensions["height"],
        expected_dpi=1000,
    )


def _health():
    return {
        "status": "ok",
        "method": "sourceafis",
        "official_implementation_family": "Java",
        "engine": "SourceAFIS",
        "sourceafis_version": "3.18.1",
        "method_version": "3.18.1",
        "maven_coordinates": "com.machinezoo.sourceafis:sourceafis:3.18.1",
        "template_format": "sourceafis",
        "template_version": "3.18.1",
        "contract_version": "sourceafis-sidecar-v2.3",
        "sidecar_implementation_version": "0.4.0",
        "java_runtime_version": "test",
        "java_runtime_vendor": "test",
        "transport": "localhost_http",
        "bind_host": "127.0.0.1",
        "port": 8765,
        "dpi_policy": {
            "required": True,
            "source": "manifest_or_prepare_metadata",
            "min_dpi": 100.0,
            "max_dpi": 4000.0,
            "silent_default": False,
        },
        "external_preprocessing": "none",
        "template_cache": False,
        "supports_template_extraction": True,
        "supports_raw_template_extraction": True,
        "supports_final_minutiae_extraction": True,
        "supports_pairwise_verification": True,
        "supports_identification": False,
        "method_internal_timing_unit": "milliseconds",
        "extract_template_internal_timing_scope": EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE,
        "extract_raw_template_internal_timing_scope": EXTRACT_RAW_TEMPLATE_INTERNAL_TIMING_SCOPE,
        "extract_final_minutiae_internal_timing_scope": EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE,
        "verify_internal_timing_scope": VERIFY_INTERNAL_TIMING_SCOPE,
        "raw_template_endpoint": "/extract-template-raw",
        "raw_template_input": "raw_uint8_grayscale_row_major",
        "final_minutiae_endpoint": "/extract-final-minutiae",
        "final_minutiae_input": "raw_uint8_grayscale_row_major",
        "final_minutiae_coordinate_space": "sourceafis_500_dpi_scaled_image",
        "final_minutiae_stage": "final_template_minutiae",
    }
