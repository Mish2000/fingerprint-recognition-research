import base64
import math
from pathlib import Path

import pytest

from fingerprint_benchmark.contract import MethodExecutionError, PreparedRepresentation
from fingerprint_benchmark.sourceafis_adapter import SourceAfisAdapter
from fingerprint_benchmark.sourceafis_client import (
    EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE,
    EXTRACT_RAW_TEMPLATE_INTERNAL_TIMING_SCOPE,
    EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE,
    VERIFY_INTERNAL_TIMING_SCOPE,
    SourceAfisClientError,
    SourceAfisHealth,
    SourceAfisTemplate,
    SourceAfisVerification,
)


def test_sourceafis_adapter_validates_health_once_and_passes_sd300_ppi_to_sidecar(tmp_path):
    client = FakeSourceAfisClient()
    adapter = SourceAfisAdapter(client)
    image_b = tmp_path / "sd300b.png"
    image_c = tmp_path / "sd300c.png"
    image_b.write_bytes(b"sd300b image")
    image_c.write_bytes(b"sd300c image")

    prepared_b = adapter.prepare(image_b, {"ppi": 1000, "dataset": "sd300b", "pair_id": "b", "side": "a"})
    prepared_c = adapter.prepare(image_c, {"ppi": 2000, "dataset": "sd300c", "pair_id": "c", "side": "a"})
    comparison = adapter.compare(prepared_b.representation, prepared_c.representation)

    assert client.health_calls == 1
    assert client.extract_dpis == [1000.0, 2000.0]
    assert client.verify_calls == [(prepared_b.representation.payload, prepared_c.representation.payload)]
    assert math.isfinite(comparison.raw_score)
    assert comparison.raw_score == pytest.approx(42.5)
    assert prepared_b.method_internal_ms == pytest.approx(8.25)
    assert prepared_c.method_internal_ms == pytest.approx(8.25)
    assert comparison.method_internal_ms == pytest.approx(1.5)
    assert prepared_b.representation.metadata["effective_dpi"] == pytest.approx(1000.0)
    assert prepared_c.representation.metadata["effective_dpi"] == pytest.approx(2000.0)
    assert "image_sha256" not in prepared_b.representation.metadata
    assert "image_metadata" not in prepared_b.representation.metadata


@pytest.mark.parametrize("metadata", [{}, {"ppi": ""}, {"ppi": "nan"}, {"ppi": -1}, {"ppi": 5000}])
def test_sourceafis_adapter_rejects_missing_or_invalid_dpi_without_500_fallback(tmp_path, metadata):
    client = FakeSourceAfisClient()
    adapter = SourceAfisAdapter(client)
    image = tmp_path / "image.png"
    image.write_bytes(b"image")

    with pytest.raises(MethodExecutionError, match="DPI|PPI"):
        adapter.prepare(image, metadata)

    assert client.extract_dpis == []


def test_sourceafis_adapter_fails_explicitly_on_version_or_contract_mismatch():
    client = FakeSourceAfisClient(health={**_health_payload(), "sourceafis_version": "9.9.9", "method_version": "9.9.9"})

    with pytest.raises(SourceAfisClientError, match="mismatch"):
        SourceAfisAdapter(client)


def test_sourceafis_adapter_records_comparison_failures_without_fake_zero_score():
    client = FakeSourceAfisClient(verify_error=SourceAfisClientError("invalid_serialized_template", "invalid template"))
    adapter = SourceAfisAdapter(client)
    representation = PreparedRepresentation(
        method="sourceafis",
        method_version="3.18.1",
        representation_format="sourceafis",
        representation_version="3.18.1",
        payload=base64.b64encode(b"template").decode("ascii"),
    )

    with pytest.raises(SourceAfisClientError, match="invalid template") as exc_info:
        adapter.compare(representation, representation)

    assert exc_info.value.error_code == "invalid_serialized_template"


def test_sourceafis_adapter_metadata_documents_no_preprocessing_cache_normalization_or_thresholding():
    adapter = SourceAfisAdapter(FakeSourceAfisClient())
    metadata = adapter.metadata()

    assert metadata.method == "sourceafis"
    assert metadata.method_version == "3.18.1"
    assert metadata.score_direction == "higher_is_more_similar"
    assert "FingerprintMatcher.match" in metadata.score_semantics
    assert metadata.implementation_provenance["official_implementation_family"] == "Java"
    assert metadata.implementation_provenance["maven_coordinates"] == "com.machinezoo.sourceafis:sourceafis:3.18.1"
    assert metadata.config["external_preprocessing"] == "none"
    assert metadata.config["template_cache"] is False
    assert metadata.config["normalization"] == "none"
    assert metadata.config["thresholding"] == "none"
    assert "timing" in metadata.runtime["timing_inclusion_exclusion_policy"]


def test_sidecar_source_does_not_log_image_template_or_base64_payloads():
    java_files = list((Path.cwd() / "apps" / "sourceafis-sidecar" / "src" / "main" / "java").rglob("*.java"))
    logging_lines = []
    for path in java_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "System.out" in line or "System.err" in line:
                logging_lines.append(line)

    joined = "\n".join(logging_lines)
    assert "image_base64" not in joined
    assert "template_base64" not in joined
    assert "template_a_base64" not in joined
    assert "template_b_base64" not in joined


class FakeSourceAfisClient:
    def __init__(self, health=None, verify_error=None) -> None:
        self.health_payload = health or _health_payload()
        self.verify_error = verify_error
        self.health_calls = 0
        self.extract_dpis: list[float] = []
        self.verify_calls: list[tuple[str, str]] = []

    def health(self):
        self.health_calls += 1
        return SourceAfisHealth(dict(self.health_payload))

    def extract_template(self, image_bytes: bytes, dpi: float):
        self.extract_dpis.append(float(dpi))
        return SourceAfisTemplate(
            template_base64=base64.b64encode(f"template:{dpi}:{len(image_bytes)}".encode("ascii")).decode("ascii"),
            template_format="sourceafis",
            template_version="3.18.1",
            sourceafis_version="3.18.1",
            effective_dpi=float(dpi),
            method_internal_ms=8.25,
        )

    def verify(self, template_a_base64: str, template_b_base64: str):
        if self.verify_error is not None:
            raise self.verify_error
        self.verify_calls.append((template_a_base64, template_b_base64))
        return SourceAfisVerification(
            raw_score=42.5,
            sourceafis_version="3.18.1",
            method_internal_ms=1.5,
        )

    def close(self):
        pass


def _health_payload():
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
        "java_runtime_version": "20.0.2+9",
        "java_runtime_vendor": "Test Vendor",
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
