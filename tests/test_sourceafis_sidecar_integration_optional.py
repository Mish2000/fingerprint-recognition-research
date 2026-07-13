import binascii
import math
import os
import struct
import zlib

import pytest

from fingerprint_benchmark.sourceafis_client import SourceAfisSidecarClient, validate_health


def test_sourceafis_sidecar_health_extract_verify_and_dpi_roundtrip():
    if not _integration_enabled():
        pytest.skip("Set SOURCEAFIS_INTEGRATION_TESTS=true to run the real SourceAFIS sidecar integration test.")

    service_url = os.getenv("SOURCEAFIS_SERVICE_URL", "http://127.0.0.1:8765")
    client = SourceAfisSidecarClient(service_url, timeout_seconds=30.0)
    try:
        health = client.health()
        validate_health(health)
        image = _synthetic_fingerprint_png(0)
        template_1000 = client.extract_template(image, 1000)
        template_2000 = client.extract_template(image, 2000)
        verification = client.verify(template_1000.template_base64, template_1000.template_base64)
    finally:
        client.close()

    assert health.raw["official_implementation_family"] == "Java"
    assert health.sourceafis_version == "3.18.1"
    assert template_1000.template_base64
    assert template_2000.template_base64
    assert template_1000.template_base64 != template_2000.template_base64
    assert template_1000.effective_dpi == pytest.approx(1000.0)
    assert template_2000.effective_dpi == pytest.approx(2000.0)
    assert math.isfinite(template_1000.method_internal_ms)
    assert template_1000.method_internal_ms >= 0.0
    assert math.isfinite(verification.raw_score)
    assert math.isfinite(verification.method_internal_ms)
    assert verification.method_internal_ms >= 0.0


def _integration_enabled() -> bool:
    return str(os.getenv("SOURCEAFIS_INTEGRATION_TESTS", "")).lower() in {"1", "true", "yes", "on"}


def _synthetic_fingerprint_png(variant: int) -> bytes:
    width = 360
    height = 460
    period = 10.5
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            dx = (x - width / 2) / (width * 0.46)
            dy = (y - height / 2) / (height * 0.43)
            if dx * dx + dy * dy > 1.0:
                row.append(255)
                continue
            warped_y = y + 18.0 * math.sin(x * 0.035 + variant * 0.35) + 5.0 * math.sin(y * 0.02)
            ridge_pos = (warped_y - 52.0) % period
            distance = min(ridge_pos, period - ridge_pos)
            row.append(22 if distance < 1.7 else 245)
        rows.append(bytes([0]) + bytes(row))
    return _png_bytes(width, height, b"".join(rows))


def _png_bytes(width: int, height: int, filtered_rows: bytes) -> bytes:
    def chunk(name: bytes, data: bytes) -> bytes:
        checksum = binascii.crc32(name)
        checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(filtered_rows)) + chunk(b"IEND", b"")
