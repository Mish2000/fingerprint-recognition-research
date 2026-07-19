import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fingerprint_benchmark.sourceafis_client import (
    EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE,
    EXTRACT_RAW_TEMPLATE_INTERNAL_TIMING_SCOPE,
    EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE,
    VERIFY_INTERNAL_TIMING_SCOPE,
    SourceAfisClientError,
    SourceAfisContractError,
    SourceAfisHealth,
    SourceAfisSidecarClient,
    validate_health,
)


def test_sourceafis_client_reuses_connection_and_maps_v2_contract():
    server = RecordingServer()
    try:
        client = SourceAfisSidecarClient(server.url)
        try:
            health = client.health()
            validate_health(health)
            template = client.extract_template(b"fake png", 1000)
            raw_template = client.extract_template_raw(b"\x00\x01\x02\x03", 2, 2, 1000)
            verification = client.verify(template.template_base64, template.template_base64)
        finally:
            client.close()

        assert health.sourceafis_version == "3.18.1"
        assert template.effective_dpi == pytest.approx(1000.0)
        assert template.method_internal_ms == pytest.approx(12.5)
        assert raw_template.template_sha256 == hashlib.sha256(b"template").hexdigest()
        assert raw_template.native_width == 2
        assert raw_template.native_height == 2
        assert verification.raw_score == pytest.approx(77.25)
        assert verification.method_internal_ms == pytest.approx(2.75)
        assert client.health_request_count == 1
        assert [request["path"] for request in server.requests] == [
            "/health",
            "/extract-template",
            "/extract-template-raw",
            "/verify",
        ]
        assert server.requests[1]["payload"]["image_base64"] == base64.b64encode(b"fake png").decode("ascii")
        assert server.requests[1]["payload"]["dpi"] == 1000.0
        assert server.requests[2]["payload"] == {
            "width": 2,
            "height": 2,
            "pixels_base64": base64.b64encode(b"\x00\x01\x02\x03").decode("ascii"),
            "dpi": 1000.0,
        }
        assert server.requests[3]["payload"]["template_a_base64"] == template.template_base64
    finally:
        server.close()


@pytest.mark.parametrize("url", ["http://example.com:8765", "http://192.168.1.5:8765", "http://0.0.0.0:8765"])
def test_sourceafis_client_rejects_non_loopback_plain_http_before_connecting(url):
    with pytest.raises(SourceAfisClientError) as exc_info:
        SourceAfisSidecarClient(url)

    assert exc_info.value.error_code == "remote_transport_forbidden"


@pytest.mark.parametrize("url", ["http://localhost:8765", "http://127.0.0.1:8765", "http://[::1]:8765"])
def test_sourceafis_client_accepts_only_the_documented_loopback_hosts(url):
    client = SourceAfisSidecarClient(url)
    client.close()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("template_format", "other"),
        ("template_version", "9.9.9"),
        ("sidecar_implementation_version", "0.1.0"),
        ("transport", "http"),
        ("external_preprocessing", "resize"),
        ("template_cache", True),
        ("supports_template_extraction", False),
        ("supports_raw_template_extraction", False),
        ("supports_pairwise_verification", False),
        ("supports_identification", True),
    ],
)
def test_validate_health_rejects_every_hardened_contract_mismatch(field_name, invalid_value):
    payload = _health_payload(8765)
    payload[field_name] = invalid_value

    with pytest.raises(SourceAfisContractError, match=field_name):
        validate_health(SourceAfisHealth(payload))


@pytest.mark.parametrize("invalid_timing", [-1.0, "1.0", True])
def test_sourceafis_client_rejects_invalid_method_internal_timing(invalid_timing):
    server = RecordingServer(extract_internal_ms=invalid_timing)
    try:
        client = SourceAfisSidecarClient(server.url)
        try:
            with pytest.raises(SourceAfisContractError, match="negative|numeric"):
                client.extract_template(b"fake png", 1000)
        finally:
            client.close()
    finally:
        server.close()


def test_sourceafis_client_maps_structured_errors_to_explicit_error_codes():
    server = RecordingServer(error_path="/verify")
    try:
        client = SourceAfisSidecarClient(server.url)
        try:
            template = base64.b64encode(b"template").decode("ascii")
            with pytest.raises(SourceAfisClientError, match="bad template") as exc_info:
                client.verify(template, template)
        finally:
            client.close()

        assert exc_info.value.error_code == "invalid_serialized_template"
    finally:
        server.close()


def test_sourceafis_client_rejects_raw_template_sha_tampering():
    server = RecordingServer(raw_template_sha256="0" * 64)
    try:
        client = SourceAfisSidecarClient(server.url)
        try:
            with pytest.raises(SourceAfisContractError, match="template_sha256"):
                client.extract_template_raw(b"\x00\x01\x02\x03", 2, 2, 1000)
        finally:
            client.close()
    finally:
        server.close()


class RecordingServer:
    def __init__(
        self,
        error_path: str | None = None,
        extract_internal_ms: object = 12.5,
        raw_template_sha256: str | None = None,
    ) -> None:
        self.requests: list[dict[str, object]] = []
        self.error_path = error_path
        self.extract_internal_ms = extract_internal_ms
        self.raw_template_sha256 = raw_template_sha256 or hashlib.sha256(b"template").hexdigest()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                owner._handle(self)

            def do_POST(self):
                owner._handle(self)

            def log_message(self, format, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        raw_body = handler.rfile.read(length) if length else b""
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else None
        self.requests.append({"path": handler.path, "payload": payload})
        if handler.path == self.error_path:
            self._write(handler, 422, {"error_code": "invalid_serialized_template", "error_message": "bad template"})
            return
        if handler.path == "/health":
            self._write(handler, 200, _health_payload(self._server.server_address[1]))
            return
        if handler.path == "/extract-template":
            self._write(
                handler,
                200,
                {
                    "template_base64": base64.b64encode(b"template").decode("ascii"),
                    "template_format": "sourceafis",
                    "template_version": "3.18.1",
                    "sourceafis_version": "3.18.1",
                    "effective_dpi": payload["dpi"],
                    "method_internal_ms": self.extract_internal_ms,
                },
            )
            return
        if handler.path == "/extract-template-raw":
            self._write(
                handler,
                200,
                {
                    "template_base64": base64.b64encode(b"template").decode("ascii"),
                    "template_sha256": self.raw_template_sha256,
                    "template_format": "sourceafis",
                    "template_version": "3.18.1",
                    "sourceafis_version": "3.18.1",
                    "effective_dpi": payload["dpi"],
                    "native_width": payload["width"],
                    "native_height": payload["height"],
                    "method_internal_ms": self.extract_internal_ms,
                },
            )
            return
        if handler.path == "/verify":
            self._write(
                handler,
                200,
                {"raw_score": 77.25, "sourceafis_version": "3.18.1", "method_internal_ms": 2.75},
            )
            return
        self._write(handler, 404, {"error_code": "not_found", "error_message": "missing"})

    def _write(self, handler: BaseHTTPRequestHandler, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)


def _health_payload(port: int) -> dict[str, object]:
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
        "port": port,
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
