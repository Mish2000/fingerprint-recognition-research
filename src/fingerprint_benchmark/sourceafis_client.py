"""HTTP client for the local SourceAFIS Java sidecar v2."""

from __future__ import annotations

import base64
import http.client
import json
import math
import re
import socket
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

from .contract import MethodExecutionError


EXPECTED_SOURCEAFIS_VERSION = "3.18.1"
EXPECTED_CONTRACT_VERSION = "sourceafis-sidecar-v2.2"
EXPECTED_SIDECAR_IMPLEMENTATION_VERSION = "0.3.0"
SOURCEAFIS_MAVEN_COORDINATES = "com.machinezoo.sourceafis:sourceafis:3.18.1"
SOURCEAFIS_TEMPLATE_FORMAT = "sourceafis"
SOURCEAFIS_TRANSPORT = "localhost_http"
METHOD_INTERNAL_TIMING_UNIT = "milliseconds"
EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE = (
    "FingerprintImageOptions construction, FingerprintImage construction, FingerprintTemplate extraction, "
    "and FingerprintTemplate.toByteArray serialization; excludes HTTP, JSON, request Base64 decoding, "
    "and response Base64 encoding."
)
VERIFY_INTERNAL_TIMING_SCOPE = (
    "FingerprintTemplate deserialization for both templates, FingerprintMatcher construction, and "
    "FingerprintMatcher.match; excludes HTTP, JSON, and request Base64 decoding."
)
EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE = (
    "FingerprintImageOptions construction, raw FingerprintImage construction, FingerprintTemplate extraction, "
    "FingerprintTemplate.toByteArray serialization, documented native-template CBOR parsing, and response "
    "model construction; excludes HTTP, JSON, request Base64 decoding, and response JSON serialization."
)
FINAL_MINUTIAE_ENDPOINT = "/extract-final-minutiae"
FINAL_MINUTIAE_INPUT = "raw_uint8_grayscale_row_major"
FINAL_MINUTIAE_COORDINATE_SPACE = "sourceafis_500_dpi_scaled_image"
FINAL_MINUTIAE_SELECTION_STAGE = "sourceafis_final_template_minutiae"
FINAL_MINUTIAE_HEALTH_STAGE = "final_template_minutiae"
FINAL_MINUTIAE_SELECTION_SEMANTICS = "sourceafis_final_selected_minutia_set"
FINAL_MINUTIAE_ORDER_SEMANTICS = "deterministic_sourceafis_template_order_not_quality_ranking"
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
ALLOWED_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class SourceAfisClientError(MethodExecutionError):
    """Expected SourceAFIS transport or sidecar failure."""


class SourceAfisContractError(SourceAfisClientError):
    """Raised when the sidecar response does not match the v2 contract."""


@dataclass(frozen=True)
class SourceAfisEndpoint:
    service_url: str
    host: str
    port: int
    base_path: str


@dataclass(frozen=True)
class SourceAfisHealth:
    raw: dict[str, Any]

    @property
    def sourceafis_version(self) -> str:
        return str(self.raw.get("sourceafis_version") or "")

    @property
    def contract_version(self) -> str:
        return str(self.raw.get("contract_version") or "")

    @property
    def method_version(self) -> str:
        return str(self.raw.get("method_version") or self.sourceafis_version)


@dataclass(frozen=True)
class SourceAfisTemplate:
    template_base64: str
    template_format: str
    template_version: str
    sourceafis_version: str
    effective_dpi: float
    method_internal_ms: float


@dataclass(frozen=True)
class SourceAfisVerification:
    raw_score: float
    sourceafis_version: str
    method_internal_ms: float


@dataclass(frozen=True, slots=True)
class SourceAfisMinutia:
    source_index: int
    x_scaled: int
    y_scaled: int
    direction_radians: float
    minutia_type: str


@dataclass(frozen=True, slots=True)
class SourceAfisFinalMinutiae:
    sourceafis_version: str
    template_version: str
    effective_dpi: float
    native_width: int
    native_height: int
    scaled_width: int
    scaled_height: int
    coordinate_space: str
    selection_stage: str
    selection_semantics: str
    source_order_semantics: str
    template_sha256: str
    minutiae: tuple[SourceAfisMinutia, ...]
    method_internal_ms: float

    @property
    def minutia_count(self) -> int:
        return len(self.minutiae)


class SourceAfisSidecarClient:
    """Small persistent HTTP client for the sidecar.

    The client intentionally exposes only health, template/final-minutia
    extraction, and pairwise verification. It does not provide fallback
    matching or identification.
    """

    def __init__(self, service_url: str, *, timeout_seconds: float = 120.0) -> None:
        endpoint = parse_sourceafis_service_url(service_url)
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise SourceAfisClientError(
                "runtime_unavailable",
                "SourceAFIS sidecar timeout must be a positive finite number.",
            ) from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise SourceAfisClientError(
                "runtime_unavailable",
                "SourceAFIS sidecar timeout must be a positive finite number.",
            )
        self.service_url = endpoint.service_url
        self._base_path = endpoint.base_path
        self._host = endpoint.host
        self._port = endpoint.port
        self._timeout_seconds = timeout
        self._connection = http.client.HTTPConnection(
            self._host,
            self._port,
            timeout=self._timeout_seconds,
        )
        self.request_count = 0
        self.health_request_count = 0

    def close(self) -> None:
        self._connection.close()

    def health(self) -> SourceAfisHealth:
        self.health_request_count += 1
        payload = self._request_json("GET", "/health")
        if payload.get("status") != "ok":
            raise SourceAfisContractError(
                "runtime_unavailable",
                f"SourceAFIS sidecar reported status {payload.get('status')!r}.",
            )
        return SourceAfisHealth(payload)

    def extract_template(self, image_bytes: bytes, dpi: float) -> SourceAfisTemplate:
        payload = self._request_json(
            "POST",
            "/extract-template",
            {
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "dpi": float(dpi),
            },
        )
        template_base64 = _required_str(payload, "template_base64", "template_extraction_failure")
        _validate_base64(template_base64, "template_base64", "template_extraction_failure")
        effective_dpi = _required_float(payload, "effective_dpi", "template_extraction_failure")
        return SourceAfisTemplate(
            template_base64=template_base64,
            template_format=_required_str(payload, "template_format", "template_extraction_failure"),
            template_version=_required_str(payload, "template_version", "template_extraction_failure"),
            sourceafis_version=_required_str(payload, "sourceafis_version", "template_extraction_failure"),
            effective_dpi=effective_dpi,
            method_internal_ms=_required_nonnegative_float(
                payload,
                "method_internal_ms",
                "template_extraction_failure",
            ),
        )

    def verify(self, template_a_base64: str, template_b_base64: str) -> SourceAfisVerification:
        payload = self._request_json(
            "POST",
            "/verify",
            {
                "template_a_base64": template_a_base64,
                "template_b_base64": template_b_base64,
            },
        )
        score = _required_float(payload, "raw_score", "comparison_failure")
        if not math.isfinite(score):
            raise SourceAfisClientError("non_finite_raw_score", "SourceAFIS sidecar returned a non-finite raw score.")
        return SourceAfisVerification(
            raw_score=score,
            sourceafis_version=_required_str(payload, "sourceafis_version", "comparison_failure"),
            method_internal_ms=_required_nonnegative_float(payload, "method_internal_ms", "comparison_failure"),
        )

    def extract_final_minutiae(
        self,
        pixels: bytes,
        width: int,
        height: int,
        dpi: float,
    ) -> SourceAfisFinalMinutiae:
        """Extract the final selected minutia set from exact raw grayscale bytes."""

        if not isinstance(pixels, bytes):
            raise SourceAfisClientError("invalid_raw_pixels", "SourceAFIS raw pixels must be bytes.")
        checked_width = _positive_int(width, "width", "invalid_dimensions")
        checked_height = _positive_int(height, "height", "invalid_dimensions")
        if len(pixels) != checked_width * checked_height:
            raise SourceAfisClientError(
                "pixel_length_mismatch",
                "SourceAFIS raw pixel length must equal width * height exactly.",
            )
        checked_dpi = _finite_float(dpi, "dpi", "invalid_dpi")
        if checked_dpi < 100.0 or checked_dpi > 4000.0:
            raise SourceAfisClientError("invalid_dpi", "SourceAFIS DPI is outside the supported range.")
        payload = self._request_json(
            "POST",
            FINAL_MINUTIAE_ENDPOINT,
            {
                "width": checked_width,
                "height": checked_height,
                "pixels_base64": base64.b64encode(pixels).decode("ascii"),
                "dpi": checked_dpi,
            },
        )
        return _parse_final_minutiae_response(
            payload,
            expected_width=checked_width,
            expected_height=checked_height,
            expected_dpi=checked_dpi,
        )

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.request_count += 1
        request_path = f"{self._base_path}{path}" if self._base_path else path
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
            headers["Content-Length"] = str(len(body))
        try:
            self._connection.request(method, request_path, body=body, headers=headers)
            response = self._connection.getresponse()
            response_body = response.read()
        except socket.timeout as exc:
            raise SourceAfisClientError("timeout", f"SourceAFIS sidecar request timed out: {method} {path}.") from exc
        except OSError as exc:
            raise SourceAfisClientError(
                "runtime_unavailable",
                f"SourceAFIS sidecar is not reachable at {self.service_url}: {exc}",
            ) from exc

        try:
            decoded = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceAfisContractError(
                "protocol_contract_mismatch",
                f"SourceAFIS sidecar returned non-JSON response for {method} {path}.",
            ) from exc
        if not isinstance(decoded, dict):
            raise SourceAfisContractError(
                "protocol_contract_mismatch",
                f"SourceAFIS sidecar returned {type(decoded).__name__}; expected JSON object.",
            )
        if response.status >= 400:
            code = str(decoded.get("error_code") or decoded.get("error") or _status_error_code(response.status))
            message = str(decoded.get("error_message") or decoded.get("detail") or decoded.get("message") or code)
            raise SourceAfisClientError(code, message)
        return decoded


def validate_health(
    health: SourceAfisHealth,
    *,
    expected_version: str = EXPECTED_SOURCEAFIS_VERSION,
    expected_contract_version: str = EXPECTED_CONTRACT_VERSION,
    expected_implementation_version: str = EXPECTED_SIDECAR_IMPLEMENTATION_VERSION,
) -> None:
    payload = health.raw
    expected_fields = {
        "status": "ok",
        "method": "sourceafis",
        "official_implementation_family": "Java",
        "engine": "SourceAFIS",
        "method_version": expected_version,
        "maven_coordinates": SOURCEAFIS_MAVEN_COORDINATES,
        "template_format": SOURCEAFIS_TEMPLATE_FORMAT,
        "template_version": expected_version,
        "contract_version": expected_contract_version,
        "sidecar_implementation_version": expected_implementation_version,
        "transport": SOURCEAFIS_TRANSPORT,
        "external_preprocessing": "none",
        "method_internal_timing_unit": METHOD_INTERNAL_TIMING_UNIT,
        "extract_template_internal_timing_scope": EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE,
        "extract_final_minutiae_internal_timing_scope": EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE,
        "verify_internal_timing_scope": VERIFY_INTERNAL_TIMING_SCOPE,
        "final_minutiae_endpoint": FINAL_MINUTIAE_ENDPOINT,
        "final_minutiae_input": FINAL_MINUTIAE_INPUT,
        "final_minutiae_coordinate_space": FINAL_MINUTIAE_COORDINATE_SPACE,
        "final_minutiae_stage": FINAL_MINUTIAE_HEALTH_STAGE,
    }
    for field_name, expected_value in expected_fields.items():
        if payload.get(field_name) != expected_value:
            raise SourceAfisContractError(
                "protocol_contract_mismatch",
                f"SourceAFIS sidecar health field {field_name!r} mismatch: "
                f"expected {expected_value!r}, got {payload.get(field_name)!r}.",
            )
    if health.contract_version != expected_contract_version:
        raise SourceAfisContractError(
            "protocol_contract_mismatch",
            f"SourceAFIS sidecar contract version mismatch: expected {expected_contract_version}, got {health.contract_version}.",
        )
    if health.sourceafis_version != expected_version:
        raise SourceAfisContractError(
            "sourceafis_version_mismatch",
            f"SourceAFIS version mismatch: expected {expected_version}, got {health.sourceafis_version}.",
        )
    if payload.get("maven_coordinates") != SOURCEAFIS_MAVEN_COORDINATES:
        raise SourceAfisContractError(
            "sourceafis_version_mismatch",
            "SourceAFIS Maven coordinates do not match the pinned research baseline.",
        )
    expected_boolean_fields = {
        "template_cache": False,
        "supports_template_extraction": True,
        "supports_final_minutiae_extraction": True,
        "supports_pairwise_verification": True,
        "supports_identification": False,
    }
    for field_name, expected_value in expected_boolean_fields.items():
        if payload.get(field_name) is not expected_value:
            raise SourceAfisContractError(
                "protocol_contract_mismatch",
                f"SourceAFIS sidecar health field {field_name!r} must be {expected_value!r}.",
            )
    bind_host = payload.get("bind_host")
    if not isinstance(bind_host, str) or bind_host.strip().lower() not in ALLOWED_LOOPBACK_HOSTS:
        raise SourceAfisContractError(
            "protocol_contract_mismatch",
            "SourceAFIS sidecar health reports a non-loopback bind host.",
        )
    port = payload.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or port <= 0 or port > 65535:
        raise SourceAfisContractError(
            "protocol_contract_mismatch",
            "SourceAFIS sidecar health reports an invalid port.",
        )
    for field_name in ("java_runtime_version", "java_runtime_vendor"):
        _required_str(payload, field_name, "protocol_contract_mismatch")
    _validate_health_dpi_policy(payload.get("dpi_policy"))


def parse_sourceafis_service_url(service_url: str) -> SourceAfisEndpoint:
    """Parse a sidecar URL while enforcing local-only plain-HTTP transport."""

    raw_url = str(service_url or "").strip()
    try:
        parsed = urlparse(raw_url)
        host = (parsed.hostname or "").lower()
        explicit_port = parsed.port
    except ValueError as exc:
        raise SourceAfisClientError(
            "runtime_unavailable",
            "SourceAFIS sidecar URL is invalid.",
        ) from exc
    if parsed.scheme != "http" or not host:
        raise SourceAfisClientError(
            "runtime_unavailable",
            "SourceAFIS sidecar URL must be an http:// loopback host URL.",
        )
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment or parsed.params:
        raise SourceAfisClientError(
            "runtime_unavailable",
            "SourceAFIS sidecar URL must not contain credentials, parameters, a query, or a fragment.",
        )
    if host not in ALLOWED_LOOPBACK_HOSTS:
        raise SourceAfisClientError(
            "remote_transport_forbidden",
            "SourceAFIS biometric payloads may only be sent to localhost, 127.0.0.1, or ::1.",
        )
    port = explicit_port if explicit_port is not None else 80
    if port <= 0 or port > 65535:
        raise SourceAfisClientError("runtime_unavailable", "SourceAFIS sidecar URL has an invalid port.")
    base_path = parsed.path.rstrip("/")
    display_host = f"[{host}]" if ":" in host else host
    authority = f"{display_host}:{port}" if explicit_port is not None else display_host
    return SourceAfisEndpoint(
        service_url=f"http://{authority}{base_path}",
        host=host,
        port=port,
        base_path=base_path,
    )


def _required_str(payload: dict[str, Any], field_name: str, error_code: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} is not a string.")
    text = value.strip()
    if not text:
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} is empty.")
    return text


def _parse_final_minutiae_response(
    payload: dict[str, Any],
    *,
    expected_width: int,
    expected_height: int,
    expected_dpi: float,
) -> SourceAfisFinalMinutiae:
    error_code = "final_minutiae_contract_mismatch"
    expected_fields = {
        "sourceafis_version",
        "template_version",
        "effective_dpi",
        "native_width",
        "native_height",
        "scaled_width",
        "scaled_height",
        "coordinate_space",
        "selection_stage",
        "selection_semantics",
        "source_order_semantics",
        "template_sha256",
        "minutia_count",
        "minutiae",
        "method_internal_ms",
    }
    _require_exact_fields(payload, expected_fields, "final-minutiae response", error_code)
    sourceafis_version = _required_str(payload, "sourceafis_version", error_code)
    if sourceafis_version != EXPECTED_SOURCEAFIS_VERSION:
        raise SourceAfisContractError(
            "sourceafis_version_mismatch",
            f"SourceAFIS final-minutiae version mismatch: {sourceafis_version!r}.",
        )
    template_version = _required_str(payload, "template_version", error_code)
    if not template_version.startswith(f"{EXPECTED_SOURCEAFIS_VERSION}-"):
        raise SourceAfisContractError(
            "sourceafis_version_mismatch",
            f"SourceAFIS native template version mismatch: {template_version!r}.",
        )
    native_width = _required_positive_int(payload, "native_width", error_code)
    native_height = _required_positive_int(payload, "native_height", error_code)
    if (native_width, native_height) != (expected_width, expected_height):
        raise SourceAfisContractError(error_code, "SourceAFIS final-minutiae native dimensions mismatch.")
    scaled_width = _required_positive_int(payload, "scaled_width", error_code)
    scaled_height = _required_positive_int(payload, "scaled_height", error_code)
    effective_dpi = _required_float(payload, "effective_dpi", error_code)
    if effective_dpi != expected_dpi:
        raise SourceAfisContractError(error_code, "SourceAFIS final-minutiae effective DPI mismatch.")
    exact_strings = {
        "coordinate_space": FINAL_MINUTIAE_COORDINATE_SPACE,
        "selection_stage": FINAL_MINUTIAE_SELECTION_STAGE,
        "selection_semantics": FINAL_MINUTIAE_SELECTION_SEMANTICS,
        "source_order_semantics": FINAL_MINUTIAE_ORDER_SEMANTICS,
    }
    for field_name, expected in exact_strings.items():
        actual = _required_str(payload, field_name, error_code)
        if actual != expected:
            raise SourceAfisContractError(
                error_code,
                f"SourceAFIS final-minutiae field {field_name!r} mismatch: {actual!r}.",
            )
    template_sha256 = _required_str(payload, "template_sha256", error_code)
    if _SHA256_PATTERN.fullmatch(template_sha256) is None:
        raise SourceAfisContractError(error_code, "SourceAFIS template_sha256 must be lowercase SHA-256 hex.")
    count = _required_nonnegative_int(payload, "minutia_count", error_code)
    raw_minutiae = payload.get("minutiae")
    if not isinstance(raw_minutiae, list):
        raise SourceAfisContractError(error_code, "SourceAFIS final-minutiae field 'minutiae' must be an array.")
    if len(raw_minutiae) != count:
        raise SourceAfisContractError(error_code, "SourceAFIS final-minutiae count does not match the array.")
    minutiae: list[SourceAfisMinutia] = []
    for index, raw in enumerate(raw_minutiae):
        if not isinstance(raw, dict):
            raise SourceAfisContractError(error_code, f"SourceAFIS minutia {index} must be an object.")
        _require_exact_fields(
            raw,
            {"source_index", "x_scaled", "y_scaled", "direction_radians", "type"},
            f"minutia {index}",
            error_code,
        )
        source_index = _required_nonnegative_int(raw, "source_index", error_code)
        if source_index != index:
            raise SourceAfisContractError(error_code, "SourceAFIS source_index must be contiguous template order.")
        x_scaled = _required_nonnegative_int(raw, "x_scaled", error_code)
        y_scaled = _required_nonnegative_int(raw, "y_scaled", error_code)
        if x_scaled >= scaled_width or y_scaled >= scaled_height:
            raise SourceAfisContractError(error_code, "SourceAFIS minutia coordinate is outside scaled bounds.")
        direction = _required_float(raw, "direction_radians", error_code)
        minutia_type = _required_str(raw, "type", error_code)
        if minutia_type not in {"ENDING", "BIFURCATION"}:
            raise SourceAfisContractError(error_code, "SourceAFIS minutia type is invalid.")
        minutiae.append(
            SourceAfisMinutia(
                source_index=source_index,
                x_scaled=x_scaled,
                y_scaled=y_scaled,
                direction_radians=direction,
                minutia_type=minutia_type,
            )
        )
    return SourceAfisFinalMinutiae(
        sourceafis_version=sourceafis_version,
        template_version=template_version,
        effective_dpi=effective_dpi,
        native_width=native_width,
        native_height=native_height,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        coordinate_space=FINAL_MINUTIAE_COORDINATE_SPACE,
        selection_stage=FINAL_MINUTIAE_SELECTION_STAGE,
        selection_semantics=FINAL_MINUTIAE_SELECTION_SEMANTICS,
        source_order_semantics=FINAL_MINUTIAE_ORDER_SEMANTICS,
        template_sha256=template_sha256,
        minutiae=tuple(minutiae),
        method_internal_ms=_required_nonnegative_float(payload, "method_internal_ms", error_code),
    )


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
    error_code: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SourceAfisContractError(
            error_code,
            f"SourceAFIS {label} schema mismatch; missing={missing}, unexpected={unexpected}.",
        )


def _positive_int(value: Any, field_name: str, error_code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceAfisClientError(error_code, f"SourceAFIS field {field_name!r} must be a positive integer.")
    return value


def _finite_float(value: Any, field_name: str, error_code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceAfisClientError(error_code, f"SourceAFIS field {field_name!r} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SourceAfisClientError(error_code, f"SourceAFIS field {field_name!r} must be finite.")
    return parsed


def _required_positive_int(payload: Mapping[str, Any], field_name: str, error_code: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} must be positive integer.")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], field_name: str, error_code: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} must be non-negative integer.")
    return value


def _required_float(payload: dict[str, Any], field_name: str, error_code: str) -> float:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} is not numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} is not finite.")
    return parsed


def _required_nonnegative_float(payload: dict[str, Any], field_name: str, error_code: str) -> float:
    parsed = _required_float(payload, field_name, error_code)
    if parsed < 0:
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} is negative.")
    return parsed


def _validate_health_dpi_policy(value: Any) -> None:
    if not isinstance(value, dict):
        raise SourceAfisContractError(
            "protocol_contract_mismatch",
            "SourceAFIS sidecar health dpi_policy must be a JSON object.",
        )
    if value.get("required") is not True or value.get("silent_default") is not False:
        raise SourceAfisContractError(
            "protocol_contract_mismatch",
            "SourceAFIS sidecar health dpi_policy has invalid required/default behavior.",
        )
    if value.get("source") != "manifest_or_prepare_metadata":
        raise SourceAfisContractError(
            "protocol_contract_mismatch",
            "SourceAFIS sidecar health dpi_policy has an invalid source.",
        )
    for field_name, expected in (("min_dpi", 100.0), ("max_dpi", 4000.0)):
        raw = value.get(field_name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) != expected:
            raise SourceAfisContractError(
                "protocol_contract_mismatch",
                f"SourceAFIS sidecar health dpi_policy field {field_name!r} must be {expected}.",
            )


def _validate_base64(value: str, field_name: str, error_code: str) -> None:
    try:
        base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise SourceAfisContractError(error_code, f"SourceAFIS response field {field_name!r} is not valid base64.") from exc


def _status_error_code(status: int) -> str:
    if status == 408:
        return "timeout"
    if status >= 500:
        return "runtime_unavailable"
    return "protocol_contract_mismatch"
