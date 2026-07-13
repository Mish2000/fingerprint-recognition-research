"""SourceAFIS method adapter for pairwise benchmark runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .contract import CompareOutcome, MethodExecutionError, MethodMetadata, PrepareOutcome, PreparedRepresentation
from .sourceafis_client import (
    SOURCEAFIS_MAVEN_COORDINATES,
    SourceAfisHealth,
    SourceAfisSidecarClient,
    validate_health,
)


DPI_POLICY = {
    "required": True,
    "metadata_keys": ["ppi", "dpi"],
    "preferred_key": "ppi",
    "min_dpi": 100,
    "max_dpi": 4000,
    "missing_policy": "fail_explicitly",
    "invalid_policy": "fail_explicitly",
    "silent_default": False,
}


class SourceAfisAdapter:
    method = "sourceafis"
    official_implementation_family = "Java"
    representation_format = "sourceafis"
    template_format = representation_format

    def __init__(
        self,
        client: SourceAfisSidecarClient,
        *,
        health: SourceAfisHealth | None = None,
    ) -> None:
        self._client = client
        self._health = health or client.health()
        validate_health(self._health)
        self._method_version = self._health.method_version

    def metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method=self.method,
            method_version=self._method_version,
            implementation_provenance={
                "official_implementation_family": self.official_implementation_family,
                "engine": "SourceAFIS",
                "maven_coordinates": SOURCEAFIS_MAVEN_COORDINATES,
                "sourceafis_version": self._method_version,
                "sidecar_contract_version": self._health.contract_version,
                "sidecar_implementation_version": self._health.raw["sidecar_implementation_version"],
                "sidecar_implementation_provenance": "apps/sourceafis-sidecar",
                "external_preprocessing": self._health.raw["external_preprocessing"],
                "template_cache": self._health.raw["template_cache"],
                "raw_score_semantics": "SourceAFIS raw similarity score from FingerprintMatcher.match",
                "method_internal_timing": {
                    "unit": self._health.raw["method_internal_timing_unit"],
                    "extract_template_scope": self._health.raw["extract_template_internal_timing_scope"],
                    "verify_scope": self._health.raw["verify_internal_timing_scope"],
                },
            },
            config={
                "adapter": "fingerprint_benchmark.sourceafis_adapter.SourceAfisAdapter",
                "official_implementation_family": self.official_implementation_family,
                "maven_coordinates": self._health.raw["maven_coordinates"],
                "sourceafis_version": self._health.sourceafis_version,
                "sidecar_contract_version": self._health.contract_version,
                "sidecar_implementation_version": self._health.raw["sidecar_implementation_version"],
                "template_format": self._health.raw["template_format"],
                "template_version": self._health.raw["template_version"],
                "dpi_policy": DPI_POLICY,
                "sidecar_dpi_policy": self._health.raw["dpi_policy"],
                "transport": self._health.raw["transport"],
                "external_preprocessing": self._health.raw["external_preprocessing"],
                "template_cache": self._health.raw["template_cache"],
                "supports_template_extraction": self._health.raw["supports_template_extraction"],
                "supports_pairwise_verification": self._health.raw["supports_pairwise_verification"],
                "supports_identification": self._health.raw["supports_identification"],
                "normalization": "none",
                "thresholding": "none",
                "decision_logic": "none",
            },
            runtime={
                **self._health.raw,
                "process_lifecycle_policy": (
                    "A dedicated JVM starts before one dataset/protocol run and remains alive for that run; "
                    "no subprocess per pair, prepare, or compare."
                ),
                "timing_inclusion_exclusion_policy": (
                    "Adapter wall timings include image reads, Base64, JSON, and HTTP transport. Method-internal "
                    "timings use the exact sidecar scopes and exclude sidecar startup, health validation, and shutdown."
                ),
            },
            score_direction="higher_is_more_similar",
            score_semantics=(
                "Unnormalized SourceAFIS similarity score returned by FingerprintMatcher.match; no threshold or "
                "decision is applied."
            ),
        )

    def prepare(self, image_path: Path, image_metadata: Mapping[str, Any]) -> PrepareOutcome:
        dpi = _effective_dpi(image_metadata)
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise MethodExecutionError("image_read_failure", f"Cannot read image file {image_path}: {exc}") from exc
        template = self._client.extract_template(image_bytes, dpi)
        if template.template_format != self.representation_format:
            raise MethodExecutionError(
                "protocol_contract_mismatch",
                f"SourceAFIS template format mismatch: {template.template_format!r}.",
            )
        if template.sourceafis_version != self._method_version or template.template_version != self._method_version:
            raise MethodExecutionError(
                "sourceafis_version_mismatch",
                "SourceAFIS template version does not match the validated runtime version.",
            )
        if template.effective_dpi != dpi:
            raise MethodExecutionError(
                "protocol_contract_mismatch",
                "SourceAFIS effective DPI does not match the requested DPI.",
            )
        representation = PreparedRepresentation(
            method=self.method,
            method_version=self._method_version,
            representation_format=template.template_format,
            representation_version=template.template_version,
            payload=template.template_base64,
            metadata={"effective_dpi": template.effective_dpi},
        )
        return PrepareOutcome(
            representation=representation,
            method_internal_ms=template.method_internal_ms,
        )

    def compare(
        self,
        representation_a: PreparedRepresentation,
        representation_b: PreparedRepresentation,
    ) -> CompareOutcome:
        _validate_representation(representation_a, self._method_version)
        _validate_representation(representation_b, self._method_version)
        verification = self._client.verify(representation_a.payload, representation_b.payload)
        if verification.sourceafis_version != self._method_version:
            raise MethodExecutionError(
                "sourceafis_version_mismatch",
                "SourceAFIS verification version does not match the validated runtime version.",
            )
        if not math.isfinite(verification.raw_score):
            raise MethodExecutionError("non_finite_raw_score", "SourceAFIS returned a non-finite raw score.")
        return CompareOutcome(
            raw_score=float(verification.raw_score),
            method_internal_ms=verification.method_internal_ms,
        )

    def close(self) -> None:
        self._client.close()


def _effective_dpi(image_metadata: Mapping[str, Any]) -> float:
    raw = image_metadata.get("ppi", image_metadata.get("dpi"))
    if raw in (None, ""):
        raise MethodExecutionError("missing_dpi", "DPI/PPI metadata is required for SourceAFIS preparation.")
    try:
        dpi = float(raw)
    except (TypeError, ValueError) as exc:
        raise MethodExecutionError("invalid_dpi", f"DPI/PPI metadata must be numeric; got {raw!r}.") from exc
    if not math.isfinite(dpi) or dpi < DPI_POLICY["min_dpi"] or dpi > DPI_POLICY["max_dpi"]:
        raise MethodExecutionError(
            "invalid_dpi",
            "DPI/PPI metadata must be finite and within the documented SourceAFIS adapter policy range.",
        )
    return dpi


def _validate_representation(representation: PreparedRepresentation, expected_version: str) -> None:
    if representation.method != SourceAfisAdapter.method:
        raise MethodExecutionError("invalid_serialized_template", "Representation was not produced by SourceAFIS.")
    if representation.method_version != expected_version or representation.representation_version != expected_version:
        raise MethodExecutionError(
            "sourceafis_version_mismatch",
            "Representation SourceAFIS version does not match the active runtime.",
        )
    if (
        representation.representation_format != SourceAfisAdapter.representation_format
        or not isinstance(representation.payload, str)
        or not representation.payload
    ):
        raise MethodExecutionError("invalid_serialized_template", "Representation is not a valid SourceAFIS template.")
