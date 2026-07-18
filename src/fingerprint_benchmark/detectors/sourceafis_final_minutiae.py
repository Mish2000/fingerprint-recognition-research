"""SourceAFIS final-template minutia locations for ``detector_only_v1``."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Mapping

import numpy as np

from fingerprint_benchmark.contract import MethodMetadata
from fingerprint_benchmark.local_features.detector_only import (
    DetectorOnlyAdapter,
    DetectorOnlyProtocolConfig,
)
from fingerprint_benchmark.sourceafis_client import (
    EXPECTED_SOURCEAFIS_VERSION,
    FINAL_MINUTIAE_COORDINATE_SPACE,
    FINAL_MINUTIAE_ENDPOINT,
    FINAL_MINUTIAE_INPUT,
    FINAL_MINUTIAE_ORDER_SEMANTICS,
    FINAL_MINUTIAE_SELECTION_SEMANTICS,
    SourceAfisFinalMinutiae,
    SourceAfisHealth,
    SourceAfisSidecarClient,
    validate_health,
)

from .types import DetectedPoint, DetectorResult


DETECTOR_NAME = "sourceafis_final_minutiae"
DETECTOR_VERSION = "sourceafis-final-minutiae-3.18.1-v1"
METHOD_NAME = "sourceafis_final_minutiae_rootsift_geometric"
METHOD_VERSION = "sourceafis-final-minutiae-3.18.1-rootsift-geometric-detector-only-v1"
COORDINATE_MAPPING = "scaled_to_native_pixel_center_v1"
SERIALIZATION_ORDER = "y_x_type_direction_source_index"


def scaled_to_native_pixel_center(
    coordinate: int,
    *,
    native_size: int,
    scaled_size: int,
) -> float:
    """Map one SourceAFIS scaled-pixel center to the native pixel space."""

    if isinstance(coordinate, bool) or not isinstance(coordinate, int):
        raise ValueError("Scaled coordinate must be an integer.")
    if coordinate < 0 or coordinate >= scaled_size:
        raise ValueError("Scaled coordinate is outside the scaled image.")
    if native_size <= 0 or scaled_size <= 0:
        raise ValueError("Native and scaled dimensions must be positive.")
    mapped = ((coordinate + 0.5) * native_size / scaled_size) - 0.5
    if not math.isfinite(mapped) or mapped < 0.0 or mapped >= native_size:
        raise ValueError("Pixel-center mapping produced an out-of-bounds native coordinate.")
    return float(mapped)


class SourceAfisFinalMinutiaeDetector:
    """Return deterministic final-template minutia locations from a live sidecar."""

    detector_name = DETECTOR_NAME
    detector_version = DETECTOR_VERSION
    reject_points_above_protocol_maximum = True

    def __init__(
        self,
        client: SourceAfisSidecarClient,
        *,
        health: SourceAfisHealth,
    ) -> None:
        validate_health(health)
        self._client = client
        self._health = health
        self.config = {
            "endpoint": FINAL_MINUTIAE_ENDPOINT,
            "input_contract": FINAL_MINUTIAE_INPUT,
            "coordinate_space": FINAL_MINUTIAE_COORDINATE_SPACE,
            "coordinate_mapping": COORDINATE_MAPPING,
            "selection_semantics": FINAL_MINUTIAE_SELECTION_SEMANTICS,
            "ranking_semantics": "none",
            "serialization_order": SERIALIZATION_ORDER,
            "response_sentinel": 0.0,
            "mask_support": "none_fail_explicitly",
        }

    def detect(
        self,
        image: np.ndarray,
        image_metadata: Mapping[str, object],
        mask: np.ndarray | None = None,
    ) -> DetectorResult:
        if mask is not None:
            raise ValueError("SourceAFIS final-minutiae detector does not support masks.")
        total_started = perf_counter_ns()
        source = np.asarray(image)
        if source.ndim != 2 or source.size == 0 or source.dtype != np.uint8:
            raise ValueError("SourceAFIS final-minutiae detector requires a non-empty uint8 grayscale image.")
        height, width = (int(source.shape[0]), int(source.shape[1]))
        dpi = _manifest_ppi(image_metadata)

        serialization_started = perf_counter_ns()
        contiguous = np.ascontiguousarray(source, dtype=np.uint8)
        pixels = contiguous.tobytes(order="C")
        pixel_serialization_ms = _elapsed(serialization_started)

        request_started = perf_counter_ns()
        extracted = self._client.extract_final_minutiae(pixels, width, height, dpi)
        sidecar_request_wall_ms = _elapsed(request_started)
        if (extracted.native_width, extracted.native_height) != (width, height):
            raise ValueError("SourceAFIS response native dimensions do not match the input image.")

        mapping_started = perf_counter_ns()
        points = _mapped_points(extracted)
        coordinate_mapping_ms = _elapsed(mapping_started)
        detector_total_ms = _elapsed(total_started)
        timing = {
            "pixel_serialization_ms": pixel_serialization_ms,
            "sidecar_request_wall_ms": sidecar_request_wall_ms,
            "sourceafis_method_internal_ms": extracted.method_internal_ms,
            "coordinate_mapping_ms": coordinate_mapping_ms,
            "detector_total_ms": detector_total_ms,
        }
        return DetectorResult(
            points=points,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            detector_config=dict(self.config),
            diagnostics={
                **timing,
                "native_width": width,
                "native_height": height,
                "scaled_width": extracted.scaled_width,
                "scaled_height": extracted.scaled_height,
                "minutia_count": extracted.minutia_count,
                "template_sha256": extracted.template_sha256,
            },
            detector_time_ms=detector_total_ms,
            metadata={
                "sourceafis_version": extracted.sourceafis_version,
                "template_version": extracted.template_version,
                "sourceafis_stage": "final_template_minutiae",
                "selection_semantics": extracted.selection_semantics,
                "source_order_semantics": extracted.source_order_semantics,
                "ranking_semantics": "none",
                "serialization_order": SERIALIZATION_ORDER,
                "coordinate_mapping": COORDINATE_MAPPING,
                "timing": timing,
            },
        )

    def implementation_source_paths(self) -> tuple[Path, ...]:
        return (
            Path("src/fingerprint_benchmark/sourceafis_client.py"),
            Path("src/fingerprint_benchmark/sourceafis_sidecar.py"),
            Path("src/fingerprint_benchmark/detectors/sourceafis_final_minutiae.py"),
            Path("apps/sourceafis-sidecar/pom.xml"),
            Path("apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/ApiException.java"),
            Path("apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/BuildInfo.java"),
            Path("apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/SourceAfisSidecarService.java"),
            Path("apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/SourceAfisV2Engine.java"),
            Path("apps/sourceafis-sidecar/src/main/resources/sourceafis-sidecar.properties"),
        )


class SourceAfisFinalMinutiaeRootSIFTGeometricAdapter(DetectorOnlyAdapter):
    """Public SourceAFIS-locations -> common RootSIFT/geometric method."""

    def __init__(
        self,
        client: SourceAfisSidecarClient,
        *,
        health: SourceAfisHealth,
        protocol_config: DetectorOnlyProtocolConfig | None = None,
    ) -> None:
        active_protocol = protocol_config or DetectorOnlyProtocolConfig()
        if active_protocol.descriptor != "rootsift":
            raise ValueError(
                "SourceAfisFinalMinutiaeRootSIFTGeometricAdapter requires descriptor='rootsift'."
            )
        self._health = health
        super().__init__(
            detector=SourceAfisFinalMinutiaeDetector(client, health=health),
            config=active_protocol,
            method_name=METHOD_NAME,
            method_version=METHOD_VERSION,
        )

    def required_runtime_artifacts(self) -> tuple[str, ...]:
        return ("sidecar_jar_sha256",)

    def metadata(self) -> MethodMetadata:
        base = super().metadata()
        sourceafis = {
            "sourceafis_version": EXPECTED_SOURCEAFIS_VERSION,
            "maven_coordinates": self._health.raw["maven_coordinates"],
            "sidecar_contract_version": self._health.contract_version,
            "sidecar_implementation_version": self._health.raw["sidecar_implementation_version"],
            "endpoint": FINAL_MINUTIAE_ENDPOINT,
            "raw_input_contract": FINAL_MINUTIAE_INPUT,
            "coordinate_mapping": COORDINATE_MAPPING,
            "selection_semantics": FINAL_MINUTIAE_SELECTION_SEMANTICS,
            "serialization_order": SERIALIZATION_ORDER,
            "required_runtime_artifacts": ["sidecar_jar_sha256"],
        }
        return replace(
            base,
            implementation_provenance={**base.implementation_provenance, **sourceafis},
            config={**base.config, **sourceafis},
            runtime={**base.runtime, "sourceafis_health": dict(self._health.raw)},
        )


def _mapped_points(extracted: SourceAfisFinalMinutiae) -> tuple[DetectedPoint, ...]:
    mapped: list[tuple[float, float, object]] = []
    for minutia in extracted.minutiae:
        x_native = scaled_to_native_pixel_center(
            minutia.x_scaled,
            native_size=extracted.native_width,
            scaled_size=extracted.scaled_width,
        )
        y_native = scaled_to_native_pixel_center(
            minutia.y_scaled,
            native_size=extracted.native_height,
            scaled_size=extracted.scaled_height,
        )
        mapped.append((x_native, y_native, minutia))
    mapped.sort(
        key=lambda item: (
            item[1],
            item[0],
            item[2].minutia_type,
            item[2].direction_radians,
            item[2].source_index,
        )
    )
    return tuple(
        DetectedPoint(
            x=x_native,
            y=y_native,
            response=0.0,
            detector_angle=math.degrees(minutia.direction_radians),
            detector_metadata={
                "minutia_type": minutia.minutia_type,
                "direction_radians": minutia.direction_radians,
                "x_scaled": minutia.x_scaled,
                "y_scaled": minutia.y_scaled,
                "source_index": minutia.source_index,
                "sourceafis_stage": "final_template_minutiae",
            },
        )
        for x_native, y_native, minutia in mapped
    )


def _manifest_ppi(metadata: Mapping[str, object]) -> float:
    raw = metadata.get("ppi")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("SourceAFIS final-minutiae detector requires numeric manifest PPI.")
    ppi = float(raw)
    if not math.isfinite(ppi) or ppi < 100.0 or ppi > 4000.0:
        raise ValueError("SourceAFIS final-minutiae detector manifest PPI is outside the supported range.")
    return ppi


def _elapsed(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


__all__ = [
    "COORDINATE_MAPPING",
    "DETECTOR_NAME",
    "DETECTOR_VERSION",
    "METHOD_NAME",
    "METHOD_VERSION",
    "SERIALIZATION_ORDER",
    "SourceAfisFinalMinutiaeDetector",
    "SourceAfisFinalMinutiaeRootSIFTGeometricAdapter",
    "scaled_to_native_pixel_center",
]
