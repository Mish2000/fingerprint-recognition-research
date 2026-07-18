"""Pairwise adapter for the isolated PPI-aware HarrisZ+ v4 method."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import cv2

from fingerprint_benchmark.contract import (
    MethodExecutionError,
    MethodMetadata,
    PrepareOutcome,
    PreparedRepresentation,
)
from fingerprint_benchmark.sift.extractor import SiftRepresentation

from .adapter import HarrisZPlusGeometricAdapter
from .extractor import _enforce_determinism
from .extractor_v4 import extract_representation_v4
from .ppi_aware_v4 import (
    METHOD_NAME,
    METHOD_VERSION,
    PARENT_METHOD_VERSION,
    REPRESENTATION_VERSION,
    PpiAwareHarrisZPlusConfig,
    build_physical_scale_contract,
)


class HarrisZPlusPpiAwareGeometricAdapter(HarrisZPlusGeometricAdapter):
    """v4 adapter; matching, geometry, score, and threshold semantics stay frozen."""

    def __init__(
        self,
        config: PpiAwareHarrisZPlusConfig | None = None,
    ) -> None:
        active = config or PpiAwareHarrisZPlusConfig()
        super().__init__(active.reference)
        self.config = active
        cv2.setNumThreads(int(self.config.opencv_threads))
        cv2.setUseOptimized(bool(self.config.opencv_optimized))
        _enforce_determinism(self.config)

    def metadata(self) -> MethodMetadata:
        parent = super().metadata()
        contract = build_physical_scale_contract(self.config)
        provenance = {
            **parent.implementation_provenance,
            "parent_method_version": PARENT_METHOD_VERSION,
            "v4_isolation": {
                "only_algorithmic_change": (
                    "all audited spatial HarrisZ+ parameters are interpreted "
                    "at manifest_ppi / 1000 reference scale"
                ),
                "descriptor_unchanged": True,
                "matcher_unchanged": True,
                "ransac_unchanged_and_already_ppi_normalized": True,
                "raw_score_unchanged": True,
                "keypoint_cap": int(self.config.max_keypoints),
                "decision_threshold_outside_adapter": 4,
                "native_grayscale_no_hidden_resize": True,
                "physical_scale_contract_passed": bool(contract["passed"]),
            },
            "ppi_aware_spatial_contract": contract,
            "timing_policy": {
                "cuda_peak_measurement": (
                    "synchronize, empty_cache, synchronize, reset peaks, then "
                    "report allocated and reserved separately"
                ),
                "peak_values_are_never_summed": True,
                "each_peak_validated_against_physical_device_memory": True,
            },
        }
        return replace(
            parent,
            method=METHOD_NAME,
            method_version=METHOD_VERSION,
            implementation_provenance=provenance,
            config={
                **self.config.as_dict(),
                "method": METHOD_NAME,
                "method_version": METHOD_VERSION,
                "parent_method_version": PARENT_METHOD_VERSION,
                "representation_version": REPRESENTATION_VERSION,
                "score_direction": parent.score_direction,
                "score_field": "geometric_inlier_count",
                "thresholding": "none_in_adapter_compare",
                "decision_threshold": None,
                "publication_decision_threshold": 4,
                "representation_cache": False,
                "cross_pair_cache": False,
            },
        )

    def prepare(
        self,
        image_path: Path,
        image_metadata: Mapping[str, Any],
    ) -> PrepareOutcome:
        representation, diagnostics, elapsed_ms = extract_representation_v4(
            image_path,
            image_metadata,
            self.config,
        )
        digest = str(diagnostics["representation_sha256"])
        spatial_scale = float(diagnostics["spatial_scale"])
        return PrepareOutcome(
            representation=PreparedRepresentation(
                method=METHOD_NAME,
                method_version=METHOD_VERSION,
                representation_format=(
                    "harriszplus-ppi-aware-keypoints-rootsift-descriptors"
                ),
                representation_version=REPRESENTATION_VERSION,
                payload=representation,
                metadata={
                    "width": int(representation.width),
                    "height": int(representation.height),
                    "ppi": float(representation.ppi),
                    "manifest_ppi": float(representation.ppi),
                    "reference_ppi": float(self.config.reference_ppi),
                    "spatial_scale": spatial_scale,
                    "keypoint_count": int(representation.keypoint_count),
                    "descriptor_count": int(
                        representation.descriptors.shape[0]
                    ),
                    "detector_backend": str(self.config.backend),
                    "descriptor_mode": "rootsift",
                    "representation_sha256": digest,
                    "prepare_total_ms": float(elapsed_ms),
                },
            ),
            method_internal_ms=float(elapsed_ms),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _payload(
        representation: PreparedRepresentation,
    ) -> SiftRepresentation:
        if (
            representation.method != METHOD_NAME
            or representation.method_version != METHOD_VERSION
        ):
            raise MethodExecutionError(
                "representation_identity_mismatch",
                "v4 comparison received another method or version.",
            )
        if representation.representation_version != REPRESENTATION_VERSION:
            raise MethodExecutionError(
                "representation_version_mismatch",
                "v4 comparison received an incompatible representation.",
            )
        if not isinstance(representation.payload, SiftRepresentation):
            raise MethodExecutionError(
                "representation_format_mismatch",
                "v4 comparison requires a SiftRepresentation payload.",
            )
        payload = representation.payload
        if payload.descriptors.ndim != 2 or payload.descriptors.shape[0] < 2:
            raise MethodExecutionError(
                "invalid_prepared_representation",
                "v4 representations require at least two RootSIFT descriptors.",
            )
        return payload


HarrisZPlusPpiAwareAdapter = HarrisZPlusPpiAwareGeometricAdapter


__all__ = [
    "HarrisZPlusPpiAwareAdapter",
    "HarrisZPlusPpiAwareGeometricAdapter",
]
