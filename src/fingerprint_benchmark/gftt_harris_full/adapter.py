"""Independent full-system identity over the parity-protected Harris pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.contract import (
    HIGHER_IS_MORE_SIMILAR,
    CompareOutcome,
    MethodExecutionError,
    MethodMetadata,
    PrepareOutcome,
    PreparedRepresentation,
)
from fingerprint_benchmark.detectors.opencv_gftt_harris import (
    OpenCVGFTTHarrisRootSIFTGeometricAdapter,
)

from .config import (
    CONFIG_SCHEMA_VERSION,
    GFTTHarrisRootSIFTGeometricConfig,
    frozen_config,
)
from .restored_equivalence import (
    PARENT_METHOD_NAME,
    PARENT_METHOD_VERSION,
    PARENT_PROTOCOL_SHA256,
    PARENT_RUN_CONFIG_HASH,
    assert_v1_equivalence,
)


METHOD_NAME = "gftt_harris_rootsift_geometric"
METHOD_VERSION = "gftt-harris-rootsift-geometric-v1"
REPRESENTATION_VERSION = "gftt-harris-rootsift-geometric-representation-v1"


class GFTTHarrisRootSIFTGeometricAdapter:
    """Complete image-to-score method with a method-owned frozen config.

    Composition is intentional: protected detector-only code remains the
    parity oracle, while this class owns full-system identity, configuration,
    provenance and representation authorization.
    """

    method_name = METHOD_NAME
    method_version = METHOD_VERSION

    def __init__(
        self,
        config: GFTTHarrisRootSIFTGeometricConfig | None = None,
    ) -> None:
        self.config = config if config is not None else frozen_config()
        assert_v1_equivalence(self.config)
        self._pipeline = OpenCVGFTTHarrisRootSIFTGeometricAdapter(
            detector_config=self.config.to_detector_config(),
            protocol_config=self.config.to_pipeline_config(),
        )

    def implementation_source_paths(self) -> tuple[Path, ...]:
        shared = set(self._pipeline.implementation_source_paths())
        package = Path(__file__).resolve().parent
        shared.update(
            {
                package / "__init__.py",
                package / "adapter.py",
                package / "config.py",
                package / "restored_equivalence.py",
            }
        )
        return tuple(sorted((path.resolve() for path in shared), key=lambda path: path.as_posix()))

    def metadata(self) -> MethodMetadata:
        algorithm_config = self.config.algorithm_config()
        algorithm_config_hash = self.config.config_hash
        return MethodMetadata(
            method=METHOD_NAME,
            method_version=METHOD_VERSION,
            score_direction=HIGHER_IS_MORE_SIMILAR,
            score_semantics=(
                "Geometric inlier count from single-scale OpenCV GFTT-Harris locations, "
                "fixed dominant-gradient orientation, supplied-keypoint RootSIFT, mutual "
                "Lowe matching and PPI-normalized partial-affine RANSAC; no decision threshold."
            ),
            implementation_provenance={
                "method_family": "handcrafted local-feature and geometric matching",
                "parent_pipeline": "detector_only_v1 Harris branch",
                "parent_method": PARENT_METHOD_NAME,
                "parent_method_version": PARENT_METHOD_VERSION,
                "parent_run_config_hash": PARENT_RUN_CONFIG_HASH,
                "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
                "components": {
                    "gftt_harris": "fingerprint_benchmark.detectors.opencv_gftt_harris",
                    "orientation": "fingerprint_benchmark.local_features.orientation",
                    "descriptor_extraction": (
                        "fingerprint_benchmark.local_features.descriptors.sift_descriptor"
                    ),
                    "rootsift": "fingerprint_benchmark.local_features.descriptors.rootsift",
                    "matching": "fingerprint_benchmark.local_features.matching",
                    "geometry": "fingerprint_benchmark.local_features.geometry",
                    "scoring": "fingerprint_benchmark.local_features.scoring",
                    "full_adapter": "fingerprint_benchmark.gftt_harris_full.adapter",
                    "full_config": "fingerprint_benchmark.gftt_harris_full.config",
                },
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
            },
            config={
                "config_schema_version": CONFIG_SCHEMA_VERSION,
                "algorithm_config": algorithm_config,
                "algorithm_config_hash": algorithm_config_hash,
                "config_hash": algorithm_config_hash,
                "method": METHOD_NAME,
                "method_version": METHOD_VERSION,
                "score_mode": "geometric_inlier_count",
                "decision_threshold": None,
                "thresholding": "none_in_adapter",
                "representation_cache": False,
            },
            runtime={
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "opencv_thread_count": cv2.getNumThreads(),
                "opencv_optimized": bool(cv2.useOptimized()),
            },
        )

    def prepare(
        self,
        image_path: Path,
        image_metadata: Mapping[str, Any],
    ) -> PrepareOutcome:
        outcome = self._pipeline.prepare(image_path, image_metadata)
        source = outcome.representation
        return PrepareOutcome(
            representation=PreparedRepresentation(
                method=METHOD_NAME,
                method_version=METHOD_VERSION,
                representation_format=source.representation_format,
                representation_version=REPRESENTATION_VERSION,
                payload=source.payload,
                metadata=dict(source.metadata),
            ),
            method_internal_ms=outcome.method_internal_ms,
            diagnostics=dict(outcome.diagnostics),
        )

    def compare(
        self,
        representation_a: PreparedRepresentation,
        representation_b: PreparedRepresentation,
    ) -> CompareOutcome:
        return self._pipeline.compare(
            self._to_pipeline_representation(representation_a),
            self._to_pipeline_representation(representation_b),
        )

    def close(self) -> None:
        self._pipeline.close()

    def _to_pipeline_representation(
        self,
        prepared: PreparedRepresentation,
    ) -> PreparedRepresentation:
        if prepared.method != METHOD_NAME or prepared.method_version != METHOD_VERSION:
            raise MethodExecutionError(
                "representation_identity_mismatch",
                f"{METHOD_VERSION} received a representation from another method or version.",
            )
        if prepared.representation_version != REPRESENTATION_VERSION:
            raise MethodExecutionError(
                "representation_version_mismatch",
                f"{METHOD_VERSION} received an incompatible representation version.",
            )
        return PreparedRepresentation(
            method=self._pipeline.method_name,
            method_version=self._pipeline.method_version,
            representation_format=prepared.representation_format,
            representation_version="detector-only-local-features-v1",
            payload=prepared.payload,
            metadata=dict(prepared.metadata),
        )


__all__ = [
    "METHOD_NAME",
    "METHOD_VERSION",
    "REPRESENTATION_VERSION",
    "GFTTHarrisRootSIFTGeometricAdapter",
]
