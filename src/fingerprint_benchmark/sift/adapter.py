"""Method-neutral adapter for the single public SIFT geometric method."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
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

from .config import METHOD_NAME, METHOD_VERSION, REPRESENTATION_VERSION, SiftGeometricConfig
from .extractor import SiftRepresentation, extract_representation
from .geometry import verify_geometry
from .matching import match_descriptors
from .scoring import raw_score, score_components


class SiftGeometricAdapter:
    def __init__(self, config: SiftGeometricConfig | None = None) -> None:
        self.config = config or SiftGeometricConfig()
        cv2.setNumThreads(int(self.config.opencv_threads))
        cv2.setUseOptimized(bool(self.config.opencv_optimized))

    def metadata(self) -> MethodMetadata:
        package = _opencv_distribution()
        sift_parameters = {
            "nfeatures": int(self.config.nfeatures),
            "nOctaveLayers": int(self.config.n_octave_layers),
            "contrastThreshold": float(self.config.contrast_threshold),
            "edgeThreshold": float(self.config.edge_threshold),
            "sigma": float(self.config.sigma),
        }
        return MethodMetadata(
            method=METHOD_NAME,
            method_version=METHOD_VERSION,
            score_direction=HIGHER_IS_MORE_SIMILAR,
            score_semantics=(
                "OpenCV SIFT L2 Lowe matches verified by PPI-normalized affine RANSAC; raw score is "
                f"{self.config.score_mode} with no acceptance thresholding in compare()."
            ),
            implementation_provenance={
                "engine": "OpenCV SIFT",
                "opencv_version": cv2.__version__,
                "opencv_distribution": package,
                "numpy_version": np.__version__,
                "sift_constructor_parameters": sift_parameters,
                "random_seed_policy": "cv2.setRNGSeed(config.rng_seed) immediately before every geometry fit",
                "implementation_source_family": "repository-native modular adapter around cv2.SIFT_create",
            },
            config={
                **self.config.as_dict(),
                "method": METHOD_NAME,
                "method_version": METHOD_VERSION,
                "thresholding": "none_in_compare",
                "representation_cache": False,
                "sift_constructor_parameters": sift_parameters,
            },
            runtime={
                "python_version": sys.version,
                "opencv_version": cv2.__version__,
                "opencv_distribution": package,
                "opencv_build_information": cv2.getBuildInformation(),
                "numpy_version": np.__version__,
                "operating_system": platform.platform(),
                "cpu_architecture": platform.machine(),
                "opencv_thread_count": cv2.getNumThreads(),
                "opencv_optimized": bool(cv2.useOptimized()),
                "timing_policy": "in-process wall time; cold-pair runner performs two prepares and one compare",
            },
        )

    def prepare(self, image_path: Path, image_metadata: Mapping[str, Any]) -> PrepareOutcome:
        representation, diagnostics, elapsed_ms = extract_representation(
            image_path, image_metadata, self.config
        )
        return PrepareOutcome(
            representation=PreparedRepresentation(
                method=METHOD_NAME,
                method_version=METHOD_VERSION,
                representation_format="opencv-sift-keypoints-descriptors",
                representation_version=REPRESENTATION_VERSION,
                payload=representation,
                metadata={
                    "width": representation.width,
                    "height": representation.height,
                    "ppi": representation.ppi,
                    "keypoint_count": representation.keypoint_count,
                    "descriptor_mode": self.config.descriptor_mode,
                },
            ),
            method_internal_ms=elapsed_ms,
            diagnostics=diagnostics,
        )

    def compare(
        self,
        representation_a: PreparedRepresentation,
        representation_b: PreparedRepresentation,
    ) -> CompareOutcome:
        started = perf_counter_ns()
        payload_a = self._payload(representation_a)
        payload_b = self._payload(representation_b)
        match_started = perf_counter_ns()
        try:
            matches = match_descriptors(
                payload_a.descriptors,
                payload_b.descriptors,
                lowe_ratio=float(self.config.lowe_ratio),
                matching_mode=self.config.matching_mode,
            )
        except (ValueError, cv2.error) as exc:
            raise MethodExecutionError(
                "descriptor_matching_failure",
                str(exc),
                method_internal_ms=_elapsed(started),
                diagnostics={"matching_mode": self.config.matching_mode},
            ) from exc
        matching_ms = _elapsed(match_started)
        geometry_started = perf_counter_ns()
        geometry = verify_geometry(payload_a, payload_b, matches.submitted, self.config)
        geometric_verification_ms = _elapsed(geometry_started)
        scoring_started = perf_counter_ns()
        components = score_components(
            inliers=geometry.inlier_count,
            matches=len(matches.submitted),
            keypoints_a=payload_a.keypoint_count,
            keypoints_b=payload_b.keypoint_count,
        )
        try:
            score = raw_score(self.config.score_mode, components)
        except ValueError as exc:
            raise MethodExecutionError(
                "raw_score_failure",
                str(exc),
                method_internal_ms=_elapsed(started),
                diagnostics={"score_components": components},
            ) from exc
        scoring_ms = _elapsed(scoring_started)
        diagnostics = {
            "keypoint_count_a": payload_a.keypoint_count,
            "keypoint_count_b": payload_b.keypoint_count,
            "descriptor_count_a": int(payload_a.descriptors.shape[0]),
            "descriptor_count_b": int(payload_b.descriptors.shape[0]),
            "lowe_ratio": float(self.config.lowe_ratio),
            **matches.diagnostics,
            **geometry.diagnostics,
            "score_mode": self.config.score_mode,
            "score_components": components,
            "zero_score_semantics": (
                "zero geometric inliers or insufficient/failed affine verification; preparation failures have no score"
            ),
            "matching_ms": matching_ms,
            "geometric_verification_ms": geometric_verification_ms,
            "scoring_ms": scoring_ms,
        }
        return CompareOutcome(
            raw_score=score,
            method_internal_ms=_elapsed(started),
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _payload(representation: PreparedRepresentation) -> SiftRepresentation:
        if representation.method != METHOD_NAME or representation.method_version != METHOD_VERSION:
            raise MethodExecutionError(
                "representation_identity_mismatch",
                "SIFT comparison received a representation from another method or version.",
            )
        if not isinstance(representation.payload, SiftRepresentation):
            raise MethodExecutionError(
                "representation_format_mismatch",
                "SIFT comparison received an invalid representation payload.",
            )
        return representation.payload


def _opencv_distribution() -> dict[str, str | None]:
    installed: list[tuple[str, str]] = []
    for name in ("opencv-python", "opencv-python-headless"):
        try:
            installed.append((name, importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) != 1:
        return {"name": None, "version": None, "status": f"expected exactly one distribution, found {installed}"}
    return {"name": installed[0][0], "version": installed[0][1], "status": "ok"}


def _elapsed(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0
