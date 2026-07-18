"""Pairwise benchmark adapter for HarrisZ+ -> RootSIFT -> affine geometry."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

import cv2

from fingerprint_benchmark.contract import (
    HIGHER_IS_MORE_SIMILAR,
    CompareOutcome,
    MethodExecutionError,
    MethodMetadata,
    PrepareOutcome,
    PreparedRepresentation,
)
from fingerprint_benchmark.sift.config import SiftGeometricConfig
from fingerprint_benchmark.sift.extractor import SiftRepresentation
from fingerprint_benchmark.sift.geometry import verify_geometry
from fingerprint_benchmark.sift.matching import match_descriptors

from .config import METHOD_NAME, METHOD_VERSION, HarrisZPlusConfig
from .extractor import _enforce_determinism, extract_representation
from .provenance import (
    clean_room_provenance,
    determinism_metadata,
    implementation_source_hashes,
    representation_sha256,
    runtime_metadata,
)


REPRESENTATION_VERSION = "harriszplus-rootsift-representation-v1"


class HarrisZPlusGeometricAdapter:
    """Detector-isolation adapter with no decision threshold and no cross-pair cache."""

    def __init__(self, config: HarrisZPlusConfig | None = None) -> None:
        self.config = config or HarrisZPlusConfig()
        cv2.setNumThreads(int(self.config.opencv_threads))
        cv2.setUseOptimized(bool(self.config.opencv_optimized))
        _enforce_determinism(self.config)
        self._sift_geometry_config = _sift_geometry_config(self.config)

    def metadata(self) -> MethodMetadata:
        clean_room = clean_room_provenance()
        source_hashes = implementation_source_hashes(strict=True)
        config_payload = _config_dict(self.config)
        implementation = {
            **clean_room,
            "engine": "clean-room HarrisZ+ CPU/CUDA detector with OpenCV supplied-keypoint SIFT",
            "detector_backend": str(self.config.backend),
            "detector_dtype": str(self.config.detector_dtype),
            "detector_only_ablation": (
                "HarrisZ+ replaces DoG/SIFT keypoint detection; descriptor, matcher, geometry, "
                "and raw-score semantics reuse the existing SIFT implementation"
            ),
            "image_policy": {
                "load": "cv2.imread(path, cv2.IMREAD_GRAYSCALE)",
                "resolution": "native",
                "ppi_source": "manifest_only",
                "enhancement": "none",
                "segmentation": "none",
                "downsampling": "none",
                "doubled_scales": [0, 1],
                "doubling": "one exact 2x cv2.resize call with INTER_LANCZOS4",
                "oracle_divergence": (
                    "OpenCV INTER_LANCZOS4 is the frozen adapter choice; the official Matlab "
                    "oracle uses Lanczos-3 support"
                ),
            },
            "keypoint_adapter": {
                "opencv_size_mapping": "size = 2 * output_integration_sigma",
                "effective_detector_gaussian_support": "recorded separately from descriptor size",
                "opencv_octave": (
                    "0 for every native-coordinate supplied keypoint; HarrisZ scale must not "
                    "alter OpenCV descriptor support"
                ),
                "opencv_class_id": "HarrisZ+ scale_index",
                "detector_source_index": "preserved separately in representation metadata",
                "orientation": {
                    "method": "SIFT-style signed dominant gradient",
                    "histogram_bins": int(self.config.orientation_bins),
                    "gaussian_sigma_factor": float(
                        self.config.orientation_gaussian_sigma_factor
                    ),
                    "radius_factor_of_weighting_sigma": float(
                        self.config.orientation_radius_factor
                    ),
                    "histogram_interpolation": "linear circular",
                    "histogram_smoothing": "repeated circular three-bin mean",
                    "histogram_smoothing_passes": int(
                        self.config.orientation_histogram_smoothing_passes
                    ),
                    "peak_interpolation": "three-sample parabola clipped to half a bin",
                    "orientations_per_keypoint": 1,
                    "tie_policy": "lowest interpolated angle",
                    "empty_histogram_angle_degrees": 0.0,
                    "ambiguity_180_degrees": "full signed gradient; no random flip",
                },
            },
            "descriptor": {
                "local_descriptor": "cv2.SIFT.compute at supplied keypoints",
                "postprocessing": "fingerprint_benchmark.sift.descriptors.rootsift unchanged",
                "non_finite_policy": "drop complete descriptor rows before RootSIFT",
                "minimum_finite_descriptors": int(self.config.minimum_descriptors),
            },
            "matching": {
                "implementation": "fingerprint_benchmark.sift.matching.match_descriptors unchanged",
                "matcher": "BF-L2 KNN k=2",
                "lowe_ratio": float(self.config.lowe_ratio),
                "direction": str(self.config.matching_mode),
            },
            "geometry": {
                "implementation": "fingerprint_benchmark.sift.geometry.verify_geometry unchanged",
                "model": str(self.config.geometry_model),
                "estimator": "OpenCV RANSAC",
                "reference_ppi": float(self.config.reference_ppi),
                "threshold_reference_pixels": float(
                    self.config.ransac_threshold_at_reference_ppi
                ),
                "ppi_normalization": bool(self.config.normalize_coordinates_by_ppi),
                "confidence": float(self.config.ransac_confidence),
                "maximum_iterations": int(self.config.ransac_max_iterations),
                "refinement_iterations": int(self.config.ransac_refine_iterations),
                "rng_seed": int(self.config.rng_seed),
            },
            "score": {
                "raw_score": "integer geometric_inlier_count",
                "direction": HIGHER_IS_MORE_SIMILAR,
                "decision_threshold_in_adapter": None,
            },
            "timing_policy": {
                "cuda_segments": "torch.cuda.synchronize before and after wall timing",
                "kernel_timing": (
                    "detector_gpu_kernel_ms is the sum of CUDA-event elapsed times for base "
                    "gradients, all five dense scales, and GPU eigen/subpixel refinement; "
                    "events are read only after explicit synchronization"
                ),
                "peak_vram": "torch.cuda max_memory_allocated/max_memory_reserved in bytes",
                "pair": "cold pair: prepare A + prepare B + compare; no cross-pair cache",
                "comparability_note": (
                    "HarrisZ+ detector uses GPU when configured; previous SourceAFIS/SIFT pilots "
                    "may use different backends, so timings are not a direct CPU-efficiency ranking"
                ),
            },
            "determinism": determinism_metadata(self.config),
            "implementation_source_sha256": source_hashes,
        }
        return MethodMetadata(
            method=METHOD_NAME,
            method_version=METHOD_VERSION,
            score_direction=HIGHER_IS_MORE_SIMILAR,
            score_semantics=(
                "Integer number of mutual-RootSIFT correspondences verified as inliers by the "
                "existing PPI-normalized partial-affine RANSAC. compare() performs no acceptance "
                "thresholding."
            ),
            implementation_provenance=implementation,
            config={
                **config_payload,
                "method": METHOD_NAME,
                "method_version": METHOD_VERSION,
                "representation_version": REPRESENTATION_VERSION,
                "score_direction": HIGHER_IS_MORE_SIMILAR,
                "score_field": "geometric_inlier_count",
                "thresholding": "none_in_adapter_compare",
                "decision_threshold": None,
                "representation_cache": False,
                "cross_pair_cache": False,
            },
            runtime=runtime_metadata(self.config),
        )

    def prepare(
        self,
        image_path: Path,
        image_metadata: Mapping[str, Any],
    ) -> PrepareOutcome:
        representation, diagnostics, elapsed_ms = extract_representation(
            image_path,
            image_metadata,
            self.config,
        )
        digest = str(diagnostics["representation_sha256"])
        return PrepareOutcome(
            representation=PreparedRepresentation(
                method=METHOD_NAME,
                method_version=METHOD_VERSION,
                representation_format="harriszplus-keypoints-rootsift-descriptors",
                representation_version=REPRESENTATION_VERSION,
                payload=representation,
                metadata={
                    "width": int(representation.width),
                    "height": int(representation.height),
                    "ppi": float(representation.ppi),
                    "keypoint_count": int(representation.keypoint_count),
                    "descriptor_count": int(representation.descriptors.shape[0]),
                    "detector_backend": str(self.config.backend),
                    "descriptor_mode": "rootsift",
                    "representation_sha256": digest,
                    "prepare_total_ms": float(elapsed_ms),
                },
            ),
            method_internal_ms=float(elapsed_ms),
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
                matching_mode=str(self.config.matching_mode),
            )
        except (ValueError, cv2.error) as exc:
            raise MethodExecutionError(
                "descriptor_matching_failure",
                str(exc),
                method_internal_ms=_elapsed(started),
                diagnostics={
                    "matching_mode": str(self.config.matching_mode),
                    "lowe_ratio": float(self.config.lowe_ratio),
                    "matcher_cpu_ms": _elapsed(match_started),
                },
            ) from exc
        matcher_cpu_ms = _elapsed(match_started)

        ransac_started = perf_counter_ns()
        try:
            geometry = verify_geometry(
                payload_a,
                payload_b,
                matches.submitted,
                self._sift_geometry_config,
            )
        except (ValueError, IndexError, cv2.error) as exc:
            raise MethodExecutionError(
                "geometric_verification_failure",
                str(exc),
                method_internal_ms=_elapsed(started),
                diagnostics={
                    **matches.diagnostics,
                    "matcher_cpu_ms": matcher_cpu_ms,
                    "ransac_cpu_ms": _elapsed(ransac_started),
                },
            ) from exc
        ransac_cpu_ms = _elapsed(ransac_started)
        score = int(geometry.inlier_count)
        compare_total_ms = _elapsed(started)
        prepare_a_ms = _prepare_elapsed(representation_a, payload_a)
        prepare_b_ms = _prepare_elapsed(representation_b, payload_b)
        end_to_end_wall_ms = prepare_a_ms + prepare_b_ms + compare_total_ms
        diagnostics: dict[str, Any] = {
            "keypoint_count_a": int(payload_a.keypoint_count),
            "keypoint_count_b": int(payload_b.keypoint_count),
            "descriptor_count_a": int(payload_a.descriptors.shape[0]),
            "descriptor_count_b": int(payload_b.descriptors.shape[0]),
            "representation_sha256_a": _prepared_representation_hash(
                representation_a, payload_a
            ),
            "representation_sha256_b": _prepared_representation_hash(
                representation_b, payload_b
            ),
            "matcher": "existing_sift_mutual_bf_l2_knn2_lowe",
            "matching_mode": str(self.config.matching_mode),
            "lowe_ratio": float(self.config.lowe_ratio),
            **matches.diagnostics,
            "tentative_match_count": int(len(matches.forward_ratio)),
            "mutual_match_count": int(len(matches.submitted)),
            "ransac_input_count": int(len(matches.submitted)),
            **geometry.diagnostics,
            "model_valid": bool(geometry.success),
            "failure_reason": geometry.failure_reason,
            "score_field": "geometric_inlier_count",
            "score_type": "integer",
            "score_direction": HIGHER_IS_MORE_SIMILAR,
            "decision_threshold_applied": False,
            "matcher_cpu_ms": matcher_cpu_ms,
            "ransac_cpu_ms": ransac_cpu_ms,
            "compare_total_ms": compare_total_ms,
            "prepare_a_total_ms": prepare_a_ms,
            "prepare_b_total_ms": prepare_b_ms,
            "end_to_end_wall_ms": end_to_end_wall_ms,
        }
        return CompareOutcome(
            raw_score=score,
            method_internal_ms=compare_total_ms,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _payload(representation: PreparedRepresentation) -> SiftRepresentation:
        if representation.method != METHOD_NAME or representation.method_version != METHOD_VERSION:
            raise MethodExecutionError(
                "representation_identity_mismatch",
                "HarrisZ+ comparison received a representation from another method or version.",
            )
        if representation.representation_version != REPRESENTATION_VERSION:
            raise MethodExecutionError(
                "representation_version_mismatch",
                "HarrisZ+ comparison received an incompatible representation version.",
            )
        if not isinstance(representation.payload, SiftRepresentation):
            raise MethodExecutionError(
                "representation_format_mismatch",
                "HarrisZ+ comparison received an invalid SiftRepresentation payload.",
            )
        payload = representation.payload
        if payload.descriptors.ndim != 2 or payload.descriptors.shape[0] < 2:
            raise MethodExecutionError(
                "invalid_prepared_representation",
                "HarrisZ+ representations must contain at least two RootSIFT descriptors.",
            )
        return payload


# Shorter public spelling retained for CLI/factory callers.
HarrisZPlusAdapter = HarrisZPlusGeometricAdapter


def _sift_geometry_config(config: HarrisZPlusConfig) -> SiftGeometricConfig:
    """Map frozen HarrisZ+ fields into the exact existing SIFT geometry contract."""

    return SiftGeometricConfig(
        image_policy="native",
        mask_mode="none",
        descriptor_mode="rootsift",
        matching_mode=str(config.matching_mode),
        geometry_model=str(config.geometry_model),
        score_mode=str(config.score_mode),
        nfeatures=int(config.max_keypoints),
        n_octave_layers=int(config.sift_n_octave_layers),
        contrast_threshold=float(config.sift_contrast_threshold),
        edge_threshold=float(config.sift_edge_threshold),
        sigma=float(config.sift_sigma),
        lowe_ratio=float(config.lowe_ratio),
        minimum_descriptors=int(config.minimum_descriptors),
        minimum_geometry_matches=int(config.minimum_geometry_matches),
        ransac_threshold_at_reference_ppi=float(
            config.ransac_threshold_at_reference_ppi
        ),
        ransac_confidence=float(config.ransac_confidence),
        ransac_max_iterations=int(config.ransac_max_iterations),
        ransac_refine_iterations=int(config.ransac_refine_iterations),
        normalize_coordinates_by_ppi=bool(config.normalize_coordinates_by_ppi),
        reference_ppi=float(config.reference_ppi),
        rng_seed=int(config.rng_seed),
        opencv_threads=int(config.opencv_threads),
        opencv_optimized=bool(config.opencv_optimized),
    )


def _config_dict(config: HarrisZPlusConfig) -> dict[str, Any]:
    if hasattr(config, "as_dict"):
        payload = config.as_dict()
        if isinstance(payload, dict):
            return payload
    from dataclasses import asdict, is_dataclass

    if is_dataclass(config):
        return asdict(config)
    raise TypeError("HarrisZPlusConfig must expose as_dict() or be a dataclass.")


def _prepare_elapsed(
    prepared: PreparedRepresentation,
    payload: SiftRepresentation,
) -> float:
    value = prepared.metadata.get("prepare_total_ms", payload.metadata.get("prepare_total_ms", 0.0))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if result >= 0.0 else 0.0


def _prepared_representation_hash(
    prepared: PreparedRepresentation,
    payload: SiftRepresentation,
) -> str:
    recorded = prepared.metadata.get("representation_sha256")
    return str(recorded) if recorded is not None else representation_sha256(payload)


def _elapsed(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0
