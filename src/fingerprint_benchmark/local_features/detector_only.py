"""Shared ``detector_only_v1`` adaptation and pairwise benchmark adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
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
from fingerprint_benchmark.detectors.types import Detector, DetectorResult

from . import scoring as local_scoring
from .descriptors import compute_sift_descriptors
from .geometry import verify_geometry
from .matching import match_descriptors
from .orientation import (
    ORIENTATION_POLICY,
    assign_orientations,
    normalize_orientation_policy,
)
from .support import assign_support_sizes
from .types import CanonicalLocalFeature, LocalFeatureRepresentation


PROTOCOL_NAME = "detector_only_v1"
PROTOCOL_VERSION = "detector-only-v1"
REPRESENTATION_VERSION = "detector-only-local-features-v1"
SCORE_MODE = "geometric_inlier_count"


@dataclass(frozen=True, slots=True)
class DetectorOnlyProtocolConfig:
    """All fixed choices downstream of the detector under comparison."""

    reference_ppi: float = 1000.0
    support_size_reference_px: float = 16.0
    maximum_keypoints: int = 3000
    descriptor: str = "rootsift"
    orientation_policy: str = ORIENTATION_POLICY
    matching_mode: str = "mutual"
    lowe_ratio: float = 0.75
    geometry_model: str = "affine_partial_2d"
    ransac_threshold_reference_px: float = 3.0
    minimum_descriptors: int = 2
    minimum_geometry_matches: int = 3
    ransac_confidence: float = 0.99
    ransac_max_iterations: int = 2000
    ransac_refine_iterations: int = 10
    rng_seed: int = 0
    opencv_threads: int = 16
    opencv_optimized: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.reference_ppi,
            self.support_size_reference_px,
            self.ransac_threshold_reference_px,
        )
        if not all(math.isfinite(float(value)) and value > 0.0 for value in positive):
            raise ValueError("Reference PPI, support size, and RANSAC threshold must be positive.")
        if self.maximum_keypoints <= 0:
            raise ValueError("maximum_keypoints must be positive.")
        if self.descriptor not in ("rootsift", "sift", "standard"):
            raise ValueError("descriptor must be rootsift, sift, or standard.")
        object.__setattr__(
            self,
            "orientation_policy",
            normalize_orientation_policy(self.orientation_policy),
        )
        if self.matching_mode not in ("one_way", "bidirectional_union", "mutual"):
            raise ValueError("Unsupported matching mode.")
        if self.geometry_model not in ("affine_full_2d", "affine_partial_2d"):
            raise ValueError("Unsupported geometry model.")
        if not 0.0 < self.lowe_ratio < 1.0:
            raise ValueError("lowe_ratio must be in (0, 1).")
        if self.minimum_descriptors < 2 or self.minimum_geometry_matches < 3:
            raise ValueError("At least two descriptors and three geometry matches are required.")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("ransac_confidence must be in (0, 1).")
        if self.ransac_max_iterations <= 0 or self.ransac_refine_iterations < 0:
            raise ValueError("RANSAC iteration counts are invalid.")
        if self.opencv_threads <= 0:
            raise ValueError("opencv_threads must be positive.")

    @property
    def ransac_threshold_at_reference_ppi(self) -> float:
        """Compatibility field consumed by the protected geometry component."""

        return float(self.ransac_threshold_reference_px)

    @property
    def normalize_coordinates_by_ppi(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_features(
    image: np.ndarray,
    image_metadata: Mapping[str, object],
    detector_result: DetectorResult,
    config: DetectorOnlyProtocolConfig,
) -> tuple[tuple[CanonicalLocalFeature, ...], dict[str, object]]:
    """Use detector locations only, then assign common support and orientation."""

    source = np.asarray(image)
    if source.ndim != 2 or source.dtype != np.uint8 or source.size == 0:
        raise ValueError("detector_only_v1 requires a non-empty uint8 grayscale image.")
    ppi = _positive_ppi(image_metadata)
    selected = detector_result.points[: int(config.maximum_keypoints)]
    points = np.asarray([(item.x, item.y) for item in selected], dtype=np.float32).reshape(-1, 2)
    if points.size and not np.isfinite(points).all():
        raise ValueError("Detector locations contain non-finite values.")
    if points.size and (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] >= source.shape[1])
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] >= source.shape[0])
    ):
        raise ValueError("Detector locations must lie inside the source image.")
    sizes = assign_support_sizes(
        len(selected),
        ppi,
        reference_ppi=float(config.reference_ppi),
        support_size_reference_px=float(config.support_size_reference_px),
    )
    angles, orientation_diagnostics = assign_orientations(
        source,
        points,
        sizes,
        policy=config.orientation_policy,
    )
    canonical = tuple(
        CanonicalLocalFeature(
            x=float(point[0]),
            y=float(point[1]),
            support_size=float(size),
            angle=float(angle),
        )
        for point, size, angle in zip(points, sizes, angles, strict=True)
    )
    return canonical, {
        "protocol": PROTOCOL_NAME,
        "detector_point_count": int(detector_result.count),
        "canonical_point_count": len(canonical),
        "maximum_keypoints": int(config.maximum_keypoints),
        "detector_fields_used": ["x", "y"],
        "detector_response_used": False,
        "detector_scale_used": False,
        "detector_angle_used": False,
        "support_policy": "fixed_physical_diameter_scaled_by_manifest_ppi",
        "support_size_reference_px": float(config.support_size_reference_px),
        "reference_ppi": float(config.reference_ppi),
        "native_support_size_px": float(sizes[0]) if sizes.size else None,
        **orientation_diagnostics,
    }


def build_representation(
    image: np.ndarray,
    image_metadata: Mapping[str, object],
    detector_result: DetectorResult,
    config: DetectorOnlyProtocolConfig | None = None,
) -> tuple[LocalFeatureRepresentation, dict[str, object], float]:
    """Build the complete common representation from one detector result."""

    active = config or DetectorOnlyProtocolConfig()
    started = perf_counter_ns()
    canonical, diagnostics = canonical_features(image, image_metadata, detector_result, active)
    supplied = [
        cv2.KeyPoint(
            float(item.x),
            float(item.y),
            float(item.support_size),
            float(item.angle),
            0.0,
            0,
            int(index),
        )
        for index, item in enumerate(canonical)
    ]
    descriptor_started = perf_counter_ns()
    descriptor_result = compute_sift_descriptors(
        image,
        supplied,
        descriptor=active.descriptor,
    )
    descriptor_ms = _elapsed(descriptor_started)
    keypoints = descriptor_result.keypoints
    ppi = _positive_ppi(image_metadata)
    representation = LocalFeatureRepresentation(
        points=np.asarray([item.pt for item in keypoints], dtype=np.float32).reshape(-1, 2),
        sizes=np.asarray([item.size for item in keypoints], dtype=np.float32),
        angles=np.asarray([item.angle for item in keypoints], dtype=np.float32),
        responses=np.zeros(len(keypoints), dtype=np.float32),
        octaves=np.zeros(len(keypoints), dtype=np.int32),
        class_ids=np.arange(len(keypoints), dtype=np.int32),
        descriptors=descriptor_result.descriptors,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        ppi=float(ppi),
        metadata={
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "detector_fields_used": ["x", "y"],
            "support_policy": "fixed_physical_diameter_scaled_by_manifest_ppi",
            "orientation_policy": ORIENTATION_POLICY,
            "descriptor": active.descriptor,
        },
    )
    digest = representation_sha256(representation)
    elapsed_ms = _elapsed(started)
    return representation, {
        **diagnostics,
        **descriptor_result.diagnostics,
        "descriptor_ms": descriptor_ms,
        "representation_sha256": digest,
        "representation_keypoint_count": representation.keypoint_count,
        "representation_descriptor_count": int(representation.descriptors.shape[0]),
        "adaptation_total_ms": elapsed_ms,
    }, elapsed_ms


# Explicit protocol spelling used by tests and downstream research code.
adapt_detector_result = build_representation


class DetectorOnlyAdapter:
    """Pairwise adapter in which only the injected detector may vary."""

    def __init__(
        self,
        detector: Detector,
        config: DetectorOnlyProtocolConfig | None = None,
        *,
        method_name: str | None = None,
        method_version: str | None = None,
    ) -> None:
        self.detector = detector
        self.config = config or DetectorOnlyProtocolConfig()
        self.method_name, self.method_version = _method_identity(
            detector,
            self.config.descriptor,
            method_name=method_name,
            method_version=method_version,
        )
        cv2.setNumThreads(int(self.config.opencv_threads))
        cv2.setUseOptimized(bool(self.config.opencv_optimized))

    def implementation_source_paths(self) -> tuple[Path, ...]:
        """Declare every repository source that can affect representation or score."""

        package_directory = Path(__file__).resolve().parent.parent
        detector_source = inspect.getsourcefile(self.detector.__class__)
        if not detector_source:
            raise ValueError(
                f"Cannot locate implementation source for detector {self.detector.__class__!r}."
            )
        relative_sources = (
            "detectors/types.py",
            "local_features/types.py",
            "local_features/support.py",
            "local_features/orientation.py",
            "local_features/detector_only.py",
            "local_features/descriptors/__init__.py",
            "local_features/descriptors/sift_descriptor.py",
            "local_features/descriptors/rootsift.py",
            "local_features/matching.py",
            "local_features/geometry.py",
            "local_features/scoring.py",
        )
        paths = {package_directory / relative for relative in relative_sources}
        paths.add(Path(detector_source).resolve())
        return tuple(
            sorted((path.resolve() for path in paths), key=lambda path: path.as_posix())
        )

    def metadata(self) -> MethodMetadata:
        detector_config = _detector_config(self.detector)
        return MethodMetadata(
            method=self.method_name,
            method_version=self.method_version,
            score_direction=HIGHER_IS_MORE_SIMILAR,
            score_semantics=(
                "Geometric inlier count after detector_only_v1 common support and orientation, "
                f"{self.config.descriptor} descriptor processing, matching, and PPI-normalized "
                "affine verification; no decision threshold."
            ),
            implementation_provenance={
                "protocol": PROTOCOL_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "detector": self.detector.detector_name,
                "detector_version": self.detector.detector_version,
                "detector_config": detector_config,
                "descriptor_engine": "OpenCV SIFT compute at supplied keypoints",
                "descriptor": self.config.descriptor,
                "descriptor_normalization": self.config.descriptor,
                "orientation_policy": self.config.orientation_policy,
                "matching_implementation": "fingerprint_benchmark.local_features.matching",
                "geometry_implementation": "fingerprint_benchmark.local_features.geometry",
                "scoring_implementation": "fingerprint_benchmark.local_features.scoring",
                "score_mode": SCORE_MODE,
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
            },
            config={
                **self.config.as_dict(),
                "method": self.method_name,
                "method_version": self.method_version,
                "detector": self.detector.detector_name,
                "detector_version": self.detector.detector_version,
                "detector_config": detector_config,
                "protocol": PROTOCOL_NAME,
                "score_mode": SCORE_MODE,
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
        started = perf_counter_ns()
        load_started = perf_counter_ns()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image_load_ms = _elapsed(load_started)
        if image is None:
            raise MethodExecutionError(
                "image_read_failure",
                f"OpenCV could not read grayscale image: {image_path}",
                method_internal_ms=_elapsed(started),
                diagnostics={"image_load_ms": image_load_ms},
            )
        try:
            detector_result = self.detector.detect(image, image_metadata, mask=None)
            representation, adaptation, _ = build_representation(
                image,
                image_metadata,
                detector_result,
                self.config,
            )
        except (ValueError, TypeError, cv2.error) as exc:
            raise MethodExecutionError(
                "detector_only_preparation_failure",
                str(exc),
                method_internal_ms=_elapsed(started),
                diagnostics={"image_load_ms": image_load_ms},
            ) from exc
        if representation.descriptors.shape[0] < int(self.config.minimum_descriptors):
            raise MethodExecutionError(
                "too_few_descriptors",
                f"The common pipeline produced {representation.descriptors.shape[0]} descriptors; "
                f"at least {self.config.minimum_descriptors} are required.",
                method_internal_ms=_elapsed(started),
                diagnostics={
                    "image_load_ms": image_load_ms,
                    "detector_diagnostics": dict(detector_result.diagnostics),
                    **adaptation,
                },
            )
        elapsed_ms = _elapsed(started)
        diagnostics = {
            "image_load_ms": image_load_ms,
            "detector_name": detector_result.detector_name,
            "detector_version": detector_result.detector_version,
            "detector_config": dict(detector_result.detector_config),
            "detector_metadata": dict(detector_result.metadata),
            "detector_diagnostics": dict(detector_result.diagnostics),
            "detector_time_ms": float(detector_result.detector_time_ms),
            **adaptation,
            "prepare_total_ms": elapsed_ms,
        }
        return PrepareOutcome(
            representation=PreparedRepresentation(
                method=self.method_name,
                method_version=self.method_version,
                representation_format="detector-only-local-features",
                representation_version=REPRESENTATION_VERSION,
                payload=representation,
                metadata={
                    "width": representation.width,
                    "height": representation.height,
                    "ppi": representation.ppi,
                    "keypoint_count": representation.keypoint_count,
                    "descriptor_count": int(representation.descriptors.shape[0]),
                    "descriptor": self.config.descriptor,
                    "orientation_policy": self.config.orientation_policy,
                    "representation_sha256": adaptation["representation_sha256"],
                    "prepare_total_ms": elapsed_ms,
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
        try:
            matches = match_descriptors(
                payload_a.descriptors,
                payload_b.descriptors,
                lowe_ratio=float(self.config.lowe_ratio),
                matching_mode=self.config.matching_mode,
            )
            geometry = verify_geometry(payload_a, payload_b, matches.submitted, self.config)
            components = local_scoring.score_components(
                inliers=geometry.inlier_count,
                matches=len(matches.submitted),
                keypoints_a=payload_a.keypoint_count,
                keypoints_b=payload_b.keypoint_count,
            )
            score = local_scoring.raw_score(SCORE_MODE, components)
        except (ValueError, IndexError, cv2.error) as exc:
            raise MethodExecutionError(
                "detector_only_comparison_failure",
                str(exc),
                method_internal_ms=_elapsed(started),
            ) from exc
        elapsed_ms = _elapsed(started)
        return CompareOutcome(
            raw_score=score,
            method_internal_ms=elapsed_ms,
            diagnostics={
                "protocol": PROTOCOL_NAME,
                "keypoint_count_a": payload_a.keypoint_count,
                "keypoint_count_b": payload_b.keypoint_count,
                "descriptor_count_a": int(payload_a.descriptors.shape[0]),
                "descriptor_count_b": int(payload_b.descriptors.shape[0]),
                "lowe_ratio": float(self.config.lowe_ratio),
                **matches.diagnostics,
                **geometry.diagnostics,
                "score_field": SCORE_MODE,
                "score_mode": SCORE_MODE,
                "score_components": components,
                "score_direction": HIGHER_IS_MORE_SIMILAR,
                "decision_threshold_applied": False,
                "decision_threshold": None,
                "compare_total_ms": elapsed_ms,
            },
        )

    def close(self) -> None:
        return None

    def _payload(self, prepared: PreparedRepresentation) -> LocalFeatureRepresentation:
        if prepared.method != self.method_name or prepared.method_version != self.method_version:
            raise MethodExecutionError(
                "representation_identity_mismatch",
                "detector_only_v1 received a representation from another method or version.",
            )
        if prepared.representation_version != REPRESENTATION_VERSION:
            raise MethodExecutionError(
                "representation_version_mismatch",
                "detector_only_v1 received an incompatible representation version.",
            )
        if not isinstance(prepared.payload, LocalFeatureRepresentation):
            raise MethodExecutionError(
                "representation_format_mismatch",
                "detector_only_v1 received an invalid local-feature payload.",
            )
        return prepared.payload


def representation_sha256(representation: LocalFeatureRepresentation) -> str:
    """Hash deterministic representation content, excluding detector provenance."""

    digest = hashlib.sha256(b"detector-only-local-features-v1\0")
    arrays = (
        ("points", representation.points, "<f4"),
        ("sizes", representation.sizes, "<f4"),
        ("angles", representation.angles, "<f4"),
        ("responses", representation.responses, "<f4"),
        ("octaves", representation.octaves, "<i4"),
        ("class_ids", representation.class_ids, "<i4"),
        ("descriptors", representation.descriptors, "<f4"),
    )
    for name, values, dtype in arrays:
        array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            {
                "width": int(representation.width),
                "height": int(representation.height),
                "ppi": float(representation.ppi),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return digest.hexdigest()


def _positive_ppi(image_metadata: Mapping[str, object]) -> float:
    try:
        ppi = float(image_metadata["ppi"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("detector_only_v1 requires finite positive manifest PPI.") from exc
    if not math.isfinite(ppi) or ppi <= 0.0:
        raise ValueError("detector_only_v1 requires finite positive manifest PPI.")
    return ppi


def _detector_config(detector: Detector) -> dict[str, Any]:
    config = getattr(detector, "config", None)
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        payload = config.as_dict()
        if isinstance(payload, dict):
            return payload
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return {"value": str(config)}


def _method_identity(
    detector: Detector,
    descriptor: str,
    *,
    method_name: str | None,
    method_version: str | None,
) -> tuple[str, str]:
    if method_name is None and method_version is None:
        if descriptor != "rootsift":
            raise ValueError(
                "Non-RootSIFT DetectorOnlyAdapter configurations require explicit "
                "method_name and method_version matching the active descriptor."
            )
        return (
            f"{detector.detector_name}_rootsift_geometric",
            f"{detector.detector_version}-rootsift-geometric-detector-only-v1",
        )
    if method_name is None or method_version is None:
        raise ValueError("method_name and method_version must be supplied together.")
    lowered_name = method_name.lower()
    lowered_version = method_version.lower()
    if descriptor != "rootsift" and (
        "rootsift" in lowered_name or "rootsift" in lowered_version
    ):
        raise ValueError("A non-RootSIFT descriptor cannot use a RootSIFT method identity.")
    if descriptor not in lowered_name or descriptor not in lowered_version:
        raise ValueError(
            f"method_name and method_version must identify the active {descriptor!r} descriptor."
        )
    return method_name, method_version


def _elapsed(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "REPRESENTATION_VERSION",
    "SCORE_MODE",
    "DetectorOnlyAdapter",
    "DetectorOnlyProtocolConfig",
    "adapt_detector_result",
    "build_representation",
    "canonical_features",
    "representation_sha256",
]
