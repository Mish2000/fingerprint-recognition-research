"""Frozen configuration and scale mapping for the clean-room HarrisZ+ method."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any


METHOD_NAME = "harriszplus_rootsift_geometric"
METHOD_VERSION = "harriszplus-rootsift-geometric-v1"
CONFIG_SCHEMA_VERSION = "harriszplus-config-schema-v1"
PAPER_VERSION = "arXiv:2109.12925v6"
DETECTOR_BACKENDS = ("reference_cpu", "cuda")


@dataclass(frozen=True, slots=True)
class HarrisZPlusConfig:
    """All algorithmic and validation choices for the public method.

    The defaults are the research configuration.  A smaller ``max_keypoints``
    may be used by focused tests, but the hard method ceiling remains 3000.
    """

    schema_version: str = CONFIG_SCHEMA_VERSION
    paper_version: str = PAPER_VERSION
    backend: str = "cuda"
    device: str | None = None
    rng_seed: int = 0
    detector_dtype: str = "float32"

    # HarrisZ/HarrisZ+ detector choices.
    color_grad: bool = False
    scale_indices: tuple[int, ...] = (0, 1, 2, 3, 4)
    doubled_scale_indices: tuple[int, ...] = (0, 1)
    scale_base: float = 2.0
    scale_exponent_divisor: float = 2.0
    integration_scale_multiplier: float = math.sqrt(2.0)
    gaussian_truncate: float = 3.0
    derivative_kernel: tuple[float, float, float] = (-1.0, 0.0, 1.0)
    derivative_border_policy: str = "zero_frame"
    gaussian_border_mode: str = "opencv_BORDER_REFLECT_edge_including"
    zscore_ddof: int = 1
    zscore_epsilon: float = 0.0
    zscore_flat_policy: str = "zero"
    raw_edge_mean_multiplier: float = 1.0
    raw_edge_comparison: str = "strict_greater_than"
    response_threshold: float = 0.0
    response_threshold_comparison: str = "strict_greater_than"
    edge_mask_threshold: float = 0.31
    edge_mask_threshold_comparison: str = "strict_greater_than"
    local_maximum_radius: int = 3
    local_maximum_policy: str = "strict_unique_7x7"
    local_maximum_border_policy: str = "partial_in_bounds"
    # Selection-only stabilization of mathematical ties.  This is 0.2% of
    # the formal CPU/CUDA response-map absolute tolerance and never changes
    # the returned raw response values.
    local_maximum_tie_atol: float = 1.0e-6
    eigen_axis_ratio_threshold: float = 0.25
    duplicate_distance: float = 1.0
    subpixel_policy: str = "separable_parabola_no_clipping_zero_invalid_denominator_or_border"
    ranking_policy: str = "response_desc_scale_desc_y_asc_x_asc_source_asc"
    uniform_distance_formula: str = "sqrt(8mn/(pi*k))"
    uniform_spacing_scope: str = "pass_local"
    max_keypoints: int = 3000
    hard_max_keypoints: int = 3000

    # Image construction and detector-to-descriptor scale mapping.
    lanczos_interpolation: str = "INTER_LANCZOS4"
    keypoint_size_formula: str = "2_times_output_integration_sigma"

    # Deterministic OpenCV SIFT orientation/descriptor choices.
    opencv_threads: int = 16
    opencv_optimized: bool = True
    orientation_bins: int = 36
    orientation_gaussian_sigma_factor: float = 1.5
    orientation_radius_factor: float = 3.0
    orientation_histogram_smoothing_passes: int = 6
    orientation_gradient_kernel: tuple[int, int, int] = (-1, 0, 1)
    orientation_border_mode: str = "reflect"
    orientation_histogram_interpolation: str = "linear_circular"
    orientation_peak_interpolation: str = "parabolic"
    orientation_tie_policy: str = "lowest_angle"
    orientation_scale_source: str = "output_integration_sigma"
    sift_n_octave_layers: int = 3
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6
    rootsift_zero_norm_epsilon: float = 0.0

    # Existing matching and partial-affine geometry contract.
    descriptor_mode: str = "rootsift"
    matching_mode: str = "mutual"
    score_mode: str = "geometric_inlier_count"
    lowe_ratio: float = 0.75
    minimum_descriptors: int = 2
    minimum_geometry_matches: int = 3
    geometry_model: str = "affine_partial_2d"
    ransac_threshold_at_reference_ppi: float = 3.0
    ransac_confidence: float = 0.99
    ransac_max_iterations: int = 2000
    ransac_refine_iterations: int = 10
    reference_ppi: float = 1000.0
    normalize_coordinates_by_ppi: bool = True

    # CPU/CUDA acceptance gates frozen before the formal immutable preflight
    # and before any 500-pair result was observed.
    validation_response_atol: float = 5.0e-4
    validation_response_rtol: float = 2.0e-4
    validation_response_max_absolute_delta: float = 0.1
    validation_response_minimum_pixel_coverage: float = 0.9999
    validation_synthetic_response_minimum_pixel_coverage: float = 1.0
    validation_response_statistics: tuple[str, ...] = (
        "minimum",
        "maximum",
        "mean",
        "sample_stddev",
    )
    validation_candidate_preuniform_minimum_ratio: float = 0.9995
    validation_uniform_final_count_exact: bool = True
    validation_spatial_tolerance_original_px: float = 0.75
    validation_scale_index_exact: bool = True
    validation_minimum_keypoint_agreement: float = 0.95
    validation_minimum_order_spearman_rank_correlation: float = 0.99
    validation_ordering_response_tie_atol: float = 1.0e-6
    validation_ordering_coordinate_tie_atol: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported HarrisZ+ config schema: {self.schema_version!r}.")
        if self.paper_version != PAPER_VERSION:
            raise ValueError(f"HarrisZ+ paper version is frozen to {PAPER_VERSION!r}.")
        if self.backend not in DETECTOR_BACKENDS:
            raise ValueError(f"backend must be one of {DETECTOR_BACKENDS}; got {self.backend!r}.")
        if self.detector_dtype != "float32":
            raise ValueError("The HarrisZ+ detector dtype is frozen to float32.")
        if self.scale_indices != (0, 1, 2, 3, 4):
            raise ValueError("HarrisZ+ scale_indices are frozen to (0, 1, 2, 3, 4).")
        if self.doubled_scale_indices != (0, 1):
            raise ValueError("HarrisZ+ doubles exactly scale indexes 0 and 1.")
        if self.color_grad:
            raise ValueError("This grayscale-only method requires color_grad=False.")
        if self.rng_seed != 0:
            raise ValueError("The reproducibility seed is frozen to zero.")
        if (
            self.scale_base != 2.0
            or self.scale_exponent_divisor != 2.0
            or self.integration_scale_multiplier != math.sqrt(2.0)
            or self.gaussian_truncate != 3.0
        ):
            raise ValueError("HarrisZ+ Gaussian scale-space constants are frozen.")
        if self.derivative_kernel != (-1.0, 0.0, 1.0):
            raise ValueError("The mathematical central-difference kernel is frozen to [-1, 0, 1].")
        if (
            self.derivative_border_policy != "zero_frame"
            or self.gaussian_border_mode != "opencv_BORDER_REFLECT_edge_including"
        ):
            raise ValueError("Derivative and Gaussian border policies are frozen.")
        if self.zscore_ddof != 1 or self.zscore_epsilon != 0.0 or self.zscore_flat_policy != "zero":
            raise ValueError("HarrisZ+ requires sample standard deviation and flat-to-zero z-scores.")
        if (
            self.raw_edge_mean_multiplier != 1.0
            or self.raw_edge_comparison != "strict_greater_than"
            or self.response_threshold != 0.0
            or self.response_threshold_comparison != "strict_greater_than"
            or self.edge_mask_threshold != 0.31
            or self.edge_mask_threshold_comparison != "strict_greater_than"
            or self.local_maximum_radius != 3
            or self.local_maximum_policy != "strict_unique_7x7"
            or self.local_maximum_border_policy != "partial_in_bounds"
            or self.local_maximum_tie_atol != 1.0e-6
            or self.eigen_axis_ratio_threshold != 0.25
            or self.duplicate_distance != 1.0
        ):
            raise ValueError(
                "HarrisZ+ raw-edge, response, smoothed-mask, and local-maximum "
                "threshold policies are frozen."
            )
        if (
            self.subpixel_policy
            != "separable_parabola_no_clipping_zero_invalid_denominator_or_border"
            or self.ranking_policy != "response_desc_scale_desc_y_asc_x_asc_source_asc"
            or self.uniform_distance_formula != "sqrt(8mn/(pi*k))"
            or self.uniform_spacing_scope != "pass_local"
        ):
            raise ValueError("HarrisZ+ subpixel, ranking, and uniform-selection policies are frozen.")
        if not 0 < self.max_keypoints <= self.hard_max_keypoints == 3000:
            raise ValueError("max_keypoints must be in [1, 3000], with a hard ceiling of 3000.")
        if self.opencv_threads != 16 or not self.opencv_optimized:
            raise ValueError("OpenCV execution is frozen to 16 threads with optimizations enabled.")
        if self.lanczos_interpolation != "INTER_LANCZOS4":
            raise ValueError("Doubled arrays must be supplied using OpenCV INTER_LANCZOS4.")
        if self.keypoint_size_formula != "2_times_output_integration_sigma":
            raise ValueError("The OpenCV keypoint-size mapping is frozen for this method.")
        if self.orientation_bins != 36:
            raise ValueError("Orientation assignment requires exactly 36 histogram bins.")
        if (
            self.orientation_gaussian_sigma_factor != 1.5
            or self.orientation_radius_factor != 3.0
            or self.orientation_histogram_smoothing_passes != 6
            or self.orientation_gradient_kernel != (-1, 0, 1)
            or self.orientation_border_mode != "reflect"
            or self.orientation_histogram_interpolation != "linear_circular"
            or self.orientation_peak_interpolation != "parabolic"
            or self.orientation_tie_policy != "lowest_angle"
            or self.orientation_scale_source != "output_integration_sigma"
        ):
            raise ValueError("Orientation-assignment choices are frozen for this method.")
        if (
            self.sift_n_octave_layers != 3
            or self.sift_contrast_threshold != 0.04
            or self.sift_edge_threshold != 10.0
            or self.sift_sigma != 1.6
            or self.rootsift_zero_norm_epsilon != 0.0
        ):
            raise ValueError("SIFT/RootSIFT descriptor choices are frozen for this method.")
        if (
            self.descriptor_mode != "rootsift"
            or self.matching_mode != "mutual"
            or self.score_mode != "geometric_inlier_count"
            or self.geometry_model != "affine_partial_2d"
        ):
            raise ValueError("Descriptor, matching, geometry, and score modes are frozen.")
        if (
            self.lowe_ratio != 0.75
            or self.minimum_descriptors != 2
            or self.minimum_geometry_matches != 3
            or self.ransac_threshold_at_reference_ppi != 3.0
            or self.ransac_confidence != 0.99
            or self.ransac_max_iterations != 2000
            or self.ransac_refine_iterations != 10
            or self.reference_ppi != 1000.0
            or not self.normalize_coordinates_by_ppi
        ):
            raise ValueError("Matching and partial-affine RANSAC choices are frozen.")
        if (
            self.validation_response_atol != 5.0e-4
            or self.validation_response_rtol != 2.0e-4
            or self.validation_response_max_absolute_delta != 0.1
            or self.validation_response_minimum_pixel_coverage != 0.9999
            or self.validation_synthetic_response_minimum_pixel_coverage != 1.0
            or self.validation_response_statistics
            != ("minimum", "maximum", "mean", "sample_stddev")
            or self.validation_candidate_preuniform_minimum_ratio != 0.9995
            or not self.validation_uniform_final_count_exact
            or self.validation_spatial_tolerance_original_px != 0.75
            or not self.validation_scale_index_exact
            or self.validation_minimum_keypoint_agreement != 0.95
            or self.validation_minimum_order_spearman_rank_correlation != 0.99
            or self.validation_ordering_response_tie_atol != 1.0e-6
            or self.validation_ordering_coordinate_tie_atol != 1.0e-3
        ):
            raise ValueError(
                "CPU/CUDA validation gates are frozen before the formal immutable "
                "preflight and before any 500-pair result."
            )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["derived_scale_table"] = self.scale_table()
        return payload

    def changed(self, **changes: Any) -> "HarrisZPlusConfig":
        return replace(self, **changes)

    @classmethod
    def from_json(cls, path: Path) -> "HarrisZPlusConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("HarrisZ+ config JSON must contain an object.")
        declared_scale_table = payload.pop("derived_scale_table", None)
        for tuple_field in (
            "scale_indices",
            "doubled_scale_indices",
            "derivative_kernel",
            "orientation_gradient_kernel",
            "validation_response_statistics",
        ):
            if tuple_field in payload and isinstance(payload[tuple_field], list):
                payload[tuple_field] = tuple(payload[tuple_field])
        config = cls(**payload)
        if declared_scale_table is not None and declared_scale_table != config.scale_table():
            raise ValueError("The serialized HarrisZ+ derived scale table is inconsistent.")
        return config

    def nominal_sigma(self, scale_index: int) -> float:
        """Differentiation sigma in original-image pixels before i=0 clamping."""

        self._validate_scale_index(scale_index)
        return self.scale_base ** ((float(scale_index) - 1.0) / self.scale_exponent_divisor)

    def working_image_scale(self, scale_index: int) -> float:
        self._validate_scale_index(scale_index)
        return 2.0 if scale_index in self.doubled_scale_indices else 1.0

    def working_sigma(self, scale_index: int) -> float:
        """Differentiation sigma in pixels of the supplied working image."""

        return self.nominal_sigma(scale_index) * self.working_image_scale(scale_index)

    def nominal_differentiation_sigma(self, scale_index: int) -> float:
        return self.nominal_sigma(scale_index)

    def internal_sigma(self, scale_index: int) -> float:
        return self.working_sigma(scale_index)

    def output_sigma(self, scale_index: int) -> float:
        """Final differentiation sigma in original-image pixels.

        HarrisZ+ promotes index 0 to the index-1 output scale so that indexes
        0 and 1 have the same descriptor support.
        """

        self._validate_scale_index(scale_index)
        adjusted_index = 1 if scale_index == 0 else scale_index
        return self.nominal_sigma(adjusted_index)

    def working_integration_sigma(self, scale_index: int) -> float:
        return self.integration_scale_multiplier * self.working_sigma(scale_index)

    def internal_integration_sigma(self, scale_index: int) -> float:
        return self.working_integration_sigma(scale_index)

    def output_integration_sigma(self, scale_index: int) -> float:
        return self.integration_scale_multiplier * self.output_sigma(scale_index)

    def nominal_integration_sigma(self, scale_index: int) -> float:
        return self.integration_scale_multiplier * self.nominal_sigma(scale_index)

    def keypoint_size(self, scale_index: int) -> float:
        return 2.0 * self.output_integration_sigma(scale_index)

    def effective_gaussian_support_diameter(self, scale_index: int) -> float:
        """Integration-kernel support diameter measured in original pixels."""

        unrounded_radius = self.gaussian_truncate * self.working_integration_sigma(scale_index)
        nearest_integer = round(unrounded_radius)
        if math.isclose(unrounded_radius, nearest_integer, rel_tol=1.0e-12, abs_tol=1.0e-12):
            radius = int(nearest_integer)
        else:
            radius = math.ceil(unrounded_radius)
        return (2.0 * radius + 1.0) / self.working_image_scale(scale_index)

    def scale_table(self) -> list[dict[str, float | int]]:
        """Return the explicit five-row detector scale mapping.

        Every distance is expressed in original-image pixels unless the key
        explicitly says ``working``.  Storing this derived table with frozen
        configuration artifacts makes the index-0 promotion and doubled-image
        bookkeeping independently auditable.
        """

        return [
            {
                "scale_index": scale_index,
                "nominal_differentiation_sigma": self.nominal_sigma(scale_index),
                "working_differentiation_sigma": self.working_sigma(scale_index),
                "output_differentiation_sigma": self.output_sigma(scale_index),
                "nominal_integration_sigma": self.nominal_integration_sigma(scale_index),
                "working_integration_sigma": self.working_integration_sigma(scale_index),
                "output_integration_sigma": self.output_integration_sigma(scale_index),
                "working_image_scale": self.working_image_scale(scale_index),
                "effective_support_diameter_original_px": (
                    self.effective_gaussian_support_diameter(scale_index)
                ),
                "keypoint_size": self.keypoint_size(scale_index),
            }
            for scale_index in self.scale_indices
        ]

    def _validate_scale_index(self, scale_index: int) -> None:
        if scale_index not in self.scale_indices:
            raise ValueError(f"Unsupported HarrisZ+ scale index: {scale_index!r}.")

    @property
    def cpu_cuda_response_atol(self) -> float:
        return self.validation_response_atol

    @property
    def cpu_cuda_response_rtol(self) -> float:
        return self.validation_response_rtol

    @property
    def cpu_cuda_spatial_tolerance(self) -> float:
        return self.validation_spatial_tolerance_original_px

    @property
    def cpu_cuda_minimum_keypoint_agreement(self) -> float:
        return self.validation_minimum_keypoint_agreement
