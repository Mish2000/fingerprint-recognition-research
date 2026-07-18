"""PPI-aware spatial configuration and physical-scale contract for v4."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from .config import HarrisZPlusConfig
from .selection import scale_suppression_distance, uniform_selection_distance


METHOD_NAME = "harriszplus_rootsift_geometric_ppi_aware"
METHOD_VERSION = "harriszplus-rootsift-geometric-ppi-aware-v4"
PARENT_METHOD_VERSION = "harriszplus-rootsift-geometric-v3"
REPRESENTATION_VERSION = "harriszplus-rootsift-ppi-aware-representation-v4"
CONFIG_SCHEMA_VERSION = "harriszplus-ppi-aware-config-v4"
PHYSICAL_CONTRACT_SCHEMA_VERSION = "harriszplus-physical-scale-contract-v4"
REFERENCE_PPI = 1000.0
DECISION_THRESHOLD = 4


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _reference_config() -> HarrisZPlusConfig:
    return HarrisZPlusConfig(backend="cuda", device="cuda:0")


@dataclass(frozen=True)
class PpiAwareHarrisZPlusConfig:
    """Operational v4 config; only spatial interpretation differs from v3."""

    reference: HarrisZPlusConfig = field(default_factory=_reference_config)
    schema_version: str = CONFIG_SCHEMA_VERSION
    method_name: str = METHOD_NAME
    method_version: str = METHOD_VERSION
    parent_method_version: str = PARENT_METHOD_VERSION
    reference_ppi: float = REFERENCE_PPI
    spatial_scale_formula: str = "manifest_ppi / 1000.0"
    border_exclusion: str = "per_scale_descriptor_safe_margin_before_uniform_cap"
    gaussian_radius_policy: str = "round_reference_kernel_radius_times_spatial_scale"
    suppression_radius_policy: str = "reference_rounded_distance_times_spatial_scale"
    duplicate_radius_reference_px: float = 1.0
    orientation_radius_policy: str = "reference_rounded_radius_times_spatial_scale"
    uniform_q_policy: str = "dimension_derived_not_independently_ppi_scaled"

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError("Unsupported v4 PPI-aware config schema.")
        if self.method_name != METHOD_NAME or self.method_version != METHOD_VERSION:
            raise ValueError("v4 method identity is frozen.")
        if self.parent_method_version != PARENT_METHOD_VERSION:
            raise ValueError("v4 parent method must be the frozen v3.")
        if self.reference_ppi != REFERENCE_PPI:
            raise ValueError("v4 reference PPI is frozen to 1000.")
        if self.reference.reference_ppi != REFERENCE_PPI:
            raise ValueError("Nested v3 reference config must use 1000 reference PPI.")
        if self.reference.max_keypoints != 3000:
            raise ValueError("v4 keypoint cap remains 3000.")
        if self.reference.lowe_ratio != 0.75:
            raise ValueError("v4 Lowe ratio remains 0.75.")
        if self.reference.ransac_threshold_at_reference_ppi != 3.0:
            raise ValueError("v4 RANSAC threshold remains 3 reference pixels.")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reference, name)

    def changed(self, **changes: Any) -> "PpiAwareHarrisZPlusConfig":
        own_fields = {
            "schema_version",
            "method_name",
            "method_version",
            "parent_method_version",
            "reference_ppi",
            "spatial_scale_formula",
            "border_exclusion",
            "gaussian_radius_policy",
            "suppression_radius_policy",
            "duplicate_radius_reference_px",
            "orientation_radius_policy",
            "uniform_q_policy",
        }
        own_changes = {key: value for key, value in changes.items() if key in own_fields}
        reference_changes = {key: value for key, value in changes.items() if key not in own_fields}
        payload = {
            "reference": (
                self.reference.changed(**reference_changes)
                if reference_changes
                else self.reference
            ),
            "schema_version": own_changes.get("schema_version", self.schema_version),
            "method_name": own_changes.get("method_name", self.method_name),
            "method_version": own_changes.get("method_version", self.method_version),
            "parent_method_version": own_changes.get(
                "parent_method_version", self.parent_method_version
            ),
            "reference_ppi": own_changes.get("reference_ppi", self.reference_ppi),
            "spatial_scale_formula": own_changes.get(
                "spatial_scale_formula", self.spatial_scale_formula
            ),
            "border_exclusion": own_changes.get("border_exclusion", self.border_exclusion),
            "gaussian_radius_policy": own_changes.get(
                "gaussian_radius_policy", self.gaussian_radius_policy
            ),
            "suppression_radius_policy": own_changes.get(
                "suppression_radius_policy", self.suppression_radius_policy
            ),
            "duplicate_radius_reference_px": own_changes.get(
                "duplicate_radius_reference_px", self.duplicate_radius_reference_px
            ),
            "orientation_radius_policy": own_changes.get(
                "orientation_radius_policy", self.orientation_radius_policy
            ),
            "uniform_q_policy": own_changes.get("uniform_q_policy", self.uniform_q_policy),
        }
        return PpiAwareHarrisZPlusConfig(**payload)

    def runtime(self, manifest_ppi: float) -> "PpiAwareRuntimeConfig":
        return PpiAwareRuntimeConfig(self, float(manifest_ppi))

    def spatial_parameter_audit(self) -> list[dict[str, str]]:
        return [
            {"parameter": "gaussian differentiation/integration sigma", "category": "scaled_by_p"},
            {"parameter": "gaussian kernel radius/size", "category": "scaled_by_p"},
            {"parameter": "greedy scale suppression", "category": "scaled_by_p"},
            {"parameter": "duplicate removal radius", "category": "scaled_by_p"},
            {"parameter": "OpenCV KeyPoint.size", "category": "scaled_by_p"},
            {"parameter": "orientation weighting sigma/radius", "category": "scaled_by_p"},
            {"parameter": "descriptor support and border margin", "category": "scaled_by_p"},
            {
                "parameter": "uniform-selection q",
                "category": "dimension_derived_not_independently_scaled",
            },
            {
                "parameter": "native image dimensions",
                "category": "dimension_derived_not_independently_scaled",
            },
            {
                "parameter": "internal Lanczos doubling for i=0,1",
                "category": "fixed_dimensionless_factor",
            },
            {
                "parameter": "strict 7x7 local-maximum topology",
                "category": "dimensionless_algorithmic_topology_unchanged",
            },
            {
                "parameter": "subpixel parabolic offset",
                "category": "dimensionless_local_pixel_cell_offset_unchanged",
            },
            {
                "parameter": "thresholds, ratios, scale indices, ranking",
                "category": "dimensionless_unchanged",
            },
            {
                "parameter": "RANSAC reprojection threshold",
                "category": "already_ppi_normalized_unchanged",
            },
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_name": self.method_name,
            "method_version": self.method_version,
            "parent_method_version": self.parent_method_version,
            "reference_ppi": self.reference_ppi,
            "spatial_scale_formula": self.spatial_scale_formula,
            "border_exclusion": self.border_exclusion,
            "gaussian_radius_policy": self.gaussian_radius_policy,
            "suppression_radius_policy": self.suppression_radius_policy,
            "duplicate_radius_reference_px": self.duplicate_radius_reference_px,
            "orientation_radius_policy": self.orientation_radius_policy,
            "uniform_q_policy": self.uniform_q_policy,
            "spatial_parameter_audit": self.spatial_parameter_audit(),
            "reference_v3_config": self.reference.as_dict(),
            "runtime_scale_tables": {
                "sd300b_1000_ppi": self.runtime(1000).scale_table(),
                "sd300c_2000_ppi": self.runtime(2000).scale_table(),
            },
        }

    @classmethod
    def from_json(cls, path: Path) -> "PpiAwareHarrisZPlusConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        reference_payload = dict(payload.pop("reference_v3_config"))
        reference_payload.pop("derived_scale_table", None)
        for name in (
            "scale_indices",
            "doubled_scale_indices",
            "derivative_kernel",
            "orientation_gradient_kernel",
            "validation_response_statistics",
        ):
            if isinstance(reference_payload.get(name), list):
                reference_payload[name] = tuple(reference_payload[name])
        for derived in ("spatial_parameter_audit", "runtime_scale_tables"):
            payload.pop(derived, None)
        return cls(reference=HarrisZPlusConfig(**reference_payload), **payload)


@dataclass(frozen=True)
class PpiAwareRuntimeConfig:
    """Per-manifest-row native-pixel view of the frozen reference config."""

    operational: PpiAwareHarrisZPlusConfig
    manifest_ppi: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.manifest_ppi) or self.manifest_ppi <= 0:
            raise ValueError("Manifest PPI must be finite and positive.")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.operational.reference, name)

    @property
    def spatial_scale(self) -> float:
        return self.manifest_ppi / self.operational.reference_ppi

    @property
    def reference_ppi(self) -> float:
        return self.operational.reference_ppi

    @property
    def duplicate_distance(self) -> float:
        return self.operational.duplicate_radius_reference_px * self.spatial_scale

    def _validate_scale_index(self, scale_index: int) -> None:
        self.operational.reference._validate_scale_index(scale_index)

    def working_image_scale(self, scale_index: int) -> float:
        return self.operational.reference.working_image_scale(scale_index)

    def nominal_sigma(self, scale_index: int) -> float:
        return self.spatial_scale * self.operational.reference.nominal_sigma(scale_index)

    def nominal_differentiation_sigma(self, scale_index: int) -> float:
        return self.nominal_sigma(scale_index)

    def working_sigma(self, scale_index: int) -> float:
        return self.nominal_sigma(scale_index) * self.working_image_scale(scale_index)

    def internal_sigma(self, scale_index: int) -> float:
        return self.working_sigma(scale_index)

    def output_sigma(self, scale_index: int) -> float:
        return self.spatial_scale * self.operational.reference.output_sigma(scale_index)

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

    def reference_kernel_radius(self, scale_index: int, kind: str) -> int:
        reference = self.operational.reference
        sigma = (
            reference.working_sigma(scale_index)
            if kind == "differentiation"
            else reference.working_integration_sigma(scale_index)
        )
        return int(math.ceil(reference.gaussian_truncate * sigma - 1.0e-12))

    def kernel_radius(self, scale_index: int, kind: str) -> int:
        return max(1, _round_half_up(self.spatial_scale * self.reference_kernel_radius(scale_index, kind)))

    def kernel_size(self, scale_index: int, kind: str) -> int:
        return 2 * self.kernel_radius(scale_index, kind) + 1

    def effective_gaussian_support_diameter(self, scale_index: int) -> float:
        radius = self.kernel_radius(scale_index, "integration")
        return 2.0 * radius / self.working_image_scale(scale_index)

    def scale_suppression_distance_working(self, scale_index: int) -> float:
        reference_distance = scale_suppression_distance(
            self.operational.reference.working_sigma(scale_index)
        )
        return self.spatial_scale * reference_distance

    def orientation_reference_radius(self, scale_index: int) -> int:
        reference_scale_sigma = (
            self.operational.reference.keypoint_size(scale_index) / 2.0
        )
        weighting_sigma = (
            reference_scale_sigma
            * self.operational.reference.orientation_gaussian_sigma_factor
        )
        return max(
            1,
            _round_half_up(
                self.operational.reference.orientation_radius_factor * weighting_sigma
            ),
        )

    def orientation_radius_pixels(self, scale_index: int) -> int:
        return max(1, _round_half_up(self.spatial_scale * self.orientation_reference_radius(scale_index)))

    def orientation_weighting_sigma(self, scale_index: int) -> float:
        return (
            self.output_integration_sigma(scale_index)
            * self.orientation_gaussian_sigma_factor
        )

    def descriptor_support_radius_estimate(self, scale_index: int) -> int:
        histogram_width = 3.0 * (self.keypoint_size(scale_index) / 2.0)
        return _round_half_up(histogram_width * math.sqrt(2.0) * 2.5)

    def descriptor_support_diameter_estimate(self, scale_index: int) -> float:
        """Continuous SIFT support diameter before integer sample-window rounding."""

        histogram_width = 3.0 * (self.keypoint_size(scale_index) / 2.0)
        return 2.0 * histogram_width * math.sqrt(2.0) * 2.5

    def descriptor_sample_window_size_estimate(self, scale_index: int) -> int:
        return 2 * self.descriptor_support_radius_estimate(scale_index) + 1

    def reference_border_margin(self, scale_index: int) -> int:
        reference_runtime = PpiAwareRuntimeConfig(self.operational, self.reference_ppi)
        return reference_runtime.descriptor_support_radius_estimate(scale_index) + 1

    def border_margin_native(self, scale_index: int) -> float:
        return self.spatial_scale * self.reference_border_margin(scale_index)

    def scale_table(self) -> list[dict[str, Any]]:
        mm_per_pixel = 25.4 / self.manifest_ppi
        return [
            {
                "scale_index": index,
                "manifest_ppi": self.manifest_ppi,
                "spatial_scale": self.spatial_scale,
                "working_image_scale": self.working_image_scale(index),
                "reference_differentiation_sigma_native_px": (
                    self.operational.reference.output_sigma(index)
                ),
                "native_differentiation_sigma_px": self.output_sigma(index),
                "reference_integration_sigma_native_px": (
                    self.operational.reference.output_integration_sigma(index)
                ),
                "native_integration_sigma_px": self.output_integration_sigma(index),
                "differentiation_kernel_radius_working_px": self.kernel_radius(index, "differentiation"),
                "differentiation_kernel_size": self.kernel_size(index, "differentiation"),
                "reference_differentiation_kernel_radius_working_px": (
                    self.reference_kernel_radius(index, "differentiation")
                ),
                "reference_differentiation_kernel_size": (
                    2 * self.reference_kernel_radius(index, "differentiation") + 1
                ),
                "integration_kernel_radius_working_px": self.kernel_radius(index, "integration"),
                "integration_kernel_size": self.kernel_size(index, "integration"),
                "reference_integration_kernel_radius_working_px": (
                    self.reference_kernel_radius(index, "integration")
                ),
                "reference_integration_kernel_size": (
                    2 * self.reference_kernel_radius(index, "integration") + 1
                ),
                "gaussian_support_diameter_native_px": self.effective_gaussian_support_diameter(index),
                "gaussian_support_diameter_mm": self.effective_gaussian_support_diameter(index) * mm_per_pixel,
                "suppression_distance_working_px": self.scale_suppression_distance_working(index),
                "suppression_distance_native_px": (
                    self.scale_suppression_distance_working(index)
                    / self.working_image_scale(index)
                ),
                "suppression_distance_mm": (
                    self.scale_suppression_distance_working(index)
                    / self.working_image_scale(index)
                    * mm_per_pixel
                ),
                "duplicate_radius_native_px": self.duplicate_distance,
                "duplicate_radius_mm": self.duplicate_distance * mm_per_pixel,
                "opencv_keypoint_size_px": self.keypoint_size(index),
                "opencv_keypoint_size_mm": self.keypoint_size(index) * mm_per_pixel,
                "orientation_weighting_sigma_px": self.orientation_weighting_sigma(index),
                "orientation_radius_px": self.orientation_radius_pixels(index),
                "orientation_radius_mm": self.orientation_radius_pixels(index) * mm_per_pixel,
                "orientation_histogram_window_size": 2 * self.orientation_radius_pixels(index) + 1,
                "descriptor_support_diameter_estimate_px": self.descriptor_support_diameter_estimate(index),
                "descriptor_support_diameter_estimate_mm": (
                    self.descriptor_support_diameter_estimate(index) * mm_per_pixel
                ),
                "descriptor_sample_window_size_estimate": (
                    self.descriptor_sample_window_size_estimate(index)
                ),
                "border_margin_native_px": self.border_margin_native(index),
                "border_margin_mm": self.border_margin_native(index) * mm_per_pixel,
            }
            for index in self.scale_indices
        ]


def build_physical_scale_contract(config: PpiAwareHarrisZPlusConfig) -> dict[str, Any]:
    b = config.runtime(1000.0)
    c = config.runtime(2000.0)
    b_rows = {row["scale_index"]: row for row in b.scale_table()}
    c_rows = {row["scale_index"]: row for row in c.scale_table()}
    physical_fields = (
        "native_differentiation_sigma_px",
        "native_integration_sigma_px",
        "gaussian_support_diameter_native_px",
        "suppression_distance_native_px",
        "duplicate_radius_native_px",
        "opencv_keypoint_size_px",
        "orientation_radius_px",
        "descriptor_support_diameter_estimate_px",
        "border_margin_native_px",
    )
    comparisons: list[dict[str, Any]] = []
    for index in config.scale_indices:
        for field_name in physical_fields:
            b_px = float(b_rows[index][field_name])
            c_px = float(c_rows[index][field_name])
            b_mm = b_px * 25.4 / 1000.0
            c_mm = c_px * 25.4 / 2000.0
            tolerance = max(0.001, 0.01 * abs(b_mm))
            comparisons.append(
                {
                    "scale_index": index,
                    "parameter": field_name,
                    "b_pixels": b_px,
                    "c_pixels": c_px,
                    "b_mm": b_mm,
                    "c_mm": c_mm,
                    "absolute_delta_mm": abs(b_mm - c_mm),
                    "tolerance_mm": tolerance,
                    "passed": abs(b_mm - c_mm) <= tolerance + 1.0e-12,
                }
            )
    ransac = {
        "reference_ppi": 1000.0,
        "reference_threshold_px": 3.0,
        "sd300b_native_threshold_px": 3.0,
        "sd300c_native_threshold_px": 6.0,
        "sd300b_threshold_mm": 3.0 * 25.4 / 1000.0,
        "sd300c_threshold_mm": 6.0 * 25.4 / 2000.0,
    }
    ransac["passed"] = math.isclose(
        ransac["sd300b_threshold_mm"],
        ransac["sd300c_threshold_mm"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    # The frozen engineering fixture demonstrates the dimension-derived Eq. 14
    # behavior on corresponding native scans. C is approximately 2x in both
    # dimensions and q therefore grows approximately 2x without another p factor.
    uniform_b = uniform_selection_distance(941, 622, config.max_keypoints)
    uniform_c = uniform_selection_distance(1883, 1244, config.max_keypoints)
    uniform_b_mm = uniform_b * 25.4 / 1000.0
    uniform_c_mm = uniform_c * 25.4 / 2000.0
    uniform_tolerance = max(0.001, 0.01 * abs(uniform_b_mm))
    uniform_q = {
        "formula": "sqrt(8*m*n/(pi*k))",
        "maximum_keypoints": config.max_keypoints,
        "sd300b_example_dimensions": [941, 622],
        "sd300c_example_dimensions": [1883, 1244],
        "sd300b_q_native_px": uniform_b,
        "sd300c_q_native_px": uniform_c,
        "sd300b_q_mm": uniform_b_mm,
        "sd300c_q_mm": uniform_c_mm,
        "pixel_ratio_c_over_b": uniform_c / uniform_b,
        "independent_ppi_multiplier": False,
        "absolute_delta_mm": abs(uniform_b_mm - uniform_c_mm),
        "tolerance_mm": uniform_tolerance,
        "passed": abs(uniform_b_mm - uniform_c_mm)
        <= uniform_tolerance + 1.0e-12,
    }
    return {
        "schema_version": PHYSICAL_CONTRACT_SCHEMA_VERSION,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "parent_method_version": PARENT_METHOD_VERSION,
        "reference_ppi": REFERENCE_PPI,
        "spatial_scale_formula": "manifest_ppi / 1000.0",
        "uniform_q_policy": "uniform q is dimension-derived and not independently PPI-scaled",
        "subpixel_policy": (
            "unchanged dimensionless parabolic offset in the local pixel cell; "
            "no clipping distance is present"
        ),
        "local_maximum_policy": (
            "strict 7x7 topology is an unchanged algorithmic neighborhood, not an "
            "independently tuned physical radius"
        ),
        "sd300b": {"ppi": 1000, "spatial_scale": 1.0, "scales": b.scale_table()},
        "sd300c": {"ppi": 2000, "spatial_scale": 2.0, "scales": c.scale_table()},
        "comparisons": comparisons,
        "uniform_q": uniform_q,
        "ransac": ransac,
        "tolerance_policy": "abs(mm_B - mm_C) <= max(0.001 mm, 1% of reference value)",
        "passed": (
            all(item["passed"] for item in comparisons)
            and uniform_q["passed"]
            and ransac["passed"]
        ),
    }


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DECISION_THRESHOLD",
    "METHOD_NAME",
    "METHOD_VERSION",
    "PARENT_METHOD_VERSION",
    "PHYSICAL_CONTRACT_SCHEMA_VERSION",
    "PpiAwareHarrisZPlusConfig",
    "PpiAwareRuntimeConfig",
    "REFERENCE_PPI",
    "REPRESENTATION_VERSION",
    "build_physical_scale_contract",
]
