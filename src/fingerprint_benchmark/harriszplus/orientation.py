"""Deterministic single-orientation assignment for HarrisZ+ keypoints.

The implementation follows the SIFT dominant-gradient construction while
remaining independent from OpenCV's detector-side orientation assignment.
Image-coordinate gradients are signed over the full 360-degree range; because
image ``y`` increases downward, the returned angle uses OpenCV's clockwise
image-coordinate convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class OrientationResult:
    """One interpolated dominant angle and its audit information."""

    angle_degrees: float
    histogram: np.ndarray
    peak_bin: int
    peak_offset: float
    peak_value: float
    sample_count: int
    radius_pixels: int
    weighting_sigma: float


_BORDER_MODES = {
    "reflect101": cv2.BORDER_REFLECT_101,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
}


def signed_gradient_maps(
    image: np.ndarray,
    *,
    gradient_kernel: Sequence[float] = (-1.0, 0.0, 1.0),
    border_mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
    """Return full signed horizontal and vertical central-difference maps."""

    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Orientation assignment requires a non-empty grayscale image.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Orientation image must be numeric.")
    source = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("Orientation image contains non-finite values.")
    kernel = np.asarray(tuple(gradient_kernel), dtype=np.float32)
    if kernel.shape != (3,) or not np.array_equal(
        kernel,
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
    ):
        raise ValueError("The frozen orientation gradient kernel must be [-1, 0, 1].")
    try:
        border = _BORDER_MODES[str(border_mode).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported orientation border mode: {border_mode!r}.") from exc
    gradient_x = cv2.filter2D(
        source,
        cv2.CV_32F,
        kernel.reshape(1, 3),
        borderType=border,
    )
    gradient_y = cv2.filter2D(
        source,
        cv2.CV_32F,
        kernel.reshape(3, 1),
        borderType=border,
    )
    return gradient_x, gradient_y


def dominant_gradient_orientation(
    image: np.ndarray,
    x: float,
    y: float,
    scale_sigma: float,
    *,
    bins: int = 36,
    gaussian_sigma_factor: float = 1.5,
    radius_factor: float = 3.0,
    smoothing_passes: int = 6,
    gradient_kernel: Sequence[float] = (-1.0, 0.0, 1.0),
    border_mode: str = "reflect",
) -> float:
    """Return one SIFT-style dominant angle in ``[0, 360)``."""

    gradient_x, gradient_y = signed_gradient_maps(
        image,
        gradient_kernel=gradient_kernel,
        border_mode=border_mode,
    )
    return orientation_from_gradients(
        gradient_x,
        gradient_y,
        x,
        y,
        scale_sigma,
        bins=bins,
        gaussian_sigma_factor=gaussian_sigma_factor,
        radius_factor=radius_factor,
        smoothing_passes=smoothing_passes,
    ).angle_degrees


def assign_dominant_orientation(
    image: np.ndarray,
    x: float,
    y: float,
    scale_sigma: float,
    **kwargs: Any,
) -> float:
    """Compatibility name for :func:`dominant_gradient_orientation`."""

    return dominant_gradient_orientation(image, x, y, scale_sigma, **kwargs)


def orientation_from_gradients(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    x: float,
    y: float,
    scale_sigma: float,
    *,
    bins: int = 36,
    gaussian_sigma_factor: float = 1.5,
    radius_factor: float = 3.0,
    smoothing_passes: int = 6,
) -> OrientationResult:
    """Build, smooth, and interpolate one orientation from precomputed gradients."""

    gx = np.asarray(gradient_x, dtype=np.float32)
    gy = np.asarray(gradient_y, dtype=np.float32)
    if gx.ndim != 2 or gx.shape != gy.shape or gx.size == 0:
        raise ValueError("Orientation gradient maps must be non-empty and have equal 2-D shapes.")
    _validate_parameters(
        x=x,
        y=y,
        scale_sigma=scale_sigma,
        bins=bins,
        gaussian_sigma_factor=gaussian_sigma_factor,
        radius_factor=radius_factor,
        smoothing_passes=smoothing_passes,
    )
    weighting_sigma = float(scale_sigma) * float(gaussian_sigma_factor)
    radius = max(1, int(math.floor(float(radius_factor) * weighting_sigma + 0.5)))
    histogram, sample_count = build_orientation_histogram(
        gx,
        gy,
        x=float(x),
        y=float(y),
        weighting_sigma=weighting_sigma,
        radius=radius,
        bins=int(bins),
    )
    smoothed = smooth_circular_histogram(histogram, passes=int(smoothing_passes))
    angle, peak_bin, peak_offset, peak_value = parabolic_peak_angle(smoothed)
    return OrientationResult(
        angle_degrees=angle,
        histogram=smoothed,
        peak_bin=peak_bin,
        peak_offset=peak_offset,
        peak_value=peak_value,
        sample_count=sample_count,
        radius_pixels=radius,
        weighting_sigma=weighting_sigma,
    )


def build_orientation_histogram(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    x: float,
    y: float,
    weighting_sigma: float,
    radius: int,
    bins: int = 36,
) -> tuple[np.ndarray, int]:
    """Accumulate a Gaussian-weighted magnitude histogram with circular interpolation."""

    gx = np.asarray(gradient_x, dtype=np.float32)
    gy = np.asarray(gradient_y, dtype=np.float32)
    if gx.ndim != 2 or gx.shape != gy.shape:
        raise ValueError("Gradient maps must be 2-D and have identical shapes.")
    if bins < 3 or radius < 1 or not math.isfinite(weighting_sigma) or weighting_sigma <= 0.0:
        raise ValueError("Histogram bins, radius, and weighting sigma must be positive.")
    height, width = gx.shape
    left = max(0, int(math.ceil(float(x) - int(radius))))
    right = min(width - 1, int(math.floor(float(x) + int(radius))))
    top = max(0, int(math.ceil(float(y) - int(radius))))
    bottom = min(height - 1, int(math.floor(float(y) + int(radius))))
    histogram = np.zeros(int(bins), dtype=np.float64)
    if left > right or top > bottom:
        return histogram, 0

    xx = np.arange(left, right + 1, dtype=np.float64)
    yy = np.arange(top, bottom + 1, dtype=np.float64)
    delta_x = xx[None, :] - float(x)
    delta_y = yy[:, None] - float(y)
    squared_radius = delta_x * delta_x + delta_y * delta_y
    support = squared_radius <= float(radius * radius)
    local_gx = np.asarray(gx[top : bottom + 1, left : right + 1], dtype=np.float64)
    local_gy = np.asarray(gy[top : bottom + 1, left : right + 1], dtype=np.float64)
    magnitude = np.hypot(local_gx, local_gy)
    valid = support & np.isfinite(magnitude) & (magnitude > 0.0)
    if not np.any(valid):
        return histogram, 0

    weights = np.exp(-squared_radius / (2.0 * float(weighting_sigma) ** 2))
    contributions = (weights * magnitude)[valid]
    angles = np.mod(np.arctan2(local_gy, local_gx), 2.0 * math.pi)[valid]
    positions = angles * (float(bins) / (2.0 * math.pi))
    lower_unwrapped = np.floor(positions).astype(np.int64)
    fraction = positions - lower_unwrapped
    lower = np.mod(lower_unwrapped, int(bins))
    upper = np.mod(lower + 1, int(bins))
    np.add.at(histogram, lower, contributions * (1.0 - fraction))
    np.add.at(histogram, upper, contributions * fraction)
    return histogram, int(np.count_nonzero(valid))


def smooth_circular_histogram(histogram: np.ndarray, *, passes: int) -> np.ndarray:
    """Apply the frozen circular three-bin SIFT smoothing pass repeatedly."""

    values = np.asarray(histogram, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("Orientation histogram must be a finite one-dimensional array.")
    if passes < 0:
        raise ValueError("Histogram smoothing pass count must be non-negative.")
    result = values.copy()
    for _ in range(int(passes)):
        result = (np.roll(result, 1) + result + np.roll(result, -1)) / 3.0
    return result


def parabolic_peak_angle(histogram: np.ndarray) -> tuple[float, int, float, float]:
    """Interpolate all tied global peaks and select the lowest resulting angle."""

    values = np.asarray(histogram, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("Orientation histogram must be a finite one-dimensional array.")
    peak_value = float(np.max(values))
    if peak_value <= 0.0:
        return 0.0, 0, 0.0, peak_value
    candidates = np.flatnonzero(values == peak_value)
    interpolated: list[tuple[float, int, float]] = []
    for raw_index in candidates:
        index = int(raw_index)
        left = float(values[(index - 1) % values.size])
        center = float(values[index])
        right = float(values[(index + 1) % values.size])
        denominator = left - 2.0 * center + right
        if denominator == 0.0 or not math.isfinite(denominator):
            offset = 0.0
        else:
            offset = 0.5 * (left - right) / denominator
            offset = min(0.5, max(-0.5, float(offset)))
        angle = float(((index + offset) * 360.0 / values.size) % 360.0)
        interpolated.append((angle, index, offset))
    angle, index, offset = min(interpolated, key=lambda item: (item[0], item[1]))
    return angle, index, offset, peak_value


def assign_orientations(
    image: np.ndarray,
    keypoints: Iterable[object],
    config: object,
) -> np.ndarray:
    """Assign one angle per detector keypoint using the frozen adapter config."""

    angles, _ = assign_orientations_with_diagnostics(image, keypoints, config)
    return angles


def assign_orientations_with_diagnostics(
    image: np.ndarray,
    keypoints: Iterable[object],
    config: object,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign one angle per keypoint and return aggregate audit diagnostics."""

    records = tuple(keypoints)
    _validate_config_policy(config)
    gradient_kernel = tuple(getattr(config, "orientation_gradient_kernel", (-1.0, 0.0, 1.0)))
    border_mode = str(getattr(config, "orientation_border_mode", "reflect"))
    gx, gy = signed_gradient_maps(
        image,
        gradient_kernel=gradient_kernel,
        border_mode=border_mode,
    )
    parameters = {
        "bins": int(getattr(config, "orientation_bins")),
        "gaussian_sigma_factor": float(
            getattr(config, "orientation_gaussian_sigma_factor")
        ),
        "radius_factor": float(getattr(config, "orientation_radius_factor")),
        "smoothing_passes": int(
            getattr(config, "orientation_histogram_smoothing_passes")
        ),
    }
    results: list[OrientationResult] = []
    for keypoint in records:
        # OpenCV size is a diameter: size == 2 * output integration sigma.
        scale_sigma = float(getattr(keypoint, "size")) / 2.0
        results.append(
            orientation_from_gradients(
                gx,
                gy,
                float(getattr(keypoint, "x")),
                float(getattr(keypoint, "y")),
                scale_sigma,
                **parameters,
            )
        )
    angles = np.asarray([result.angle_degrees for result in results], dtype=np.float32)
    diagnostics: dict[str, Any] = {
        "orientation_method": "sift_style_dominant_signed_gradient",
        "orientation_count": int(angles.size),
        "orientation_bins": parameters["bins"],
        "orientation_gradient_kernel": list(gradient_kernel),
        "orientation_border_mode": border_mode,
        "orientation_gaussian_sigma_factor": parameters["gaussian_sigma_factor"],
        "orientation_radius_factor_of_weighting_sigma": parameters["radius_factor"],
        "orientation_histogram_interpolation": "linear_circular",
        "orientation_histogram_smoothing": "repeated_three_bin_circular_mean",
        "orientation_histogram_smoothing_passes": parameters["smoothing_passes"],
        "orientation_peak_interpolation": "three_sample_parabola_clipped_to_half_bin",
        "orientation_tie_policy": "lowest_interpolated_angle",
        "orientation_180_degree_policy": "full_signed_gradient_no_random_flip",
        "orientation_scale_source": "opencv_keypoint_size_divided_by_2",
        "empty_histogram_angle_degrees": 0.0,
        "orientation_sample_count_min": (
            min(result.sample_count for result in results) if results else 0
        ),
        "orientation_sample_count_max": (
            max(result.sample_count for result in results) if results else 0
        ),
    }
    return angles, diagnostics


def _validate_config_policy(config: object) -> None:
    expected = {
        "orientation_histogram_interpolation": "linear_circular",
        "orientation_peak_interpolation": "parabolic",
        "orientation_tie_policy": "lowest_angle",
        "orientation_scale_source": "output_integration_sigma",
        "orientation_border_mode": "reflect",
    }
    for field, required in expected.items():
        value = str(getattr(config, field, required))
        if value != required:
            raise ValueError(f"{field} is frozen to {required!r}; got {value!r}.")


def _validate_parameters(
    *,
    x: float,
    y: float,
    scale_sigma: float,
    bins: int,
    gaussian_sigma_factor: float,
    radius_factor: float,
    smoothing_passes: int,
) -> None:
    if not all(math.isfinite(value) for value in (x, y, scale_sigma)):
        raise ValueError("Keypoint coordinates and scale must be finite.")
    if scale_sigma <= 0.0:
        raise ValueError("Keypoint scale sigma must be positive.")
    if bins != 36:
        raise ValueError("The frozen HarrisZ+ orientation histogram has exactly 36 bins.")
    if not math.isfinite(gaussian_sigma_factor) or gaussian_sigma_factor <= 0.0:
        raise ValueError("Orientation Gaussian sigma factor must be positive.")
    if not math.isfinite(radius_factor) or radius_factor <= 0.0:
        raise ValueError("Orientation radius factor must be positive.")
    if smoothing_passes < 0:
        raise ValueError("Orientation smoothing pass count must be non-negative.")
