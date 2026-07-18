from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fingerprint_benchmark.harriszplus.orientation import (
    assign_orientations_with_diagnostics,
    dominant_gradient_orientation,
    orientation_from_gradients,
    parabolic_peak_angle,
    signed_gradient_maps,
)


def _constant_gradients(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((41, 41), dx, dtype=np.float32),
        np.full((41, 41), dy, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [(1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (-1.0, 0.0, 180.0)],
)
def test_signed_orientation_uses_full_360_degree_range(
    dx: float, dy: float, expected: float
) -> None:
    gx, gy = _constant_gradients(dx, dy)
    result = orientation_from_gradients(gx, gy, 20.0, 20.0, 2.0)
    assert result.angle_degrees == pytest.approx(expected, abs=1e-6)
    assert 0.0 <= result.angle_degrees < 360.0


def test_orientation_rotation_consistency_on_a_synthetic_ramp() -> None:
    image = np.tile(np.arange(65, dtype=np.float32), (65, 1))
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    first = dominant_gradient_orientation(
        image, 32.0, 32.0, 2.0, border_mode="reflect"
    )
    second = dominant_gradient_orientation(
        rotated, 32.0, 32.0, 2.0, border_mode="reflect"
    )
    assert first == pytest.approx(0.0, abs=1e-6)
    assert second == pytest.approx(90.0, abs=1e-6)


def test_orientation_peak_tie_chooses_the_lowest_interpolated_angle() -> None:
    histogram = np.zeros(36, dtype=np.float64)
    histogram[1] = histogram[35] = 4.0
    angle, peak_bin, _, peak_value = parabolic_peak_angle(histogram)
    assert angle == pytest.approx(10.0)
    assert peak_bin == 1
    assert peak_value == 4.0


def test_orientation_assignment_is_deterministic_and_never_returns_minus_one() -> None:
    image = np.tile(np.arange(65, dtype=np.float32), (65, 1))
    keypoints = [SimpleNamespace(x=32.0, y=32.0, size=4.0)]
    config = SimpleNamespace(
        orientation_gradient_kernel=(-1.0, 0.0, 1.0),
        orientation_border_mode="reflect",
        orientation_bins=36,
        orientation_gaussian_sigma_factor=1.5,
        orientation_radius_factor=3.0,
        orientation_histogram_smoothing_passes=6,
    )
    first, diagnostics = assign_orientations_with_diagnostics(image, keypoints, config)
    second, _ = assign_orientations_with_diagnostics(image, keypoints, config)
    assert np.array_equal(first, second)
    assert first[0] == pytest.approx(0.0)
    assert first[0] != -1.0
    assert diagnostics["orientation_180_degree_policy"] == (
        "full_signed_gradient_no_random_flip"
    )


def test_orientation_gradient_border_mode_is_frozen_and_finite() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[:, 0] = 255.0
    gx, gy = signed_gradient_maps(image, border_mode="reflect")
    assert gx.shape == gy.shape == image.shape
    assert np.isfinite(gx).all()
    assert np.isfinite(gy).all()
