from __future__ import annotations

import math

import numpy as np
import pytest

from fingerprint_benchmark.harriszplus.config import HarrisZPlusConfig
from fingerprint_benchmark.harriszplus.kernels import (
    central_difference_numpy,
    gaussian_kernel1d,
    sample_zscore_numpy,
)
from fingerprint_benchmark.harriszplus.selection import (
    iterative_uniform_selection_with_diagnostics,
    parabolic_offset,
    scale_suppression_distance,
    strict_local_maxima_numpy,
)
from fingerprint_benchmark.harriszplus.types import SelectionCandidate


def test_scale_table_and_support_radii_match_frozen_oracle_mapping() -> None:
    config = HarrisZPlusConfig(backend="reference_cpu")
    assert [config.working_sigma(index) for index in range(5)] == pytest.approx(
        [math.sqrt(2.0), 2.0, math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0)]
    )
    assert [config.working_integration_sigma(index) for index in range(5)] == pytest.approx(
        [2.0, 2.0 * math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0), 4.0]
    )
    assert [config.output_sigma(index) for index in range(5)] == pytest.approx(
        [1.0, 1.0, math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0)]
    )
    assert [scale_suppression_distance(config.working_sigma(index)) for index in range(5)] == [
        5,
        6,
        5,
        6,
        9,
    ]
    table = config.scale_table()
    assert len(table) == 5
    assert [row["scale_index"] for row in table] == list(range(5))
    assert [row["working_image_scale"] for row in table] == [2.0, 2.0, 1.0, 1.0, 1.0]
    assert [row["output_differentiation_sigma"] for row in table] == pytest.approx(
        [1.0, 1.0, math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0)]
    )
    assert [row["effective_support_diameter_original_px"] for row in table] == [
        6.5,
        9.5,
        13.0,
        19.0,
        25.0,
    ]
    frozen = config.as_dict()
    assert frozen["paper_version"] == "arXiv:2109.12925v6"
    assert frozen["detector_dtype"] == "float32"
    assert frozen["raw_edge_mean_multiplier"] == 1.0
    assert frozen["raw_edge_comparison"] == "strict_greater_than"
    assert frozen["response_threshold"] == 0.0
    assert frozen["response_threshold_comparison"] == "strict_greater_than"
    assert frozen["edge_mask_threshold"] == 0.31
    assert frozen["edge_mask_threshold_comparison"] == "strict_greater_than"
    assert frozen["derived_scale_table"] == table


def test_positive_ramp_central_difference_and_zero_frame() -> None:
    image = np.tile(np.arange(7, dtype=np.float32), (7, 1))
    gradient_x, gradient_y = central_difference_numpy(image)
    assert np.array_equal(gradient_x[1:-1, 1:-1], np.full((5, 5), 2.0, np.float32))
    assert not np.any(gradient_y)
    assert not np.any(gradient_x[[0, -1], :])
    assert not np.any(gradient_x[:, [0, -1]])


def test_sampled_gaussian_and_sample_zscore_flat_policy() -> None:
    kernel = gaussian_kernel1d(2.0)
    assert kernel.shape == (13,)
    assert float(np.sum(kernel, dtype=np.float32)) == pytest.approx(1.0, abs=1.0e-7)
    assert np.array_equal(kernel, kernel[::-1])
    assert not np.any(sample_zscore_numpy(np.full((9, 9), 7.0, np.float32)))
    invalid = np.zeros((3, 3), dtype=np.float32)
    invalid[1, 1] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        sample_zscore_numpy(invalid)


def test_local_maximum_near_tie_is_rejected_without_changing_response() -> None:
    response = np.zeros((11, 11), dtype=np.float32)
    response[5, 5] = 5.0
    response[5, 7] = 5.0 - 5.0e-7
    original = response.copy()
    mask = response > 0.0
    maxima = strict_local_maxima_numpy(response, mask, tie_atol=1.0e-6)
    assert not np.any(maxima)
    assert np.array_equal(response, original)

    response[5, 7] = 5.0 - 2.0e-6
    maxima = strict_local_maxima_numpy(response, response > 0.0, tie_atol=1.0e-6)
    assert np.argwhere(maxima).tolist() == [[5, 5]]


def test_uniform_selection_runs_passes_even_below_cap() -> None:
    candidates = (
        SelectionCandidate(0.0, 0.0, 3.0, 0, 0),
        SelectionCandidate(0.1, 0.0, 2.0, 0, 1),
        SelectionCandidate(9.0, 9.0, 1.0, 0, 2),
    )
    selected, diagnostics = iterative_uniform_selection_with_diagnostics(
        candidates,
        10,
        10,
        maximum_keypoints=3000,
    )
    assert [candidate.source_index for candidate in selected] == [0, 2, 1]
    assert diagnostics["passes"] == 2
    assert diagnostics["selected_per_pass"] == [2, 1]


def test_subpixel_parabola_zero_denominator_and_no_clipping() -> None:
    assert parabolic_offset(1.0, 1.0, 1.0) == 0.0
    assert parabolic_offset(3.0, 4.0, 0.0) == pytest.approx(-0.3)
