from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

try:
    import torch as _torch
except ImportError:  # Optional CUDA dependency is not required by CPU tests.
    _CUDA_AVAILABLE = False
else:
    _CUDA_AVAILABLE = bool(_torch.cuda.is_available())

from fingerprint_benchmark.harriszplus.config import HarrisZPlusConfig
from fingerprint_benchmark.harriszplus.cuda_detector import (
    configure_torch_determinism,
    detect_harriszplus_cuda,
)
from fingerprint_benchmark.harriszplus.kernels import (
    central_difference_numpy,
    gaussian_blur_numpy,
    gaussian_kernel1d,
    sample_zscore_numpy,
)
from fingerprint_benchmark.harriszplus.reference_cpu import (
    compute_harrisz_scale_cpu,
    detect_harriszplus_cpu,
)
from fingerprint_benchmark.harriszplus.selection import (
    candidate_rank_key,
    greedy_distance_suppression,
    iterative_uniform_selection_with_diagnostics,
    parabolic_offset,
    passes_eigen_axis_ratio,
    refine_subpixel_numpy,
    remove_scale_01_duplicates,
    scale_suppression_distance,
    strict_local_maxima_numpy,
    uniform_selection_distance,
)
from fingerprint_benchmark.harriszplus.types import SelectionCandidate


def _double_lanczos(image: np.ndarray) -> np.ndarray:
    source_u8 = np.ascontiguousarray(image, dtype=np.uint8)
    return np.ascontiguousarray(
        cv2.resize(
            source_u8,
            (source_u8.shape[1] * 2, source_u8.shape[0] * 2),
            interpolation=cv2.INTER_LANCZOS4,
        ),
        dtype=np.float32,
    )


def _checkerboard(size: int = 96, block: int = 12) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return ((((xx // block) + (yy // block)) % 2) * 255).astype(np.float32)


def _candidate(
    x: float,
    y: float,
    response: float,
    scale_index: int = 0,
    source_index: int = 0,
) -> SelectionCandidate:
    return SelectionCandidate(x, y, response, scale_index, source_index)


def test_gradient_convention_is_positive_right_and_down_with_zero_frame() -> None:
    horizontal = np.tile(np.arange(7, dtype=np.float32), (6, 1))
    vertical = np.tile(np.arange(6, dtype=np.float32)[:, None], (1, 7))

    dx, dy = central_difference_numpy(horizontal)
    np.testing.assert_array_equal(dx[1:-1, 1:-1], 2.0)
    np.testing.assert_array_equal(dy, 0.0)
    assert not np.any(dx[[0, -1], :]) and not np.any(dx[:, [0, -1]])

    dx, dy = central_difference_numpy(vertical)
    np.testing.assert_array_equal(dx, 0.0)
    np.testing.assert_array_equal(dy[1:-1, 1:-1], 2.0)
    assert not np.any(dy[[0, -1], :]) and not np.any(dy[:, [0, -1]])


def test_gaussian_scale_table_support_and_keypoint_mapping_are_exact() -> None:
    config = HarrisZPlusConfig(backend="reference_cpu")
    root_two = math.sqrt(2.0)
    expected_working_d = (root_two, 2.0, root_two, 2.0, 2.0 * root_two)
    expected_working_i = (2.0, 2.0 * root_two, 2.0, 2.0 * root_two, 4.0)
    expected_output_d = (1.0, 1.0, root_two, 2.0, 2.0 * root_two)
    expected_output_i = (root_two, root_two, 2.0, 2.0 * root_two, 4.0)

    assert tuple(config.scale_indices) == (0, 1, 2, 3, 4)
    assert [config.working_sigma(i) for i in range(5)] == pytest.approx(expected_working_d)
    assert [config.working_integration_sigma(i) for i in range(5)] == pytest.approx(
        expected_working_i
    )
    assert [config.output_sigma(i) for i in range(5)] == pytest.approx(expected_output_d)
    assert [config.output_integration_sigma(i) for i in range(5)] == pytest.approx(
        expected_output_i
    )
    assert [scale_suppression_distance(config.working_sigma(i)) for i in range(5)] == [
        5,
        6,
        5,
        6,
        9,
    ]
    assert [config.keypoint_size(i) for i in range(5)] == pytest.approx(
        [2.0 * value for value in expected_output_i]
    )
    assert config.keypoint_size(0) == config.keypoint_size(1)
    assert all(
        config.keypoint_size(left) < config.keypoint_size(right)
        for left, right in ((1, 2), (2, 3), (3, 4))
    )

    for sigma in expected_working_d + expected_working_i:
        kernel = gaussian_kernel1d(sigma)
        radius = math.ceil(3.0 * sigma)
        assert kernel.dtype == np.float32
        assert kernel.shape == (2 * radius + 1,)
        np.testing.assert_array_equal(kernel, kernel[::-1])
        assert float(np.sum(kernel, dtype=np.float32)) == pytest.approx(1.0, abs=1e-7)


def test_gaussian_padding_is_edge_including_border_reflect() -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    kernel = gaussian_kernel1d(1.0)
    expected = cv2.sepFilter2D(
        image,
        cv2.CV_32F,
        kernel,
        kernel,
        borderType=cv2.BORDER_REFLECT,
    )
    np.testing.assert_array_equal(gaussian_blur_numpy(image, 1.0), expected)


def test_sample_zscore_flat_and_dense_harrisz_equations() -> None:
    flat = np.full((17, 19), 7.0, dtype=np.float32)
    np.testing.assert_array_equal(sample_zscore_numpy(flat), np.zeros_like(flat))

    image = np.zeros((49, 51), dtype=np.float32)
    image[20:, 24:] = 255.0
    config = HarrisZPlusConfig(backend="reference_cpu")
    maps = compute_harrisz_scale_cpu(image, 2, config)
    expected_raw_mask = maps["gradient_magnitude"] > np.mean(
        maps["gradient_magnitude"], dtype=np.float32
    )
    expected_response = sample_zscore_numpy(maps["determinant"]) - sample_zscore_numpy(
        maps["trace_squared"]
    )
    np.testing.assert_array_equal(maps["raw_edge_mask"], expected_raw_mask)
    np.testing.assert_allclose(maps["response"], expected_response, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        maps["candidate_mask"],
        (maps["response"] > 0.0) & (maps["edge_mask"] > 0.31),
    )
    assert np.isfinite(maps["response"]).all()


def test_flat_detector_has_no_candidates_or_nonfinite_values() -> None:
    image = np.full((64, 64), 127.0, dtype=np.float32)
    config = HarrisZPlusConfig(backend="reference_cpu")
    result = detect_harriszplus_cpu(
        image,
        config,
        doubled_image=_double_lanczos(image),
        return_response_maps=True,
    )
    assert result.keypoints == ()
    assert result.diagnostics["final_keypoint_count"] == 0
    assert result.diagnostics["candidates_after_uniform_selection"] == 0
    assert result.diagnostics["invalid_or_nan_outputs"] == 0
    assert all(
        scale["nonfinite_response_count"] == 0
        for scale in result.diagnostics["scales"].values()
    )
    assert result.response_maps is not None
    for response in result.response_maps.values():
        np.testing.assert_array_equal(response, np.zeros_like(response))


def test_strict_unique_local_maxima_rejects_ties_and_allows_partial_border() -> None:
    response = np.zeros((11, 11), dtype=np.float32)
    mask = np.zeros_like(response, dtype=bool)
    response[0, 0] = 9.0
    mask[0, 0] = True
    response[5, 4] = 8.0
    response[5, 6] = np.float32(8.0 - 5.0e-7)
    mask[5, 4] = mask[5, 6] = True
    maxima = strict_local_maxima_numpy(response, mask)
    assert maxima[0, 0]
    assert not maxima[5, 4]
    assert not maxima[5, 6]
    assert int(maxima.sum()) == 1


def test_scale_suppression_subpixel_and_eigen_ratio_boundaries() -> None:
    selected, discarded = greedy_distance_suppression(
        (
            _candidate(0.0, 0.0, 3.0, source_index=0),
            _candidate(4.999, 0.0, 2.0, source_index=1),
            _candidate(5.0, 0.0, 1.0, source_index=2),
        ),
        5.0,
    )
    assert [item.source_index for item in selected] == [0, 2]
    assert [item.source_index for item in discarded] == [1]

    response = np.zeros((5, 5), dtype=np.float32)
    # f(x) = -(x-2.25)^2 has its vertex at +0.25 from integer x=2.
    for x in range(5):
        response[2, x] = -float(x - 2.25) ** 2
    refined_x, refined_y = refine_subpixel_numpy(response, 2, 2)
    assert refined_x == pytest.approx(2.25)
    assert refined_y == 2.0
    assert refine_subpixel_numpy(response, 0, 2)[0] == 0.0
    assert parabolic_offset(0.0, 0.0, 1.0) == pytest.approx(-0.5)

    assert not passes_eigen_axis_ratio(1.0, 0.0, 16.0, threshold=0.25)
    assert passes_eigen_axis_ratio(1.0, 0.0, 15.0, threshold=0.25)
    assert not passes_eigen_axis_ratio(0.0, 0.0, 1.0, threshold=0.25)


def test_duplicate_removal_ranking_uniform_reselection_and_cap() -> None:
    duplicates = (
        _candidate(0.0, 0.0, 5.0, scale_index=0, source_index=0),
        _candidate(0.999, 0.0, 4.0, scale_index=1, source_index=1),
        _candidate(1.0, 0.0, 3.0, scale_index=1, source_index=2),
        _candidate(0.1, 0.1, 2.0, scale_index=2, source_index=3),
    )
    retained = remove_scale_01_duplicates(duplicates, distance=1.0)
    assert [item.source_index for item in retained] == [0, 2, 3]

    ties = (
        _candidate(4.0, 3.0, 1.0, 1, 9),
        _candidate(2.0, 3.0, 1.0, 2, 8),
        _candidate(1.0, 2.0, 1.0, 2, 7),
        _candidate(1.0, 2.0, 1.0, 2, 6),
    )
    assert [item.source_index for item in sorted(ties, key=candidate_rank_key)] == [6, 7, 8, 9]

    assert uniform_selection_distance(10, 10, 4) == pytest.approx(
        math.sqrt(8.0 * 10.0 * 10.0 / (math.pi * 4.0))
    )
    pass_local, diagnostics = iterative_uniform_selection_with_diagnostics(
        (
            _candidate(0.0, 0.0, 3.0, source_index=0),
            _candidate(1.0, 0.0, 2.0, source_index=1),
            _candidate(9.0, 0.0, 1.0, source_index=2),
        ),
        10,
        10,
        maximum_keypoints=4,
    )
    assert [item.source_index for item in pass_local] == [0, 2, 1]
    assert diagnostics["selected_per_pass"] == [2, 1]

    many = tuple(
        _candidate(float(index * 2), 0.0, float(3001 - index), source_index=index)
        for index in range(3001)
    )
    capped, cap_diagnostics = iterative_uniform_selection_with_diagnostics(
        many,
        1,
        3001,
        maximum_keypoints=3000,
    )
    assert len(capped) == 3000
    assert cap_diagnostics["cap_truncated_count"] == 1


def test_cpu_detector_is_byte_stable_and_preserves_all_stage_counts() -> None:
    image = _checkerboard()
    doubled = _double_lanczos(image)
    config = HarrisZPlusConfig(backend="reference_cpu", max_keypoints=256)
    first = detect_harriszplus_cpu(
        image,
        config,
        doubled_image=doubled,
        return_response_maps=True,
    )
    second = detect_harriszplus_cpu(
        image,
        config,
        doubled_image=doubled,
        return_response_maps=True,
    )
    assert first.keypoints == second.keypoints
    assert first.keypoints == tuple(sorted(first.keypoints, key=candidate_rank_key))
    assert first.count <= 256
    assert first.response_maps is not None and second.response_maps is not None
    for scale_index in range(5):
        np.testing.assert_array_equal(
            first.response_maps[scale_index], second.response_maps[scale_index]
        )
    for key in (
        "candidates_before_mask",
        "candidates_after_mask",
        "candidates_after_local_maxima",
        "candidates_after_scale_suppression",
        "candidates_after_duplicate_removal",
        "candidates_after_uniform_selection",
        "final_keypoint_count",
        "cap_truncated_count",
    ):
        assert isinstance(first.diagnostics[key], int)


def test_larger_synthetic_corner_scale_never_reduces_dominant_keypoint_size() -> None:
    base = np.zeros((256, 256), dtype=np.float32)
    base[128:, 128:] = 255.0
    config = HarrisZPlusConfig(backend="reference_cpu")
    dominant_sizes: list[float] = []
    for blur_sigma in (0.5, 3.0, 6.0):
        blurred = cv2.GaussianBlur(
            base,
            (0, 0),
            blur_sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        image_u8 = np.clip(np.rint(blurred), 0.0, 255.0).astype(np.uint8)
        image = image_u8.astype(np.float32)
        result = detect_harriszplus_cpu(
            image,
            config,
            doubled_image=_double_lanczos(image),
        )
        near_corner = [
            point
            for point in result.keypoints
            if (point.x - 128.0) ** 2 + (point.y - 128.0) ** 2 < 30.0**2
        ]
        assert near_corner
        dominant = max(near_corner, key=lambda point: point.response)
        dominant_sizes.append(dominant.size)

    assert dominant_sizes == sorted(dominant_sizes)
    assert dominant_sizes[-1] > dominant_sizes[0]


@pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="CUDA is required for HarrisZ+ CUDA validation",
)
def test_cuda_is_exact_across_repeats_and_close_to_cpu() -> None:
    import torch

    image = _checkerboard(size=72, block=9)
    doubled = _double_lanczos(image)
    cpu_config = HarrisZPlusConfig(backend="reference_cpu", max_keypoints=128)
    cuda_config = cpu_config.changed(backend="cuda")
    cpu = detect_harriszplus_cpu(
        image,
        cpu_config,
        doubled_image=doubled,
        return_response_maps=True,
    )
    first = detect_harriszplus_cuda(
        image,
        cuda_config,
        doubled_image=doubled,
        return_response_maps=True,
    )
    second = detect_harriszplus_cuda(
        image,
        cuda_config,
        doubled_image=doubled,
        return_response_maps=True,
    )

    assert first.keypoints == second.keypoints
    assert first.response_maps is not None and second.response_maps is not None
    assert cpu.response_maps is not None
    for scale_index in range(5):
        np.testing.assert_array_equal(first.response_maps[scale_index], second.response_maps[scale_index])
        np.testing.assert_allclose(
            first.response_maps[scale_index],
            cpu.response_maps[scale_index],
            atol=cuda_config.validation_response_atol,
            rtol=cuda_config.validation_response_rtol,
        )

    policy = configure_torch_determinism(cuda_config.rng_seed)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert policy["autocast"] is False
    assert policy["dtype"] == "float32"
    assert first.diagnostics["deterministic_policy"]["autocast"] is False
    assert (
        first.diagnostics["detector_gpu_kernel_timing_source"]
        == "summed_cuda_event_elapsed_time"
    )
    assert first.diagnostics["invalid_or_nan_outputs"] == 0
    assert first.diagnostics["final_keypoint_count"] <= 128
    for key in ("detector_gpu_kernel_ms", "candidate_transfer_ms", "selection_cpu_ms"):
        assert key in first.timings
        assert float(first.timings[key]) >= 0.0
