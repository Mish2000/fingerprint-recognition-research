from __future__ import annotations

import numpy as np
import pytest

from fingerprint_benchmark.harriszplus.config import HarrisZPlusConfig
from fingerprint_benchmark.harriszplus.cuda_detector import (
    _WorkingCandidate as TorchWorkingCandidate,
)
from fingerprint_benchmark.harriszplus.cuda_detector import _refine_and_filter_torch
from fingerprint_benchmark.harriszplus.reference_cpu import (
    _WorkingCandidate as CpuWorkingCandidate,
)
from fingerprint_benchmark.harriszplus.reference_cpu import _refine_and_filter_cpu


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the CUDA path")
def test_doubled_scales_map_refined_working_coordinates_back_by_exactly_one_half() -> None:
    response = np.zeros((9, 9), dtype=np.float32)
    integer_x, integer_y = 4, 6
    response[integer_y, integer_x] = np.float32(9.875)
    response[integer_y, integer_x - 1] = np.float32(6.875)
    response[integer_y, integer_x + 1] = np.float32(8.875)
    response[integer_y - 1, integer_x] = np.float32(8.875)
    response[integer_y + 1, integer_x] = np.float32(6.875)
    positive_definite = np.full(response.shape, 4.0, dtype=np.float32)
    zero = np.zeros(response.shape, dtype=np.float32)
    cpu_maps = {
        "response": response,
        "autocorrelation_xx": positive_definite,
        "autocorrelation_xy": zero,
        "autocorrelation_yy": positive_definite,
    }
    cuda_maps = {
        name: torch.as_tensor(array, device="cuda") for name, array in cpu_maps.items()
    }
    refined_working_x = 4.25
    refined_working_y = 5.75

    for scale_index in (0, 1):
        cpu_config = HarrisZPlusConfig(backend="reference_cpu")
        cuda_config = HarrisZPlusConfig(backend="cuda")
        assert cpu_config.working_image_scale(scale_index) == 2.0
        assert cuda_config.working_image_scale(scale_index) == 2.0

        cpu_candidate = CpuWorkingCandidate(
            x=float(integer_x),
            y=float(integer_y),
            response=float(response[integer_y, integer_x]),
            scale_index=scale_index,
            source_index=scale_index,
            integer_x=integer_x,
            integer_y=integer_y,
        )
        cuda_candidate = TorchWorkingCandidate(
            x=float(integer_x),
            y=float(integer_y),
            response=float(response[integer_y, integer_x]),
            scale_index=scale_index,
            source_index=scale_index,
            integer_x=integer_x,
            integer_y=integer_y,
        )

        cpu_output = _refine_and_filter_cpu(
            (cpu_candidate,), cpu_maps, cpu_config, scale_index
        )
        cuda_output = _refine_and_filter_torch(
            (cuda_candidate,), cuda_maps, cuda_config, scale_index
        )

        assert len(cpu_output) == len(cuda_output) == 1
        for output in (cpu_output[0], cuda_output[0]):
            assert output.x == refined_working_x * 0.5
            assert output.y == refined_working_y * 0.5
            assert output.x * 2.0 == refined_working_x
            assert output.y * 2.0 == refined_working_y
            assert output.scale_index == scale_index
