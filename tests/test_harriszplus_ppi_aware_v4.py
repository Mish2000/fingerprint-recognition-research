from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from fingerprint_benchmark.harriszplus.detector_v4 import (
    detect_harriszplus_v4_cpu,
    detect_harriszplus_v4_cuda,
)
from fingerprint_benchmark.harriszplus.extractor_v4 import (
    _cuda_measurement_finish,
    _cuda_measurement_start,
)
from fingerprint_benchmark.harriszplus.pilot_v4 import run_complete_workflow
from fingerprint_benchmark.harriszplus.ppi_aware_v4 import (
    DECISION_THRESHOLD,
    REFERENCE_PPI,
    PpiAwareHarrisZPlusConfig,
    build_physical_scale_contract,
)
from fingerprint_benchmark.harriszplus.selection import (
    uniform_selection_distance,
)
from fingerprint_benchmark.harriszplus.v4_integrity import (
    compare_inventories,
    protected_v1_v3_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> PpiAwareHarrisZPlusConfig:
    return PpiAwareHarrisZPlusConfig()


def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((192, 192))
    source = (((xx // 16 + yy // 16) % 2) * 255).astype(np.uint8)
    doubled = cv2.resize(
        source, (384, 384), interpolation=cv2.INTER_LANCZOS4
    )
    return source.astype(np.float32), doubled.astype(np.float32)


def test_01_reference_ppi_is_1000() -> None:
    assert REFERENCE_PPI == 1000.0
    assert _config().reference_ppi == 1000.0


def test_02_manifest_ppi_requires_finite_positive_value() -> None:
    with pytest.raises(ValueError):
        _config().runtime(float("nan"))
    with pytest.raises(ValueError):
        _config().runtime(0)


def test_03_b_scale_factor_is_one() -> None:
    assert _config().runtime(1000).spatial_scale == 1.0


def test_04_c_scale_factor_is_two() -> None:
    assert _config().runtime(2000).spatial_scale == 2.0


def test_05_sigma_scales_once() -> None:
    config = _config()
    for index in config.scale_indices:
        assert config.runtime(2000).output_sigma(index) == pytest.approx(
            2.0 * config.runtime(1000).output_sigma(index)
        )
        assert config.runtime(2000).output_integration_sigma(
            index
        ) == pytest.approx(
            2.0 * config.runtime(1000).output_integration_sigma(index)
        )


def test_06_kernel_support_radius_scales_once() -> None:
    config = _config()
    for index in config.scale_indices:
        for kind in ("differentiation", "integration"):
            assert config.runtime(2000).kernel_radius(
                index, kind
            ) == 2 * config.runtime(1000).kernel_radius(index, kind)


def test_07_suppression_scales_once() -> None:
    config = _config()
    for index in config.scale_indices:
        assert config.runtime(2000).scale_suppression_distance_working(
            index
        ) == pytest.approx(
            2.0
            * config.runtime(1000).scale_suppression_distance_working(index)
        )


def test_08_duplicate_radius_scales_once() -> None:
    config = _config()
    assert config.runtime(1000).duplicate_distance == 1.0
    assert config.runtime(2000).duplicate_distance == 2.0


def test_09_uniform_q_is_not_double_scaled() -> None:
    b = uniform_selection_distance(941, 622, 3000)
    c = uniform_selection_distance(1883, 1244, 3000)
    assert c / b == pytest.approx(2.0, rel=0.001)
    assert _config().uniform_q_policy.startswith("dimension_derived")


def test_10_keypoint_size_scales_once() -> None:
    config = _config()
    for index in config.scale_indices:
        assert config.runtime(2000).keypoint_size(index) == pytest.approx(
            2.0 * config.runtime(1000).keypoint_size(index)
        )


def test_11_orientation_radius_scales_once() -> None:
    config = _config()
    for index in config.scale_indices:
        assert config.runtime(2000).orientation_radius_pixels(
            index
        ) == 2 * config.runtime(1000).orientation_radius_pixels(index)


def test_12_descriptor_support_is_physically_equal() -> None:
    config = _config()
    for index in config.scale_indices:
        b = (
            config.runtime(1000).descriptor_support_diameter_estimate(index)
            * 25.4
            / 1000
        )
        c = (
            config.runtime(2000).descriptor_support_diameter_estimate(index)
            * 25.4
            / 2000
        )
        assert c == pytest.approx(b, abs=1e-12)


def test_13_border_margin_scales_once() -> None:
    config = _config()
    for index in config.scale_indices:
        assert config.runtime(2000).border_margin_native(
            index
        ) == 2 * config.runtime(1000).border_margin_native(index)


def test_14_internal_doubling_coordinate_mapping() -> None:
    config = _config()
    for ppi in (1000, 2000):
        runtime = config.runtime(ppi)
        assert runtime.working_image_scale(0) == 2.0
        assert runtime.working_image_scale(1) == 2.0
        assert runtime.working_image_scale(2) == 1.0
        assert runtime.working_sigma(0) / 2.0 == pytest.approx(
            runtime.nominal_sigma(0)
        )
        assert runtime.output_sigma(0) == runtime.output_sigma(1)


def test_15_no_double_ppi_scaling() -> None:
    config = _config()
    runtime = config.runtime(2000)
    assert runtime.output_sigma(4) == pytest.approx(
        2.0 * config.reference.output_sigma(4)
    )
    assert runtime.output_sigma(4) != pytest.approx(
        4.0 * config.reference.output_sigma(4)
    )


def test_16_physical_mm_contract_passes() -> None:
    contract = build_physical_scale_contract(_config())
    assert contract["passed"] is True
    assert all(row["passed"] for row in contract["comparisons"])
    assert contract["uniform_q"]["passed"] is True


def test_17_ransac_is_unchanged() -> None:
    config = _config()
    contract = build_physical_scale_contract(config)
    assert config.ransac_threshold_at_reference_ppi == 3.0
    assert config.normalize_coordinates_by_ppi is True
    assert contract["ransac"]["sd300b_native_threshold_px"] == 3.0
    assert contract["ransac"]["sd300c_native_threshold_px"] == 6.0
    assert contract["ransac"]["passed"] is True


def test_18_threshold_is_unchanged() -> None:
    assert DECISION_THRESHOLD == 4


def test_19_keypoint_cap_is_unchanged() -> None:
    config = _config()
    assert config.max_keypoints == config.hard_max_keypoints == 3000
    with pytest.raises(ValueError):
        config.changed(max_keypoints=2999)


def test_20_cpu_cuda_equivalence_with_scaled_kernels() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    image, doubled = _synthetic()
    config = _config()
    cpu = detect_harriszplus_v4_cpu(
        image,
        config.changed(backend="reference_cpu", device=None).runtime(2000),
        doubled_image=doubled,
        return_response_maps=True,
    )
    cuda = detect_harriszplus_v4_cuda(
        image,
        config.changed(backend="cuda", device="cuda:0").runtime(2000),
        doubled_image=doubled,
        device="cuda:0",
        return_response_maps=True,
    )
    for index in config.scale_indices:
        assert np.isclose(
            cpu.response_maps[index],
            cuda.response_maps[index],
            atol=5e-4,
            rtol=2e-4,
        ).mean() >= 0.9999
    assert abs(len(cpu.keypoints) - len(cuda.keypoints)) <= max(
        2, math.ceil(0.005 * max(len(cpu.keypoints), len(cuda.keypoints)))
    )


def test_21_cuda_repeat_is_exact() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    image, doubled = _synthetic()
    runtime = _config().changed(
        backend="cuda", device="cuda:0"
    ).runtime(1000)
    first = detect_harriszplus_v4_cuda(
        image, runtime, doubled_image=doubled, device="cuda:0"
    )
    repeat = detect_harriszplus_v4_cuda(
        image, runtime, doubled_image=doubled, device="cuda:0"
    )
    first_rows = [
        (
            point.x,
            point.y,
            point.response,
            point.scale_index,
            point.size,
            point.source_index,
        )
        for point in first.keypoints
    ]
    repeat_rows = [
        (
            point.x,
            point.y,
            point.response,
            point.scale_index,
            point.size,
            point.source_index,
        )
        for point in repeat.keypoints
    ]
    assert first_rows == repeat_rows


def test_22_vram_measurement_reports_separate_peaks() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = _config().changed(backend="cuda", device="cuda:0").runtime(1000)
    runtime = _cuda_measurement_start(config)
    allocation = torch.zeros(1024, device="cuda:0")
    measurement = _cuda_measurement_finish(runtime)
    del allocation
    assert measurement["peak_vram_allocated"] >= 4096
    assert measurement["peak_vram_reserved"] >= measurement[
        "peak_vram_allocated"
    ]
    assert "sum" not in measurement


def test_23_vram_peaks_do_not_exceed_physical_memory() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    runtime = _cuda_measurement_start(
        _config().changed(backend="cuda", device="cuda:0").runtime(1000)
    )
    measurement = _cuda_measurement_finish(runtime)
    assert measurement["vram_measurement_valid"] is True
    assert measurement["peak_vram_allocated"] <= measurement[
        "vram_physical_total_bytes"
    ]
    assert measurement["peak_vram_reserved"] <= measurement[
        "vram_physical_total_bytes"
    ]


def test_24_workflow_freezes_before_500_run() -> None:
    source = inspect.getsource(run_complete_workflow)
    assert source.index("freeze_after_preflight") < source.index("run_pilot")


def test_25_v1_v3_inventory_is_immutable() -> None:
    before_path = (
        PROJECT_ROOT
        / "results/harriszplus_rootsift_geometric_ppi_aware_v4/"
        "integrity/v1_v3_before.json"
    )
    before = json.loads(before_path.read_text(encoding="utf-8"))
    comparison = compare_inventories(
        before, protected_v1_v3_inventory(PROJECT_ROOT)
    )
    assert comparison["byte_identical"] is True


def test_26_no_tuning_from_500_results() -> None:
    config = _config().as_dict()
    assert config["reference_v3_config"]["response_threshold"] == 0.0
    assert config["reference_v3_config"]["edge_mask_threshold"] == 0.31
    assert config["reference_v3_config"]["lowe_ratio"] == 0.75
    assert config["reference_v3_config"]["max_keypoints"] == 3000
    rationale = (
        PROJECT_ROOT / "docs/harriszplus_v4_ppi_aware_rationale.md"
    ).read_text(encoding="utf-8")
    assert "not used for parameter choice" in rationale
