from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fingerprint_benchmark import provenance as provenance_module
from fingerprint_benchmark.detectors import (
    OpenCVGFTTHarrisDetector,
    OpenCVGFTTHarrisRootSIFTGeometricAdapter,
    OpenCVHarrisConfig,
)
from fingerprint_benchmark.hashing import stable_hash
from fingerprint_benchmark.local_features.detector_only import (
    DetectorOnlyAdapter,
    DetectorOnlyProtocolConfig,
    build_representation,
)
from fingerprint_benchmark.local_features.orientation import ORIENTATION_POLICY
from fingerprint_benchmark.provenance import implementation_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = PROJECT_ROOT / "src/fingerprint_benchmark/runner.py"
DECLARED_COMPONENT = "adapter_declared_implementation_sources"
REQUIRED_HARRIS_SOURCES = {
    "src/fingerprint_benchmark/detectors/types.py",
    "src/fingerprint_benchmark/detectors/opencv_gftt_harris.py",
    "src/fingerprint_benchmark/local_features/types.py",
    "src/fingerprint_benchmark/local_features/support.py",
    "src/fingerprint_benchmark/local_features/orientation.py",
    "src/fingerprint_benchmark/local_features/detector_only.py",
    "src/fingerprint_benchmark/local_features/descriptors/sift_descriptor.py",
    "src/fingerprint_benchmark/local_features/descriptors/rootsift.py",
    "src/fingerprint_benchmark/local_features/matching.py",
    "src/fingerprint_benchmark/local_features/geometry.py",
    "src/fingerprint_benchmark/local_features/scoring.py",
    "src/fingerprint_benchmark/sift/descriptors.py",
    "src/fingerprint_benchmark/sift/matching.py",
    "src/fingerprint_benchmark/sift/geometry.py",
    "src/fingerprint_benchmark/sift/scoring.py",
}


def _checkerboard(size: int = 128, block: int = 8) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return (((xx // block + yy // block) % 2) * 255).astype(np.uint8)


def _implementation(adapter: DetectorOnlyAdapter):
    return implementation_provenance(
        adapter=adapter,
        method_metadata=adapter.metadata(),
        startup_validation={},
        runner_source_path=RUNNER_SOURCE,
    )


@pytest.mark.parametrize("descriptor", ["sift", "standard"])
def test_public_rootsift_adapter_rejects_non_rootsift_descriptors(descriptor: str) -> None:
    with pytest.raises(ValueError, match="requires descriptor='rootsift'"):
        OpenCVGFTTHarrisRootSIFTGeometricAdapter(
            protocol_config=DetectorOnlyProtocolConfig(descriptor=descriptor)
        )


def test_generic_adapter_requires_nonmisleading_explicit_non_rootsift_identity() -> None:
    detector = OpenCVGFTTHarrisDetector()
    config = DetectorOnlyProtocolConfig(descriptor="sift")
    with pytest.raises(ValueError, match="require explicit method_name and method_version"):
        DetectorOnlyAdapter(detector, config)
    with pytest.raises(ValueError, match="cannot use a RootSIFT method identity"):
        DetectorOnlyAdapter(
            detector,
            config,
            method_name="opencv_gftt_harris_rootsift_geometric",
            method_version="opencv-gftt-harris-rootsift-geometric-v1",
        )


def test_active_descriptor_is_consistent_in_method_and_representation_metadata(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "checkerboard.png"
    assert cv2.imwrite(str(image_path), _checkerboard())
    adapter = DetectorOnlyAdapter(
        OpenCVGFTTHarrisDetector(OpenCVHarrisConfig(max_corners=100, min_distance=3.0)),
        DetectorOnlyProtocolConfig(descriptor="standard"),
        method_name="opencv_gftt_harris_standard_geometric",
        method_version="opencv-gftt-harris-standard-geometric-v1",
    )
    metadata = adapter.metadata()
    prepared = adapter.prepare(image_path, {"ppi": 1000.0})

    assert "rootsift" not in metadata.method
    assert "rootsift" not in metadata.method_version
    assert metadata.config["descriptor"] == "standard"
    assert metadata.implementation_provenance["descriptor"] == "standard"
    assert "standard descriptor processing" in metadata.score_semantics
    assert prepared.representation.metadata["descriptor"] == "standard"
    assert prepared.representation.payload.metadata["descriptor"] == "standard"


def test_harris_provenance_covers_common_and_legacy_implementation_sources() -> None:
    adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
    full, components, implementation_hash = _implementation(adapter)
    component = components[DECLARED_COMPONENT]
    files = component["files"]
    paths = [record["path"] for record in files]

    assert REQUIRED_HARRIS_SOURCES.issubset(paths)
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert all(set(record) == {"path", "sha256"} for record in files)
    assert all(len(record["sha256"]) == 64 for record in files)
    assert component["component_sha256"] == stable_hash(files)
    assert full[DECLARED_COMPONENT] == component
    assert len(implementation_hash) == 64


def test_harris_source_component_and_implementation_hash_are_deterministic() -> None:
    adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
    first_full, first_components, first_hash = _implementation(adapter)
    second_full, second_components, second_hash = _implementation(adapter)

    assert first_components[DECLARED_COMPONENT] == second_components[DECLARED_COMPONENT]
    assert first_full[DECLARED_COMPONENT] == second_full[DECLARED_COMPONENT]
    assert first_hash == second_hash


def test_simulated_source_hash_change_changes_component_and_implementation_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
    _, original_components, original_hash = _implementation(adapter)
    target = (PROJECT_ROOT / "src/fingerprint_benchmark/local_features/support.py").resolve()
    real_file_sha256 = provenance_module.file_sha256

    def simulated_file_sha256(path: Path) -> str:
        return "f" * 64 if Path(path).resolve() == target else real_file_sha256(Path(path))

    monkeypatch.setattr(provenance_module, "file_sha256", simulated_file_sha256)
    _, changed_components, changed_hash = _implementation(adapter)

    assert original_components[DECLARED_COMPONENT] != changed_components[DECLARED_COMPONENT]
    assert original_hash != changed_hash
    changed_record = next(
        record
        for record in changed_components[DECLARED_COMPONENT]["files"]
        if record["path"] == "src/fingerprint_benchmark/local_features/support.py"
    )
    assert changed_record["sha256"] == "f" * 64


@pytest.mark.parametrize(
    "alias",
    ["sift_dominant_gradient", "sift_dominant_gradient_v1", "dominant_gradient_v1"],
)
def test_orientation_aliases_normalize_without_changing_representation(alias: str) -> None:
    image = _checkerboard()
    detector = OpenCVGFTTHarrisDetector(OpenCVHarrisConfig(max_corners=80))
    result = detector.detect(image, {"ppi": 1000.0})
    canonical_config = DetectorOnlyProtocolConfig(orientation_policy=ORIENTATION_POLICY)
    alias_config = DetectorOnlyProtocolConfig(orientation_policy=alias)
    canonical, canonical_diagnostics, _ = build_representation(
        image, {"ppi": 1000.0}, result, canonical_config
    )
    normalized, alias_diagnostics, _ = build_representation(
        image, {"ppi": 1000.0}, result, alias_config
    )

    assert alias_config.orientation_policy == ORIENTATION_POLICY
    assert alias_diagnostics["orientation_policy"] == ORIENTATION_POLICY
    assert normalized.metadata["orientation_policy"] == ORIENTATION_POLICY
    assert canonical_diagnostics["representation_sha256"] == alias_diagnostics[
        "representation_sha256"
    ]
    for field in ("points", "sizes", "angles", "descriptors"):
        assert np.array_equal(getattr(canonical, field), getattr(normalized, field))
