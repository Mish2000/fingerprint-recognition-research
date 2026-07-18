from __future__ import annotations

from pathlib import Path

from fingerprint_benchmark.hashing import stable_config_hash
from fingerprint_benchmark.harriszplus.pilot_v3 import (
    AUTHORIZATION_RELATIVE,
    METHOD_VERSION,
    PILOT_NAMESPACE,
    HarrisZPlusV3PublicationAdapter,
    _v3_publication_markers,
    v3_validation_policy,
)
from fingerprint_benchmark.harriszplus.preflight import _effective_runner_config
from fingerprint_benchmark.harriszplus.preflight_v2 import (
    EXPECTED_CANDIDATE_CONFIG_SHA256,
)
from fingerprint_benchmark.harriszplus.provenance import (
    implementation_source_hashes,
)


def test_v3_publication_proxy_does_not_change_algorithm_config() -> None:
    adapter = HarrisZPlusV3PublicationAdapter()
    try:
        metadata = adapter.metadata()
        assert metadata.method_version == METHOD_VERSION
        assert metadata.config["score_producing_adapter_method_version"].endswith(
            "-v1"
        )
        assert metadata.config["candidate_algorithm_config_sha256"] == (
            EXPECTED_CANDIDATE_CONFIG_SHA256
        )
        assert stable_config_hash(_effective_runner_config(metadata)) != (
            EXPECTED_CANDIDATE_CONFIG_SHA256
        )
    finally:
        adapter.close()


def test_required_score_producing_sources_remain_the_frozen_set() -> None:
    assert set(
        implementation_source_hashes(strict=True)[
            "required_score_producing_sources"
        ]
    ) == {
        "adapter.py",
        "config.py",
        "cuda_detector.py",
        "extractor.py",
        "kernels.py",
        "orientation.py",
        "provenance.py",
        "reference_cpu.py",
        "selection.py",
        "types.py",
    }


def test_v3_validation_policy_has_no_spearman_or_v4_gate() -> None:
    policy = v3_validation_policy()
    assert policy["spearman_is_gate"] is False
    assert policy["spearman_minimum_threshold"] is None
    assert policy["v4_relaxation_path_allowed"] is False
    assert policy["auto_relaxation_allowed"] is False


def test_v3_namespace_and_authorization_are_separate_from_v1() -> None:
    assert PILOT_NAMESPACE.endswith("_v3")
    assert "harriszplus_rootsift_geometric_v3" in str(AUTHORIZATION_RELATIVE)


def test_v3_integrity_markers_allow_preflight_but_not_pilot_outputs(
    tmp_path: Path,
) -> None:
    method_root = tmp_path / "method"
    pilot_root = tmp_path / "pilot"
    preflight = method_root / "preflight/pass.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text("pass", encoding="utf-8")
    paths = {"method_root": method_root, "pilot_root": pilot_root}
    assert _v3_publication_markers(paths, exclude_before=True) == []
    config = method_root / "config/freeze_manifest.json"
    config.parent.mkdir()
    config.write_text("freeze", encoding="utf-8")
    run = pilot_root / "runs/sd300b/plain_self/pairs.csv"
    run.parent.mkdir(parents=True)
    run.write_text("rows", encoding="utf-8")
    markers = _v3_publication_markers(paths, exclude_before=True)
    assert str(config.resolve()) in markers
    assert str(run.resolve()) in markers
