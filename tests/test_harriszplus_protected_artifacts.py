from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from fingerprint_benchmark.hashing import file_sha256
from fingerprint_benchmark.harriszplus.preflight import (
    HarrisZPlusPreflightError,
    _validate_recorded_artifact_manifest,
    compare_protected_snapshots,
)
from fingerprint_benchmark.harriszplus.provenance import implementation_source_hashes


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These identities come from the already-published, protected pilot artifacts and
# their before/after attestations.  Keeping them literal makes this test an
# independent tripwire rather than merely accepting whatever is on disk today.
EXPECTED_SIFT_REUSED_SOURCE_SHA256 = {
    "config.py": "0dddaf996c10605ec6b08afc3494fefc9c8ff4adb26a5de819ee9049ac7317fb",
    "descriptors.py": "073ed322c327457259d995a1719dbc13e29e682f2b2b06139887d70d72148f78",
    "extractor.py": "90d86105190e0414c198ecce03783f18216ecd7884be688458726d11adc5e4c6",
    "geometry.py": "35b51bd5888c296882a8c80c19df2af4443c1dea9ef91753a6ce5c4a84835982",
    "matching.py": "3c9fd3d82b8c62674bcc52d3992caac07b0f5d0dd6eb67813316f684f15e127d",
}
EXPECTED_SOURCEAFIS_RUNTIME_SHA256 = {
    "src/fingerprint_benchmark/sourceafis_adapter.py": (
        "882e7e57d196e389365bdfded4f6fac29c71480962f60830b41640a45253cf69"
    ),
    "src/fingerprint_benchmark/sourceafis_client.py": (
        "d2cac9b76bb17c58ac20407b9554f378612423c7d0a406f0f8614a3be10363d1"
    ),
    "src/fingerprint_benchmark/sourceafis_negative_protocol.py": (
        "cc3a88e30074a820f7172ee778eb17419926b0272aced7f39300968895fd491c"
    ),
    "src/fingerprint_benchmark/sourceafis_sidecar.py": (
        "fc1e26d33bc8a83bbe5588eab620f6a13966c72a5e89eb8858ea15c7e2bfd782"
    ),
    "apps/sourceafis-sidecar/pom.xml": (
        "201998a8beb924e87798fc942a72073ce78d5b4fd81c2947737c70cb9efbf197"
    ),
    "apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/ApiException.java": (
        "1878aa4c21df5f49bcb6ea46201ece6b8a00e881c7d7edc82c306f5e50110456"
    ),
    "apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/BuildInfo.java": (
        "c8f44eb2e8754852a98b51c67e4f1f22fd6944eceed5143ac95f25bf9d976bb6"
    ),
    "apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/SourceAfisSidecarService.java": (
        "45e5abee5ef70ee0f48b5285999efcdd6bb56a774999b5e5af3ece93e93d9c76"
    ),
    "apps/sourceafis-sidecar/src/main/java/org/fingerprintresearch/sourceafis/v2/SourceAfisV2Engine.java": (
        "288de4f5264c04abedd541ea6571fda3933708d93f15343f5b475aebe937e7c9"
    ),
    "apps/sourceafis-sidecar/src/main/resources/sourceafis-sidecar.properties": (
        "5824f1c4f85b55cc3e90e86c8ab600f69bdc22a1633429e8c2c2ac822f1aa2f6"
    ),
    "apps/sourceafis-sidecar/target/sourceafis-sidecar-0.2.0-shaded.jar": (
        "84df3b736a6e7de1c4493e126433a7ac6aa92c174c24446ecb220ddf71a2712e"
    ),
}


def _snapshot() -> dict[str, object]:
    return {
        "tree_sha256": "before-tree",
        "file_count": 2,
        "files": [
            {"path": "protected/a.bin", "size": 1, "sha256": "a" * 64},
            {"path": "protected/b.bin", "size": 2, "sha256": "b" * 64},
        ],
    }


@pytest.mark.parametrize(
    ("change", "report_field", "changed_path"),
    (
        ("mutated", "changed_paths", "protected/a.bin"),
        ("added", "added_paths", "protected/c.bin"),
        ("removed", "removed_paths", "protected/b.bin"),
    ),
)
def test_artifact_protection_detects_changed_added_and_removed_records(
    change: str,
    report_field: str,
    changed_path: str,
) -> None:
    before = _snapshot()
    assert compare_protected_snapshots(before, deepcopy(before))["passed"] is True

    after = deepcopy(before)
    after["tree_sha256"] = f"{change}-tree"
    files = after["files"]
    assert isinstance(files, list)
    if change == "mutated":
        files[0]["sha256"] = "f" * 64
    elif change == "added":
        files.append({"path": changed_path, "size": 3, "sha256": "c" * 64})
        after["file_count"] = 3
    else:
        files.pop()
        after["file_count"] = 1

    with pytest.raises(HarrisZPlusPreflightError) as error:
        compare_protected_snapshots(before, after)
    assert report_field in str(error.value)
    assert changed_path in str(error.value)


def test_existing_sift_reused_sources_and_protected_pilot_are_byte_exact() -> None:
    assert (
        implementation_source_hashes(strict=True)["reused_sift_unchanged"]
        == EXPECTED_SIFT_REUSED_SOURCE_SHA256
    )

    pilot_root = PROJECT_ROOT / "results/pilots/sift_geometric_joint_500_v1"
    manifest = pilot_root / "artifact_manifest.json"
    assert file_sha256(manifest) == (
        "e7a92d06736929bb6a59e7be5199f7582f3c570da4a352533d09d63d6575d2a8"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["file_count"] == 46
    assert payload["tree_sha256"] == (
        "daf1f0ffe319bc02105b1b7a17652b2cd720a47523e80d98d4d07866fe2a81d3"
    )
    assert len(_validate_recorded_artifact_manifest(manifest)) == 46

    attestation = pilot_root / "integrity/protected_artifact_integrity.json"
    assert file_sha256(attestation) == (
        "48bf5d4046715b42e79784341526e0822e5e82f02bbf387b3a4a5165ee371567"
    )
    integrity = json.loads(attestation.read_text(encoding="utf-8"))
    assert integrity["protected_artifacts_unchanged"] is True
    assert integrity["before"]["tree_sha256"] == integrity["after"]["tree_sha256"] == (
        "860e1e05b50376b3530affc8a7e56c7257905c45325deea9b2a005bd10972834"
    )


def test_existing_sourceafis_runtime_and_protected_pilot_are_byte_exact() -> None:
    actual_runtime_hashes = {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in EXPECTED_SOURCEAFIS_RUNTIME_SHA256
    }
    assert actual_runtime_hashes == EXPECTED_SOURCEAFIS_RUNTIME_SHA256

    pilot_root = PROJECT_ROOT / "results/pilots/sourceafis_joint_500_v1"
    manifest = pilot_root / "artifact_manifest.json"
    assert file_sha256(manifest) == (
        "514ef8343d0eff83ad4b05a868ba4d7256836067232f5e87cf167575b11a3d90"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["file_count"] == 55
    assert payload["tree_sha256"] == (
        "392a67221371dc25ecd98ebcbb680833300a1cf1dac3a6601cfb0a63934d4699"
    )
    assert len(_validate_recorded_artifact_manifest(manifest)) == 55

    artifact_attestation = pilot_root / "integrity/protected_artifact_integrity.json"
    assert file_sha256(artifact_attestation) == (
        "91d054ce5b486078caa464284020c646473708a098687d98ce6c06912d9d4c80"
    )
    artifact_integrity = json.loads(artifact_attestation.read_text(encoding="utf-8"))
    assert artifact_integrity["protected_artifacts_unchanged"] is True
    assert (
        artifact_integrity["before"]["tree_sha256"]
        == artifact_integrity["after"]["tree_sha256"]
        == "a643f4042f2a68e03839fba98cc4a511ae556a081334b82126f4a19e7c575776"
    )

    repository_attestation = pilot_root / "integrity/protected_repository_integrity.json"
    assert file_sha256(repository_attestation) == (
        "06456a95c048a0a099d83d9960c0689c6f575cd8b46dc084e1b284a690215d70"
    )
    repository_integrity = json.loads(
        repository_attestation.read_text(encoding="utf-8")
    )
    assert repository_integrity["protected_artifacts_unchanged"] is True
    assert (
        repository_integrity["before"]["tree_sha256"]
        == repository_integrity["after"]["tree_sha256"]
        == "6eb285a5535864e25020620c539d42331125bdc817c5eb87f05c74d310f9c576"
    )
