"""Read-only input inventories used to prove that protected inputs did not change."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from fingerprint_benchmark.hashing import canonical_json_bytes, file_sha256
from fingerprint_benchmark.manifest import read_pair_manifest


def protected_input_inventory(repo_root: Path) -> dict[str, Any]:
    manifests = sorted((repo_root / "protocols").rglob("*.csv"))
    dataset_paths = sorted(
        {
            path.resolve()
            for manifest in manifests
            for pair in read_pair_manifest(manifest)
            for path in (pair.path_a, pair.path_b)
        },
        key=lambda path: str(path).lower(),
    )
    sourceafis_paths = sorted(
        _protected_sourceafis_paths(repo_root), key=lambda path: str(path).lower()
    )
    manifest_hashes = {str(path.resolve()): file_sha256(path) for path in manifests}
    sourceafis_hashes = {str(path.resolve()): file_sha256(path) for path in sourceafis_paths}
    dataset_stats = [_stat_record(path) for path in dataset_paths]
    return {
        "inventory_schema": "sift-protected-input-inventory-v1",
        "protocol_manifests": manifest_hashes,
        "protocol_manifest_inventory_sha256": _hash_records(manifest_hashes),
        "sourceafis_file_count": len(sourceafis_hashes),
        "sourceafis_files": sourceafis_hashes,
        "sourceafis_inventory_sha256": _hash_records(sourceafis_hashes),
        "dataset_file_count": len(dataset_stats),
        "dataset_stat_inventory_sha256": _hash_records(dataset_stats),
        "dataset_integrity_policy": (
            "all manifest-referenced paths, sizes, and nanosecond modification times; selected pilot images also use content SHA-256"
        ),
    }


def compare_inventories(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol_manifest_inventory_sha256",
        "sourceafis_inventory_sha256",
        "dataset_stat_inventory_sha256",
    )
    matches = {key: before.get(key) == after.get(key) for key in keys}
    return {
        "status": "ok" if all(matches.values()) else "changed",
        "matches": matches,
        "before": {key: before.get(key) for key in keys},
        "after": {key: after.get(key) for key in keys},
    }


def _protected_sourceafis_paths(repo_root: Path) -> Iterable[Path]:
    roots = (
        repo_root / "src" / "fingerprint_benchmark",
        repo_root / "apps" / "sourceafis-sidecar",
        repo_root / "results",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lowered = str(path).lower()
            if "sourceafis" in lowered and "sift_geometric" not in lowered:
                yield path


def _stat_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _hash_records(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
