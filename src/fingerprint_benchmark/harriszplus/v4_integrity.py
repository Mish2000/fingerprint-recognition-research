"""Stable inventories proving that all pre-v4 HarrisZ+ assets remain unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTECTED_RESULT_DIRECTORIES = (
    "results/harriszplus_rootsift_geometric",
    "results/harriszplus_rootsift_geometric_v2",
    "results/harriszplus_rootsift_geometric_v3",
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v1",
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v3",
)
V4_SOURCE_NAMES = {
    "v4_integrity.py",
    "ppi_aware_v4.py",
    "detector_v4.py",
    "orientation_v4.py",
    "extractor_v4.py",
    "adapter_v4.py",
    "preflight_v4.py",
    "pilot_v4.py",
    "reporting_v4.py",
}
V4_TEST_NAMES = {
    "test_harriszplus_ppi_aware_v4.py",
    "test_harriszplus_preflight_v4.py",
    "test_harriszplus_pilot_v4.py",
}
V4_DOC_NAMES = {
    "harriszplus_v4_ppi_aware_rationale.md",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_v1_v3_inventory(project_root: Path) -> dict[str, Any]:
    """Hash the exact stable v1-v3 source/test/doc/result scope."""

    root = project_root.resolve()
    paths: set[Path] = set()
    source_root = root / "src/fingerprint_benchmark/harriszplus"
    paths.update(
        path
        for path in source_root.rglob("*.py")
        if path.name not in V4_SOURCE_NAMES
    )
    paths.update(
        path
        for path in (root / "tests").glob("test_harriszplus*.py")
        if path.name not in V4_TEST_NAMES
    )
    paths.update(
        path
        for path in (root / "docs").glob("harriszplus*.md")
        if path.name not in V4_DOC_NAMES
    )
    for relative in PROTECTED_RESULT_DIRECTORIES:
        directory = root / relative
        if directory.exists():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    records = [
        {
            "path": path.resolve().relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(paths, key=lambda item: str(item).lower())
    ]
    lines = "".join(
        f"{record['path']}\t{record['size']}\t{record['sha256']}\n"
        for record in sorted(records, key=lambda item: item["path"])
    ).encode("utf-8")
    return {
        "schema_version": "harriszplus-v1-v3-protected-inventory-v4",
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(lines).hexdigest(),
        "files": records,
    }


def compare_inventories(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_files = {record["path"]: record for record in before["files"]}
    after_files = {record["path"]: record for record in after["files"]}
    missing = sorted(set(before_files) - set(after_files))
    added = sorted(set(after_files) - set(before_files))
    changed = sorted(
        path
        for path in set(before_files) & set(after_files)
        if before_files[path] != after_files[path]
    )
    byte_identical = not missing and not added and not changed
    return {
        "schema_version": "harriszplus-v1-v3-integrity-comparison-v4",
        "byte_identical": byte_identical,
        "before_file_count": before["file_count"],
        "after_file_count": after["file_count"],
        "before_tree_sha256": before["tree_sha256"],
        "after_tree_sha256": after["tree_sha256"],
        "missing": missing,
        "added": added,
        "changed": changed,
    }


def write_inventory(path: Path, inventory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "compare_inventories",
    "file_sha256",
    "protected_v1_v3_inventory",
    "write_inventory",
]
