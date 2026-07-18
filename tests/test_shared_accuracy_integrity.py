from __future__ import annotations

import json
from pathlib import Path

import pytest

from fingerprint_benchmark.shared_accuracy_integrity import (
    FROZEN_BENCHMARK_FILES,
    FROZEN_SIFT_FILES,
    FROZEN_SOURCEAFIS_FILES,
    SharedAccuracyIntegrityError,
    build_artifact_manifest,
    capture_protected_before,
    capture_snapshot,
    compare_snapshot,
    enumerate_protected_paths,
    read_snapshot,
    require_protected_unchanged,
    verify_protected_after,
    write_artifact_manifest,
)


def test_protected_inventory_covers_required_inputs_and_excludes_shared_accuracy(
    tmp_path: Path,
) -> None:
    project_root, data_root = _milestone_layout(tmp_path)

    paths = enumerate_protected_paths(project_root, data_root)
    indexed = {path.relative_to(tmp_path).as_posix(): category for path, category in paths.items()}

    assert indexed["data/NIST/sd300b/images/b.wsq"] == "dataset_sd300b"
    assert indexed["data/NIST/sd300c/images/c.wsq"] == "dataset_sd300c"
    assert indexed["project/protocols/sd300b/plain_roll.csv"] == "base_protocols"
    assert indexed["project/results/legacy/result.json"] == "preexisting_results"
    assert indexed["project/apps/sourceafis-sidecar/target/sidecar.jar"] == "sourceafis_app"
    assert (
        indexed["project/src/fingerprint_benchmark/runner.py"]
        == "benchmark_implementation"
    )
    assert (
        indexed["project/src/fingerprint_benchmark/sourceafis_adapter.py"]
        == "sourceafis_python_implementation"
    )
    assert (
        indexed["project/src/fingerprint_benchmark/sift/adapter.py"]
        == "sift_implementation"
    )
    assert not any("project/results/shared_accuracy/" in path for path in indexed)
    assert all(path.is_absolute() for path in paths)


def test_snapshot_is_deterministic_timestamp_free_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "inputs"
    first = _write(source / "a.txt", "alpha")
    second = _write(source / "b.txt", "beta")
    snapshot = tmp_path / "output" / "before.jsonl"
    inventory = {second: "second", first: "first"}

    initial = capture_snapshot(inventory, snapshot)
    initial_bytes = snapshot.read_bytes()
    repeated = capture_snapshot(dict(reversed(list(inventory.items()))), snapshot)

    assert initial == repeated
    assert snapshot.read_bytes() == initial_bytes
    records, footer = read_snapshot(snapshot)
    assert [record["path"] for record in records] == sorted(
        (first.resolve().as_posix(), second.resolve().as_posix()), key=str.casefold
    )
    assert all(set(record) == {"category", "path", "size", "sha256"} for record in records)
    assert not _contains_timestamp_key(records)
    assert not _contains_timestamp_key(footer)

    first.write_text("changed", encoding="utf-8")
    with pytest.raises(SharedAccuracyIntegrityError, match="different content"):
        capture_snapshot(inventory, snapshot)


def test_snapshot_comparison_reports_added_removed_changed_and_category_mismatches(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path / "inputs" / "a.txt", "alpha")
    second = _write(tmp_path / "inputs" / "b.txt", "beta")
    third = _write(tmp_path / "inputs" / "c.txt", "gamma")
    snapshot = tmp_path / "before.jsonl"
    capture_snapshot({first: "one", second: "two", third: "three"}, snapshot)

    first.write_text("ALPHA", encoding="utf-8")
    second.unlink()
    fourth = _write(tmp_path / "inputs" / "d.txt", "delta")
    report, _, _ = compare_snapshot(
        snapshot,
        {first: "one", third: "changed-category", fourth: "four"},
    )

    assert report["protected_artifacts_unchanged"] is False
    assert report["added_paths"] == [fourth.resolve().as_posix()]
    assert report["removed_paths"] == [second.resolve().as_posix()]
    assert report["changed_file_count"] == 1
    assert report["changed_files"][0]["path"] == first.resolve().as_posix()
    assert report["category_mismatch_count"] == 1
    assert report["category_mismatches"][0]["path"] == third.resolve().as_posix()
    assert report["mismatch_count"] == 4
    with pytest.raises(SharedAccuracyIntegrityError, match="mismatch_count=4"):
        require_protected_unchanged(report)


def test_high_level_after_verification_ignores_new_shared_output(tmp_path: Path) -> None:
    project_root, data_root = _milestone_layout(tmp_path)
    shared_root = project_root / "results" / "shared_accuracy"
    before_path = shared_root / "sourceafis_sift_v1" / "integrity" / "before.jsonl"
    after_path = shared_root / "sourceafis_sift_v1" / "integrity" / "after.jsonl"
    report_path = shared_root / "sourceafis_sift_v1" / "integrity" / "report.json"

    capture_protected_before(project_root, data_root, before_path)
    _write(shared_root / "sourceafis_sift_v1" / "scores" / "new.csv", "score\n")
    report = verify_protected_after(
        project_root,
        data_root,
        before_path,
        after_snapshot_path=after_path,
        report_path=report_path,
        raise_on_mismatch=True,
    )

    assert report["protected_artifacts_unchanged"] is True
    assert report["mismatch_count"] == 0
    assert after_path.is_file()
    assert report_path.is_file()
    assert "created_at" not in report_path.read_text(encoding="utf-8")

    # Exact reruns are explicitly permitted.
    repeated = verify_protected_after(
        project_root,
        data_root,
        before_path,
        after_snapshot_path=after_path,
        report_path=report_path,
        raise_on_mismatch=True,
    )
    assert repeated["protected_artifacts_unchanged"] is True


def test_artifact_manifest_is_deterministic_excludes_itself_and_refuses_change(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "sourceafis_sift_v1"
    _write(output_root / "scores" / "a.csv", "a\n")
    _write(output_root / "reports" / "summary.json", "{}\n")

    first = write_artifact_manifest(output_root)
    manifest_path = output_root / "artifact_manifest.json"
    first_bytes = manifest_path.read_bytes()
    second = write_artifact_manifest(output_root)

    assert first == second == build_artifact_manifest(output_root)
    assert manifest_path.read_bytes() == first_bytes
    assert first["file_count"] == 2
    assert [record["path"] for record in first["files"]] == [
        "reports/summary.json",
        "scores/a.csv",
    ]
    assert "artifact_manifest.json" not in {record["path"] for record in first["files"]}
    assert not _contains_timestamp_key(first)

    _write(output_root / "reports" / "new.json", "{}\n")
    with pytest.raises(SharedAccuracyIntegrityError, match="different content"):
        write_artifact_manifest(output_root)


def test_artifact_manifest_must_live_inside_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write(output_root / "file.txt", "data")

    with pytest.raises(SharedAccuracyIntegrityError, match="inside the output root"):
        build_artifact_manifest(output_root, manifest_path=tmp_path / "outside.json")


def _milestone_layout(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    data_root = tmp_path / "data"
    _write(data_root / "NIST" / "sd300b" / "images" / "b.wsq", "b")
    _write(data_root / "NIST" / "sd300c" / "images" / "c.wsq", "c")
    _write(project_root / "protocols" / "sd300b" / "plain_roll.csv", "pair_id\n")
    _write(project_root / "results" / "legacy" / "result.json", "{}\n")
    _write(
        project_root / "results" / "shared_accuracy" / "old-candidate" / "partial.csv",
        "excluded\n",
    )
    _write(project_root / "apps" / "sourceafis-sidecar" / "target" / "sidecar.jar", "jar")

    benchmark_root = project_root / "src" / "fingerprint_benchmark"
    for filename in FROZEN_BENCHMARK_FILES:
        _write(benchmark_root / filename, f"# {filename}\n")
    for filename in FROZEN_SOURCEAFIS_FILES:
        _write(benchmark_root / filename, f"# {filename}\n")
    for filename in FROZEN_SIFT_FILES:
        _write(benchmark_root / "sift" / filename, f"# {filename}\n")
    return project_root, data_root


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _contains_timestamp_key(payload: object) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if "time" in lowered or "date" in lowered or "created" in lowered:
                return True
            if _contains_timestamp_key(value):
                return True
    elif isinstance(payload, list):
        return any(_contains_timestamp_key(value) for value in payload)
    return False
