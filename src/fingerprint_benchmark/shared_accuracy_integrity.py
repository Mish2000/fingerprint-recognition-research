"""Deterministic integrity helpers for the shared-accuracy milestone.

The shared-accuracy output is intentionally excluded from the protected input
inventory.  Everything else enumerated here is an immutable input: the two
datasets, base protocols, pre-existing result artifacts, SourceAFIS sidecar,
and the implementation files hashed by the frozen pairwise benchmark.

Snapshots and manifests contain paths, byte sizes, SHA-256 digests, and
categories only.  They deliberately contain no wall-clock or filesystem
timestamps.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .hashing import canonical_json_bytes, file_sha256


SNAPSHOT_SCHEMA_VERSION = "shared-accuracy-protected-snapshot-v1"
INTEGRITY_REPORT_SCHEMA_VERSION = "shared-accuracy-protected-integrity-v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "shared-accuracy-artifact-manifest-v1"

DATASETS = ("sd300b", "sd300c")

# These are the exact Python sources included by implementation_provenance().
# Keeping this list explicit avoids accidentally treating new shared-accuracy
# orchestration code as part of an already-frozen method implementation.
FROZEN_BENCHMARK_FILES = (
    "bundle.py",
    "contract.py",
    "hashing.py",
    "io.py",
    "manifest.py",
    "preflight.py",
    "provenance.py",
    "runner.py",
)
FROZEN_SOURCEAFIS_FILES = (
    "sourceafis_adapter.py",
    "sourceafis_client.py",
    "sourceafis_sidecar.py",
)
FROZEN_SIFT_FILES = (
    "__init__.py",
    "adapter.py",
    "config.py",
    "descriptors.py",
    "extractor.py",
    "geometry.py",
    "matching.py",
    "preprocessing.py",
    "scoring.py",
)


class SharedAccuracyIntegrityError(ValueError):
    """Raised when an integrity artifact is invalid or would be overwritten."""


def enumerate_protected_paths(
    project_root: Path,
    data_root: Path,
    *,
    shared_results_root: Path | None = None,
    require_expected: bool = True,
) -> dict[Path, str]:
    """Return the exact protected-file inventory for the milestone.

    ``results/shared_accuracy`` is excluded as a whole, including sibling
    candidate/work directories below it.  Consequently producing the new
    milestone output cannot appear as an added protected input during the
    after-snapshot.
    """

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    shared_results_root = (
        shared_results_root or project_root / "results" / "shared_accuracy"
    ).resolve()

    if require_expected:
        _require_directory(project_root, "project root")
        _require_directory(data_root, "data root")

    paths: dict[Path, str] = {}

    for dataset in DATASETS:
        _add_tree(
            paths,
            data_root / "NIST" / dataset,
            f"dataset_{dataset}",
            required=require_expected,
        )

    _add_tree(
        paths,
        project_root / "protocols",
        "base_protocols",
        required=require_expected,
    )
    _add_tree(
        paths,
        project_root / "results",
        "preexisting_results",
        required=require_expected,
        excluded_root=shared_results_root,
    )
    _add_tree(
        paths,
        project_root / "apps" / "sourceafis-sidecar",
        "sourceafis_app",
        required=require_expected,
    )

    benchmark_root = project_root / "src" / "fingerprint_benchmark"
    for filename in FROZEN_BENCHMARK_FILES:
        _add_file(
            paths,
            benchmark_root / filename,
            "benchmark_implementation",
            required=require_expected,
        )
    for filename in FROZEN_SOURCEAFIS_FILES:
        _add_file(
            paths,
            benchmark_root / filename,
            "sourceafis_python_implementation",
            required=require_expected,
        )
    for filename in FROZEN_SIFT_FILES:
        _add_file(
            paths,
            benchmark_root / "sift" / filename,
            "sift_implementation",
            required=require_expected,
        )

    return dict(sorted(paths.items(), key=lambda item: _path_sort_key(item[0])))


def capture_snapshot(paths: Mapping[Path, str], output_path: Path) -> dict[str, Any]:
    """Hash ``paths`` and immutably publish one deterministic JSONL snapshot."""

    records, footer = _snapshot_records(paths)
    payload = _snapshot_bytes(records, footer)
    publish_immutable_bytes(output_path, payload)
    return {
        **footer,
        "path": str(output_path.resolve()),
        "sha256": file_sha256(output_path.resolve()),
    }


def capture_protected_before(
    project_root: Path,
    data_root: Path,
    output_path: Path,
    *,
    shared_results_root: Path | None = None,
    require_expected: bool = True,
) -> dict[str, Any]:
    """Enumerate protected inputs and immutably capture the before-snapshot."""

    paths = enumerate_protected_paths(
        project_root,
        data_root,
        shared_results_root=shared_results_root,
        require_expected=require_expected,
    )
    return capture_snapshot(paths, output_path)


def compare_snapshot(
    before_snapshot_path: Path,
    current_paths: Mapping[Path, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Compare a frozen snapshot with a freshly enumerated exact path set.

    The return value is ``(report, after_records, after_footer)``.  Returning
    the after payload separately lets callers publish both the raw after
    snapshot and the compact mismatch report without hashing the files twice.
    """

    before_records, before_footer = read_snapshot(before_snapshot_path)
    after_records, after_footer = _snapshot_records(current_paths)
    before_by_path = {record["path"]: record for record in before_records}
    after_by_path = {record["path"]: record for record in after_records}

    before_paths = set(before_by_path)
    after_paths = set(after_by_path)
    added_paths = sorted(after_paths - before_paths, key=str.casefold)
    removed_paths = sorted(before_paths - after_paths, key=str.casefold)

    changed_files: list[dict[str, Any]] = []
    category_mismatches: list[dict[str, str]] = []
    for path in sorted(before_paths & after_paths, key=str.casefold):
        before = before_by_path[path]
        after = after_by_path[path]
        if before["size"] != after["size"] or before["sha256"] != after["sha256"]:
            changed_files.append(
                {
                    "path": path,
                    "category": before["category"],
                    "before_size": before["size"],
                    "after_size": after["size"],
                    "before_sha256": before["sha256"],
                    "after_sha256": after["sha256"],
                }
            )
        if before["category"] != after["category"]:
            category_mismatches.append(
                {
                    "path": path,
                    "before_category": before["category"],
                    "after_category": after["category"],
                }
            )

    mismatch_count = (
        len(added_paths)
        + len(removed_paths)
        + len(changed_files)
        + len(category_mismatches)
    )
    report = {
        "schema_version": INTEGRITY_REPORT_SCHEMA_VERSION,
        "protected_artifacts_unchanged": mismatch_count == 0,
        "before_snapshot_path": str(before_snapshot_path.resolve()),
        "before_snapshot_sha256": file_sha256(before_snapshot_path.resolve()),
        "before": before_footer,
        "after": after_footer,
        "added_path_count": len(added_paths),
        "added_paths": added_paths,
        "removed_path_count": len(removed_paths),
        "removed_paths": removed_paths,
        "changed_file_count": len(changed_files),
        "changed_files": changed_files,
        "category_mismatch_count": len(category_mismatches),
        "category_mismatches": category_mismatches,
        "mismatch_count": mismatch_count,
    }
    return report, after_records, after_footer


def verify_protected_after(
    project_root: Path,
    data_root: Path,
    before_snapshot_path: Path,
    *,
    shared_results_root: Path | None = None,
    after_snapshot_path: Path | None = None,
    report_path: Path | None = None,
    require_expected: bool = True,
    raise_on_mismatch: bool = False,
) -> dict[str, Any]:
    """Re-enumerate protected inputs, publish optional after/report artifacts."""

    current_paths = enumerate_protected_paths(
        project_root,
        data_root,
        shared_results_root=shared_results_root,
        require_expected=require_expected,
    )
    report, after_records, after_footer = compare_snapshot(
        before_snapshot_path,
        current_paths,
    )
    if after_snapshot_path is not None:
        publish_immutable_bytes(
            after_snapshot_path,
            _snapshot_bytes(after_records, after_footer),
        )
        report["after_snapshot_path"] = str(after_snapshot_path.resolve())
        report["after_snapshot_sha256"] = file_sha256(after_snapshot_path.resolve())
    if report_path is not None:
        publish_immutable_json(report_path, report)
        report["report_path"] = str(report_path.resolve())
        report["report_sha256"] = file_sha256(report_path.resolve())
    if raise_on_mismatch and not report["protected_artifacts_unchanged"]:
        raise SharedAccuracyIntegrityError(
            "Protected artifacts changed: "
            f"added={report['added_path_count']}, "
            f"removed={report['removed_path_count']}, "
            f"changed={report['changed_file_count']}, "
            f"category={report['category_mismatch_count']}."
        )
    return report


def require_protected_unchanged(report: Mapping[str, Any]) -> None:
    """Raise unless a comparison report proves exact protected-input equality."""

    if report.get("protected_artifacts_unchanged") is not True:
        raise SharedAccuracyIntegrityError(
            f"Protected artifacts changed; mismatch_count={report.get('mismatch_count')}."
        )


def build_artifact_manifest(
    output_root: Path,
    *,
    manifest_path: Path | None = None,
    namespace: str = "sourceafis_sift_v1",
) -> dict[str, Any]:
    """Build a deterministic manifest of every completed output file.

    Paths in the manifest are relative to ``output_root``.  The manifest file
    itself is excluded whether or not it already exists, which makes exact
    rebuilds stable and avoids a self-referential digest.
    """

    output_root = output_root.resolve()
    _require_directory(output_root, "shared-accuracy output root")
    manifest_path = (manifest_path or output_root / "artifact_manifest.json").resolve()
    try:
        manifest_path.relative_to(output_root)
    except ValueError as exc:
        raise SharedAccuracyIntegrityError(
            "artifact_manifest.json must be located inside the output root."
        ) from exc

    records: list[dict[str, Any]] = []
    for path in sorted(
        (item.resolve() for item in output_root.rglob("*") if item.is_file()),
        key=_path_sort_key,
    ):
        if path == manifest_path:
            continue
        relative = path.relative_to(output_root).as_posix()
        size, digest = _stable_file_identity(path)
        records.append({"path": relative, "size": size, "sha256": digest})

    tree = _records_tree_sha256(records)
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "namespace": namespace,
        "hash_algorithm": "sha256",
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "tree_sha256": tree,
        "files": records,
    }


def write_artifact_manifest(
    output_root: Path,
    *,
    manifest_path: Path | None = None,
    namespace: str = "sourceafis_sift_v1",
) -> dict[str, Any]:
    """Build and immutably publish ``artifact_manifest.json``."""

    output_root = output_root.resolve()
    manifest_path = (manifest_path or output_root / "artifact_manifest.json").resolve()
    manifest = build_artifact_manifest(
        output_root,
        manifest_path=manifest_path,
        namespace=namespace,
    )
    publish_immutable_json(manifest_path, manifest)
    return manifest


def read_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and fully validate one deterministic protected-input snapshot."""

    path = path.resolve()
    records: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                if line_number == 1:
                    header = payload.get("header", {})
                    if header != {
                        "schema_version": SNAPSHOT_SCHEMA_VERSION,
                        "hash_algorithm": "sha256",
                    }:
                        raise SharedAccuracyIntegrityError(
                            f"Invalid protected snapshot header: {path}"
                        )
                    continue
                if "footer" in payload:
                    if footer is not None:
                        raise SharedAccuracyIntegrityError(
                            f"Protected snapshot has multiple footers: {path}"
                        )
                    footer = payload["footer"]
                    continue
                if footer is not None:
                    raise SharedAccuracyIntegrityError(
                        f"Protected snapshot contains records after its footer: {path}"
                    )
                _validate_snapshot_record(payload, path)
                records.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise SharedAccuracyIntegrityError(
            f"Cannot read protected snapshot {path}: {exc}"
        ) from exc

    if footer is None:
        raise SharedAccuracyIntegrityError(f"Protected snapshot footer is missing: {path}")
    paths = [record["path"] for record in records]
    if paths != sorted(paths, key=str.casefold) or len(paths) != len(set(paths)):
        raise SharedAccuracyIntegrityError(
            f"Protected snapshot paths are not unique and deterministically sorted: {path}"
        )
    expected_footer = _snapshot_footer(records)
    if footer != expected_footer:
        raise SharedAccuracyIntegrityError(
            f"Protected snapshot footer does not match its records: {path}"
        )
    return records, footer


def publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish canonical human-readable JSON, allowing only an exact repeat."""

    publish_immutable_bytes(path, _json_bytes(payload))


def publish_immutable_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes once; exact repeats pass, changed repeats fail."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise SharedAccuracyIntegrityError(
            f"Immutable artifact already exists with different content: {path}"
        )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise SharedAccuracyIntegrityError(
            f"Cannot publish immutable artifact {path}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _add_tree(
    paths: dict[Path, str],
    root: Path,
    category: str,
    *,
    required: bool,
    excluded_root: Path | None = None,
) -> None:
    root = root.resolve()
    if not root.is_dir():
        if required:
            raise SharedAccuracyIntegrityError(
                f"Required protected directory is missing ({category}): {root}"
            )
        return
    excluded_root = excluded_root.resolve() if excluded_root is not None else None
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=_path_sort_key):
        resolved = path.resolve()
        if excluded_root is not None and _is_within(resolved, excluded_root):
            continue
        _register_path(paths, resolved, category)


def _add_file(
    paths: dict[Path, str],
    path: Path,
    category: str,
    *,
    required: bool,
) -> None:
    path = path.resolve()
    if not path.is_file():
        if required:
            raise SharedAccuracyIntegrityError(
                f"Required protected file is missing ({category}): {path}"
            )
        return
    _register_path(paths, path, category)


def _register_path(paths: dict[Path, str], path: Path, category: str) -> None:
    previous = paths.get(path)
    if previous is not None and previous != category:
        raise SharedAccuracyIntegrityError(
            f"Protected file belongs to multiple categories: {path}: {previous}, {category}"
        )
    paths[path] = category


def _snapshot_records(
    paths: Mapping[Path, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized: dict[Path, str] = {}
    for raw_path, raw_category in paths.items():
        path = Path(raw_path).resolve()
        category = str(raw_category).strip()
        if not category:
            raise SharedAccuracyIntegrityError(f"Protected file has an empty category: {path}")
        if path in normalized and normalized[path] != category:
            raise SharedAccuracyIntegrityError(
                f"Protected file belongs to multiple categories: {path}"
            )
        normalized[path] = category

    records: list[dict[str, Any]] = []
    for path, category in sorted(normalized.items(), key=lambda item: _path_sort_key(item[0])):
        if not path.is_file():
            raise SharedAccuracyIntegrityError(f"Protected file is missing: {path}")
        size, digest = _stable_file_identity(path)
        records.append(
            {
                "category": category,
                "path": path.as_posix(),
                "size": size,
                "sha256": digest,
            }
        )
    return records, _snapshot_footer(records)


def _snapshot_footer(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "tree_sha256": _records_tree_sha256(records),
        "category_counts": dict(
            sorted(Counter(record["category"] for record in records).items())
        ),
    }


def _snapshot_bytes(records: list[dict[str, Any]], footer: Mapping[str, Any]) -> bytes:
    payloads: list[dict[str, Any]] = [
        {
            "header": {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "hash_algorithm": "sha256",
            }
        },
        *records,
        {"footer": dict(footer)},
    ]
    return (
        "\n".join(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            for payload in payloads
        )
        + "\n"
    ).encode("utf-8")


def _records_tree_sha256(records: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json_bytes(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_file_identity(path: Path) -> tuple[int, str]:
    """Hash a file and reject a concurrent mutation without persisting mtimes."""

    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise SharedAccuracyIntegrityError(f"File changed while it was being hashed: {path}")
    return int(after.st_size), digest


def _validate_snapshot_record(record: Any, snapshot_path: Path) -> None:
    expected_keys = {"category", "path", "size", "sha256"}
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise SharedAccuracyIntegrityError(
            f"Invalid protected snapshot record in {snapshot_path}: {record!r}"
        )
    if not isinstance(record["category"], str) or not record["category"].strip():
        raise SharedAccuracyIntegrityError(f"Invalid snapshot category in {snapshot_path}")
    if not isinstance(record["path"], str) or not record["path"].strip():
        raise SharedAccuracyIntegrityError(f"Invalid snapshot path in {snapshot_path}")
    if type(record["size"]) is not int or record["size"] < 0:
        raise SharedAccuracyIntegrityError(f"Invalid snapshot size in {snapshot_path}")
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SharedAccuracyIntegrityError(f"Invalid snapshot SHA-256 in {snapshot_path}")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SharedAccuracyIntegrityError(f"Required {label} does not exist: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _path_sort_key(path: Path) -> tuple[str, str]:
    value = path.as_posix()
    return value.casefold(), value
