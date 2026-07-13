"""Deterministic implementation and repository provenance helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
from typing import Any

from .contract import BENCHMARK_CONTRACT_VERSION, MethodAdapter, MethodMetadata
from .hashing import file_sha256, stable_hash


class ProvenanceError(ValueError):
    """Raised when required implementation provenance is unavailable."""


def implementation_provenance(
    *,
    adapter: MethodAdapter,
    method_metadata: MethodMetadata,
    startup_validation: dict[str, Any],
    runner_source_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return full provenance, deterministic hash components, and their hash."""

    adapter_source_path = _source_path(adapter.__class__)
    package_directory = Path(__file__).resolve().parent
    contract_source_path = package_directory / "contract.py"
    benchmark_support_sources = _source_hashes(
        package_directory,
        ("bundle.py", "hashing.py", "io.py", "manifest.py", "preflight.py", "provenance.py"),
    )
    method_support_sources = (
        _source_hashes(
            package_directory,
            ("sourceafis_client.py", "sourceafis_sidecar.py"),
        )
        if method_metadata.method == "sourceafis"
        else (
            _source_hashes(
                package_directory / "sift",
                (
                    "__init__.py",
                    "adapter.py",
                    "config.py",
                    "descriptors.py",
                    "extractor.py",
                    "geometry.py",
                    "matching.py",
                    "preprocessing.py",
                    "scoring.py",
                ),
            )
            if method_metadata.method == "sift_geometric"
            else {}
        )
    )
    fixed_components = {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "method": method_metadata.method,
        "method_version": method_metadata.method_version,
        "score_direction": method_metadata.score_direction,
        "score_semantics": method_metadata.score_semantics,
        "adapter_declared_provenance": method_metadata.implementation_provenance,
        "sidecar_jar_sha256": startup_validation.get("jar_sha256"),
        "python_adapter_source_sha256": file_sha256(adapter_source_path),
        "benchmark_runner_source_sha256": file_sha256(runner_source_path),
        "benchmark_contract_source_sha256": file_sha256(contract_source_path),
        "benchmark_support_source_sha256": benchmark_support_sources,
        "method_support_source_sha256": method_support_sources,
    }
    if method_metadata.method == "sourceafis" and not fixed_components["sidecar_jar_sha256"]:
        raise ProvenanceError("SourceAFIS persisted runs require the managed sidecar JAR SHA-256.")

    implementation_hash = stable_hash(fixed_components)
    full = {
        **method_metadata.implementation_provenance,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "sidecar_jar_path": startup_validation.get("jar_path"),
        "sidecar_jar_sha256": startup_validation.get("jar_sha256"),
        "java_executable": startup_validation.get("java_executable"),
        "python_adapter_source": {
            "path": str(adapter_source_path),
            "sha256": fixed_components["python_adapter_source_sha256"],
        },
        "benchmark_runner_source": {
            "path": str(runner_source_path.resolve()),
            "sha256": fixed_components["benchmark_runner_source_sha256"],
        },
        "benchmark_contract_source": {
            "path": str(contract_source_path.resolve()),
            "sha256": fixed_components["benchmark_contract_source_sha256"],
        },
        "benchmark_support_sources": benchmark_support_sources,
        "method_support_sources": method_support_sources,
        "repository": repository_state(runner_source_path),
    }
    return full, fixed_components, implementation_hash


def repository_state(path: Path) -> dict[str, Any]:
    root = path.resolve().parent
    probe = _git(root, "rev-parse", "--show-toplevel")
    if probe is None:
        return {
            "is_git_checkout": False,
            "root": None,
            "commit": None,
            "dirty": None,
        }
    repo_root = Path(probe)
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "is_git_checkout": True,
        "root": str(repo_root),
        "commit": commit,
        "dirty": bool(status),
    }


def _source_path(subject: type[Any]) -> Path:
    source = inspect.getsourcefile(subject)
    if not source:
        raise ProvenanceError(f"Cannot locate source for adapter class {subject!r}.")
    path = Path(source)
    if not path.is_file():
        raise ProvenanceError(f"Adapter source does not exist: {path}")
    return path.resolve()


def _source_hashes(directory: Path, filenames: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in filenames:
        path = directory / filename
        if not path.is_file():
            raise ProvenanceError(f"Required implementation source does not exist: {path}")
        hashes[filename] = file_sha256(path)
    return hashes


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
