"""Clean-room, runtime, source, and representation provenance for HarrisZ+."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.hashing import file_sha256


PAPER_PROVENANCE: dict[str, Any] = {
    "title": "HarrisZ+: Harris Corner Selection for Next-Gen Image Matching Pipelines",
    "authors": ["Fabio Bellavia", "Dmytro Mishkin"],
    "arxiv_identifier": "2109.12925",
    "arxiv_version": "v6",
    "url": "https://arxiv.org/abs/2109.12925v6",
    "specification_scope": {
        "sections": ["3.1", "3.2"],
        "equations": "1-16",
    },
    "reviewed_pdf_filename": "2109.12925v6.pdf",
    "reviewed_pdf_sha256": "c9560b7e89849e7e4a05a4b376bc4eac2116eac8816b13fae9ff69859cb374df",
}

ORIGINAL_HARRISZ_PAPER_PROVENANCE: dict[str, Any] = {
    "title": "Improving Harris corner selection strategy",
    "authors": ["Fabio Bellavia", "Domenico Tegolo", "Carlo Valenti"],
    "venue": "IET Computer Vision",
    "year": 2011,
    "reviewed_pdf_filename": "iet.pdf",
    "reviewed_pdf_sha256": "513a8c18783ca7d6ce9b5beeecdfb796797df673a3289f303aa4124f079bcd86",
}

REFERENCE_ORACLE_PROVENANCE: dict[str, Any] = {
    "repository_url": "https://github.com/fb82/HarrisZ",
    "commit": "ec3a285ae3a86cc71d3026c01e24ec13554e3800",
    "commit_date": "2025-07-18T01:50:14+02:00",
    "branch_at_review": "main",
    "license": "GNU General Public License v3.0",
    "role": (
        "reference oracle only for response/keypoint comparison and underspecified "
        "conventions; it is not a runtime dependency"
    ),
    "reviewed_head_date": "2026-07-16",
}

CLEAN_ROOM_STATEMENT = (
    "This repository implementation was written independently from the equations and prose "
    "in Sections 3.1 and 3.2 of the HarrisZ+ paper. The authors' GPL-3.0 repository was used "
    "only as a source-level reference oracle for validation and underspecified conventions. No "
    "function, code fragment, or translated implementation from that repository was copied."
)

_HARRISZPLUS_SOURCE_FILES = (
    "config.py",
    "types.py",
    "kernels.py",
    "reference_cpu.py",
    "cuda_detector.py",
    "selection.py",
    "orientation.py",
    "extractor.py",
    "adapter.py",
    "provenance.py",
)

_REUSED_SIFT_SOURCE_FILES = (
    "config.py",
    "descriptors.py",
    "extractor.py",
    "geometry.py",
    "matching.py",
)


def clean_room_provenance() -> dict[str, Any]:
    """Return the immutable paper/oracle provenance and clean-room declaration."""

    return {
        "implementation_policy": "clean_room_from_paper",
        "statement": CLEAN_ROOM_STATEMENT,
        "primary_paper": dict(PAPER_PROVENANCE),
        "original_harrisz_paper": dict(ORIGINAL_HARRISZ_PAPER_PROVENANCE),
        "official_reference_oracle": dict(REFERENCE_ORACLE_PROVENANCE),
        "copied_external_functions_or_fragments": False,
    }


def implementation_source_hashes(*, strict: bool = True) -> dict[str, Any]:
    """Hash every method source plus the unchanged SIFT modules it directly reuses."""

    package_dir = Path(__file__).resolve().parent
    sift_dir = package_dir.parent / "sift"
    required_hashes = _hash_files(
        package_dir,
        _HARRISZPLUS_SOURCE_FILES,
        strict=strict,
    )
    package_python_files = tuple(path.name for path in sorted(package_dir.glob("*.py")))
    return {
        "harriszplus": _hash_files(package_dir, package_python_files, strict=True),
        "required_score_producing_sources": required_hashes,
        "reused_sift_unchanged": _hash_files(
            sift_dir,
            _REUSED_SIFT_SOURCE_FILES,
            strict=strict,
        ),
        "reuse_contract": {
            "representation": "fingerprint_benchmark.sift.extractor.SiftRepresentation",
            "rootsift": "fingerprint_benchmark.sift.descriptors.rootsift",
            "matching": "fingerprint_benchmark.sift.matching.match_descriptors",
            "geometry": "fingerprint_benchmark.sift.geometry.verify_geometry",
            "modification_policy": "imported unchanged",
        },
    }


def dependency_artifact_hashes(*, strict: bool = False) -> dict[str, str | None]:
    """Hash repository dependency declarations without depending on a lockfile."""

    repo_root = Path(__file__).resolve().parents[3]
    result: dict[str, str | None] = {}
    for name in ("pyproject.toml", "environment.yml"):
        path = repo_root / name
        if path.is_file():
            result[name] = file_sha256(path)
        elif strict:
            raise FileNotFoundError(f"Required dependency artifact does not exist: {path}")
        else:
            result[name] = None
    return result


def runtime_metadata(config: object | None = None) -> dict[str, Any]:
    """Collect CPU, OpenCV, Python, PyTorch, CUDA, driver, and GPU metadata."""

    torch_info = _torch_runtime(config)
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "opencv_version": cv2.__version__,
        "opencv_distribution": _distribution_versions(
            ("opencv-python", "opencv-python-headless")
        ),
        "opencv_build_information": cv2.getBuildInformation(),
        "opencv_thread_count": int(cv2.getNumThreads()),
        "opencv_optimized": bool(cv2.useOptimized()),
        "numpy_version": np.__version__,
        "operating_system": platform.platform(),
        "cpu_architecture": platform.machine(),
        "processor": platform.processor(),
        "torch": torch_info,
        "nvidia_smi": _nvidia_smi_metadata(),
        "dependency_artifact_sha256": dependency_artifact_hashes(),
        "runtime_network_policy": "no runtime downloads or model weights",
    }


def determinism_metadata(config: object) -> dict[str, Any]:
    """Describe the deterministic settings enforced by the adapter."""

    return {
        "seed": int(getattr(config, "rng_seed")),
        "numpy_seeded": True,
        "opencv_rng_seeded_before_geometry": True,
        "torch_manual_seeded": True,
        "torch_cuda_manual_seeded": True,
        "torch_use_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "autocast": False,
        "detector_dtype": "float32",
        "fp16": False,
        "bf16": False,
    }


def representation_sha256(representation: object) -> str:
    """Hash only deterministic representation content, excluding wall-clock diagnostics."""

    payload = getattr(representation, "payload", None)
    if payload is not None and hasattr(payload, "descriptors"):
        representation = payload
    digest = hashlib.sha256()
    digest.update(b"harriszplus-rootsift-representation-v1\0")
    arrays = (
        ("points", "<f4"),
        ("sizes", "<f4"),
        ("angles", "<f4"),
        ("responses", "<f4"),
        ("octaves", "<i4"),
        ("class_ids", "<i4"),
        ("descriptors", "<f4"),
    )
    for name, dtype in arrays:
        value = np.ascontiguousarray(getattr(representation, name), dtype=np.dtype(dtype))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    scalars = {
        "width": int(getattr(representation, "width")),
        "height": int(getattr(representation, "height")),
        "ppi": float(getattr(representation, "ppi")),
    }
    digest.update(json.dumps(scalars, sort_keys=True, separators=(",", ":")).encode("ascii"))
    metadata = getattr(representation, "metadata", {})
    stable_metadata = {
        key: metadata[key]
        for key in (
            "harriszplus_scale_indices",
            "harriszplus_source_indices",
            "scale_mapping_records",
            "opencv_octave_policy",
            "opencv_class_id_policy",
        )
        if key in metadata
    }
    digest.update(
        json.dumps(
            stable_metadata,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _hash_files(
    directory: Path,
    names: tuple[str, ...],
    *,
    strict: bool,
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            if strict:
                raise FileNotFoundError(f"Required implementation source does not exist: {path}")
            result[name] = None
            continue
        result[name] = file_sha256(path)
    return result


def _distribution_versions(names: tuple[str, ...]) -> list[dict[str, str]]:
    installed: list[dict[str, str]] = []
    for name in names:
        try:
            installed.append({"name": name, "version": importlib.metadata.version(name)})
        except importlib.metadata.PackageNotFoundError:
            continue
    return installed


def _torch_runtime(config: object | None) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "cuda_available": False}

    payload: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": (
            None if not torch.backends.cudnn.is_available() else torch.backends.cudnn.version()
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if not torch.cuda.is_available():
        return payload
    requested_device = getattr(config, "device", None) if config is not None else None
    device = torch.device(requested_device or "cuda")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    properties = torch.cuda.get_device_properties(index)
    payload.update(
        {
            "selected_device": str(device),
            "device_index": index,
            "gpu_model": properties.name,
            "total_vram_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    )
    return payload


def _nvidia_smi_metadata() -> list[Mapping[str, Any]] | dict[str, Any]:
    query = (
        "index,name,memory.total,driver_version,compute_cap"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}
    if completed.returncode != 0:
        return {
            "status": "error",
            "returncode": int(completed.returncode),
            "stderr": completed.stderr.strip(),
        }
    records: list[Mapping[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        records.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "driver_version": fields[3],
                "compute_capability": fields[4],
            }
        )
    return records
