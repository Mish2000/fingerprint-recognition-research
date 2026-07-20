"""Exact parity gate against the protected detector-only Harris pathway."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.detectors.opencv_gftt_harris import (
    OpenCVGFTTHarrisRootSIFTGeometricAdapter,
)
from fingerprint_benchmark.hashing import file_sha256, stable_hash
from fingerprint_benchmark.manifest import read_pair_manifest
from fingerprint_benchmark.runner import _execute_pair, prepare_run_context

from .adapter import GFTTHarrisRootSIFTGeometricAdapter
from .restored_equivalence import PARENT_PROTOCOL_SHA256, PARENT_RUN_CONFIG_HASH


DATASETS = ("sd300b", "sd300c")
PAIR_KINDS = (
    "plain_self",
    "roll_self",
    "plain_roll_genuine",
    "plain_roll_impostor",
)
SAMPLE_STRIDE_COUNT = 10


class ParityError(RuntimeError):
    """The immutable oracle or required dataset cannot be evaluated."""


def deterministic_value(value: Any) -> Any:
    """Recursively remove wall-clock measurements and volatile path/process fields."""

    if isinstance(value, dict):
        return {
            key: deterministic_value(item)
            for key, item in value.items()
            if not key.endswith("_ms")
            and key not in {"timestamp", "temporary_path", "process_id"}
        }
    if isinstance(value, list):
        return [deterministic_value(item) for item in value]
    return value


def deterministic_diagnostics(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ParityError("Diagnostics payload must be a JSON object.")
    return deterministic_value(payload)


def select_sample_indices(rows: list[Mapping[str, str]], *, pair_kind: str) -> dict[int, list[str]]:
    """Frozen stride plus every available outcome class missed by the stride."""

    if not rows:
        raise ParityError("Detector-only oracle contains no rows.")
    total = len(rows)
    selected: dict[int, list[str]] = {}
    for i in range(SAMPLE_STRIDE_COUNT):
        index = (i * total) // SAMPLE_STRIDE_COUNT
        selected.setdefault(index, []).append(f"stride_{i}")

    def score(row: Mapping[str, str]) -> float | None:
        raw = row.get("raw_score", "")
        return None if raw == "" else float(raw)

    def compare_diag(row: Mapping[str, str]) -> dict[str, Any]:
        return deterministic_diagnostics(row.get("compare_diagnostics", ""))

    coverage: tuple[tuple[str, Any], ...] = (
        ("positive_score", lambda row: row.get("status") == "ok" and (score(row) or 0.0) > 0.0),
        ("valid_zero_score", lambda row: row.get("status") == "ok" and score(row) == 0.0),
        ("technical_failure", lambda row: row.get("status") != "ok"),
        ("few_descriptors", lambda row: row.get("error_code") == "too_few_descriptors"),
        (
            "few_matches",
            lambda row: row.get("status") == "ok"
            and int(compare_diag(row).get("matches_submitted_to_geometry", 3)) < 3,
        ),
        (
            "geometry_failure",
            lambda row: row.get("status") == "ok"
            and compare_diag(row).get("geometry_failure_reason") is not None,
        ),
    )
    for reason, predicate in coverage:
        index = next((i for i, row in enumerate(rows) if predicate(row)), None)
        if index is not None:
            selected.setdefault(index, []).append(reason)

    scored = [(float(row["raw_score"]), index) for index, row in enumerate(rows) if row["raw_score"]]
    if scored:
        _, index = max(scored)
        selected.setdefault(index, []).append("highest_historical_score")
    if pair_kind == "plain_roll_impostor":
        index = next(
            (
                i
                for i, row in enumerate(rows)
                if row.get("status") == "ok" and (score(row) or 0.0) > 0.0
            ),
            None,
        )
        if index is not None:
            selected.setdefault(index, []).append("historical_positive_impostor")
    return selected


def compare_result_rows(
    expected: Mapping[str, str],
    actual: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Compare all deterministic outcome content with no tolerance."""

    mismatches: list[dict[str, Any]] = []
    for field in ("status", "raw_score", "error_code", "error_message"):
        if expected.get(field, "") != actual.get(field, ""):
            mismatches.append(
                {
                    "field": field,
                    "expected": expected.get(field, ""),
                    "actual": actual.get(field, ""),
                }
            )
    for field in ("prepare_a_diagnostics", "prepare_b_diagnostics", "compare_diagnostics"):
        expected_value = deterministic_diagnostics(expected.get(field, ""))
        actual_value = deterministic_diagnostics(actual.get(field, ""))
        if expected_value != actual_value:
            mismatches.append(
                {"field": field, "expected": expected_value, "actual": actual_value}
            )
    return mismatches


def result_summary(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "status": row.get("status", ""),
        "raw_score": row.get("raw_score", ""),
        "error_code": row.get("error_code", ""),
        "error_message": row.get("error_message", ""),
        "prepare_a_diagnostics": deterministic_diagnostics(
            row.get("prepare_a_diagnostics", "")
        ),
        "prepare_b_diagnostics": deterministic_diagnostics(
            row.get("prepare_b_diagnostics", "")
        ),
        "compare_diagnostics": deterministic_diagnostics(row.get("compare_diagnostics", "")),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ParityError(f"Historical detector-only result is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _commit(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_bundle_parity(
    *,
    dataset: str,
    pair_kind: str,
    repository_root: Path,
    historical_results_root: Path,
) -> dict[str, Any]:
    manifest_path = (
        repository_root
        / "protocols"
        / "detector_only_joint_500_v1"
        / dataset
        / f"{pair_kind}.csv"
    )
    oracle_directory = historical_results_root / dataset / pair_kind / (
        "opencv_gftt_harris_rootsift_geometric"
    )
    historical_rows = _read_rows(oracle_directory / "pairs.csv")
    historical_metadata = json.loads(
        (oracle_directory / "run_metadata.json").read_text(encoding="utf-8")
    )
    if historical_metadata.get("config_hash") != PARENT_RUN_CONFIG_HASH:
        raise ParityError(
            f"Historical Harris config hash changed in {oracle_directory}: "
            f"{historical_metadata.get('config_hash')!r}."
        )
    pairs = read_pair_manifest(manifest_path)
    if len(pairs) != len(historical_rows):
        raise ParityError(
            f"Manifest/oracle row-count mismatch for {dataset}/{pair_kind}: "
            f"{len(pairs)} != {len(historical_rows)}."
        )
    historical_by_id = {row["pair_id"]: row for row in historical_rows}
    selection = select_sample_indices(historical_rows, pair_kind=pair_kind)

    detector_adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
    full_adapter = GFTTHarrisRootSIFTGeometricAdapter()
    protocol = f"detector_only_joint_500_v1_{pair_kind}"
    try:
        detector_context = prepare_run_context(
            manifest_path=manifest_path,
            expected_dataset=dataset,
            expected_protocol=protocol,
            adapter=detector_adapter,
            results_root=repository_root / "results",
        )
        full_context = prepare_run_context(
            manifest_path=manifest_path,
            expected_dataset=dataset,
            expected_protocol=protocol,
            adapter=full_adapter,
            results_root=repository_root / "results",
        )
        if detector_context.spec.config_hash != PARENT_RUN_CONFIG_HASH:
            raise ParityError(
                "Current detector-only config hash no longer matches the Joint-500 oracle: "
                f"{detector_context.spec.config_hash}."
            )
        pair_results = []
        for index in sorted(selection):
            pair = pairs[index]
            historical = historical_by_id.get(pair.pair_id)
            if historical is None:
                raise ParityError(f"Oracle has no row for {pair.pair_id!r}.")
            detector_row = _execute_pair(
                pair,
                detector_adapter,
                run_spec=detector_context.spec,
                method_metadata=detector_context.method_metadata,
            )
            full_row = _execute_pair(
                pair,
                full_adapter,
                run_spec=full_context.spec,
                method_metadata=full_context.method_metadata,
            )
            oracle_mismatches = compare_result_rows(historical, detector_row)
            parity_mismatches = compare_result_rows(detector_row, full_row)
            pair_results.append(
                {
                    "index": index,
                    "pair_id": pair.pair_id,
                    "selection_reasons": selection[index],
                    "expected_result": result_summary(detector_row),
                    "actual_result": result_summary(full_row),
                    "historical_oracle_result": result_summary(historical),
                    "oracle_mismatches": oracle_mismatches,
                    "mismatch_details": parity_mismatches,
                    "pass": not oracle_mismatches and not parity_mismatches,
                }
            )
    finally:
        detector_adapter.close()
        full_adapter.close()

    return {
        "dataset": dataset,
        "pair_kind": pair_kind,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "historical_pairs_path": str(oracle_directory / "pairs.csv"),
        "historical_pairs_sha256": file_sha256(oracle_directory / "pairs.csv"),
        "historical_metadata_sha256": file_sha256(oracle_directory / "run_metadata.json"),
        "historical_row_count": len(historical_rows),
        "selected_pair_count": len(pair_results),
        "matched_pair_count": sum(1 for item in pair_results if item["pass"]),
        "status": "pass" if all(item["pass"] for item in pair_results) else "fail",
        "pairs": pair_results,
    }


def run_parity(
    *,
    repository_root: Path,
    historical_results_root: Path | None = None,
    output_path: Path | None = None,
    current_commit_before_implementation: str | None = None,
) -> dict[str, Any]:
    """Run at least 80 exact comparisons and persist an untracked report."""

    root = repository_root.resolve()
    results_root = (
        historical_results_root.resolve()
        if historical_results_root is not None
        else root / "results" / "detector_only_joint_500_v1"
    )
    destination = (
        output_path.resolve()
        if output_path is not None
        else root
        / "results"
        / "restoration_preflight"
        / "gftt_harris_rootsift_geometric_v1"
        / "parity_report.json"
    )
    protocol_metadata_path = (
        root / "protocols" / "detector_only_joint_500_v1" / "protocol_metadata.json"
    )
    protocol_metadata = json.loads(protocol_metadata_path.read_text(encoding="utf-8"))
    if protocol_metadata.get("protocol_sha256") != PARENT_PROTOCOL_SHA256:
        raise ParityError("detector_only_joint_500_v1 protocol SHA-256 changed.")

    full_adapter = GFTTHarrisRootSIFTGeometricAdapter()
    detector_adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
    try:
        full_metadata = full_adapter.metadata()
        detector_metadata = detector_adapter.metadata()
    finally:
        full_adapter.close()
        detector_adapter.close()

    bundle_specs = [(dataset, pair_kind) for dataset in DATASETS for pair_kind in PAIR_KINDS]
    # Each worker owns its OpenCV global RNG/thread settings.  Process isolation
    # avoids the RNG races that a thread pool would introduce while cutting the
    # wall time of this read-only gate; bundle result ordering remains frozen.
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_bundle_parity,
                dataset=dataset,
                pair_kind=pair_kind,
                repository_root=root,
                historical_results_root=results_root,
            )
            for dataset, pair_kind in bundle_specs
        ]
        bundles = [future.result() for future in futures]
    selected_count = sum(bundle["selected_pair_count"] for bundle in bundles)
    mismatches = [
        {
            "dataset": bundle["dataset"],
            "pair_kind": bundle["pair_kind"],
            "pair_id": pair["pair_id"],
            "oracle_mismatches": pair["oracle_mismatches"],
            "mismatch_details": pair["mismatch_details"],
        }
        for bundle in bundles
        for pair in bundle["pairs"]
        if not pair["pass"]
    ]
    report: dict[str, Any] = {
        "parity_schema": "gftt-harris-rootsift-geometric-full-parity-v1",
        "current_commit_before_implementation": (
            current_commit_before_implementation or _commit(root)
        ),
        "full_adapter_identity": {
            "method_name": full_metadata.method,
            "method_version": full_metadata.method_version,
            "score_direction": full_metadata.score_direction,
        },
        "detector_only_adapter_identity": {
            "method_name": detector_metadata.method,
            "method_version": detector_metadata.method_version,
            "score_direction": detector_metadata.score_direction,
        },
        "full_config": full_metadata.config,
        "detector_only_config": detector_metadata.config,
        "full_algorithm_config_hash": full_metadata.config["algorithm_config_hash"],
        "detector_only_historical_run_config_hash": PARENT_RUN_CONFIG_HASH,
        "detector_only_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "comparison_policy": {
            "tolerance": "none; exact equality required",
            "compared": [
                "status",
                "raw_score",
                "error_code",
                "error_message",
                "all deterministic prepare and compare diagnostics",
            ],
            "excluded": [
                "wall-clock fields ending in _ms",
                "timestamps",
                "temporary paths",
                "process IDs",
                "method identity",
                "method version",
                "implementation hash",
            ],
            "sample_rule": (
                "index_i = floor(i * N / 10), i=0..9, in each of eight groups; "
                "plus available historical outcome/edge classes"
            ),
            "execution_isolation": "two processes; each bundle executes cold pairs serially",
        },
        "runtime": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
        },
        "bundle_count": len(bundles),
        "selected_pair_count": selected_count,
        "matched_pair_count": sum(bundle["matched_pair_count"] for bundle in bundles),
        "mismatch_count": len(mismatches),
        "status": "pass" if not mismatches and selected_count >= 80 else "fail",
        "mismatches": mismatches,
        "bundles": bundles,
        "report_sha256_scope": "canonical report object excluding report_sha256",
    }
    report["report_sha256"] = stable_hash(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report["report_path"] = str(destination)
    report["report_file_sha256"] = file_sha256(destination)
    return report


__all__ = [
    "DATASETS",
    "PAIR_KINDS",
    "ParityError",
    "compare_result_rows",
    "deterministic_diagnostics",
    "run_bundle_parity",
    "run_parity",
    "select_sample_indices",
]
