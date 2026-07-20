"""Cross-process repeatability gate for representative Harris comparisons."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.hashing import file_sha256, stable_hash
from fingerprint_benchmark.manifest import PairRecord, read_pair_manifest

from .adapter import GFTTHarrisRootSIFTGeometricAdapter
from .parity import deterministic_diagnostics, deterministic_value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_one(
    *,
    repository_root: Path,
    historical_results_root: Path,
    dataset: str,
    pair_kind: str,
    predicate: Callable[[Mapping[str, str], dict[str, Any]], bool],
    excluded_pair_ids: set[str],
) -> PairRecord:
    rows = _read_rows(
        historical_results_root
        / dataset
        / pair_kind
        / "opencv_gftt_harris_rootsift_geometric"
        / "pairs.csv"
    )
    manifest = read_pair_manifest(
        repository_root
        / "protocols"
        / "detector_only_joint_500_v1"
        / dataset
        / f"{pair_kind}.csv"
    )
    pairs = {pair.pair_id: pair for pair in manifest}
    for row in rows:
        diagnostics = deterministic_diagnostics(row.get("compare_diagnostics", ""))
        if row["pair_id"] not in excluded_pair_ids and predicate(row, diagnostics):
            return pairs[row["pair_id"]]
    raise ValueError(f"No repeatability case found for {dataset}/{pair_kind}.")


def select_repeatability_cases(
    *,
    repository_root: Path,
    historical_results_root: Path,
) -> list[tuple[str, PairRecord]]:
    """Select six distinct cases named by the implementation requirement."""

    selected: list[tuple[str, PairRecord]] = []
    used: set[str] = set()

    def add(
        label: str,
        dataset: str,
        pair_kind: str,
        predicate: Callable[[Mapping[str, str], dict[str, Any]], bool],
    ) -> None:
        pair = _select_one(
            repository_root=repository_root,
            historical_results_root=historical_results_root,
            dataset=dataset,
            pair_kind=pair_kind,
            predicate=predicate,
            excluded_pair_ids=used,
        )
        selected.append((label, pair))
        used.add(pair.pair_id)

    positive = lambda row, _diag: row["status"] == "ok" and float(row["raw_score"]) > 0.0
    zero = lambda row, _diag: row["status"] == "ok" and float(row["raw_score"]) == 0.0
    add("plain_self_positive", "sd300b", "plain_self", positive)
    add("roll_self_positive", "sd300b", "roll_self", positive)
    add("plain_roll_genuine_positive", "sd300b", "plain_roll_genuine", positive)
    add("plain_roll_genuine_zero", "sd300b", "plain_roll_genuine", zero)
    add("plain_roll_impostor", "sd300b", "plain_roll_impostor", lambda row, _diag: row["status"] == "ok")
    add(
        "geometry_failure",
        "sd300c",
        "plain_self",
        lambda row, diag: row["status"] == "ok"
        and diag.get("geometry_failure_reason") is not None,
    )
    return selected


def _run_cases(cases: list[tuple[str, PairRecord]]) -> list[dict[str, Any]]:
    results = []
    for label, pair in cases:
        adapter = GFTTHarrisRootSIFTGeometricAdapter()
        try:
            prepared_a = adapter.prepare(pair.path_a, pair.image_metadata_a())
            prepared_b = adapter.prepare(pair.path_b, pair.image_metadata_b())
            comparison = adapter.compare(prepared_a.representation, prepared_b.representation)
            results.append(
                {
                    "label": label,
                    "dataset": pair.dataset,
                    "protocol": pair.protocol,
                    "pair_id": pair.pair_id,
                    "status": "ok",
                    "error_code": "",
                    "raw_score": repr(float(comparison.raw_score)),
                    "representation_sha256_a": prepared_a.diagnostics[
                        "representation_sha256"
                    ],
                    "representation_sha256_b": prepared_b.diagnostics[
                        "representation_sha256"
                    ],
                    "prepare_a_diagnostics": deterministic_value(prepared_a.diagnostics),
                    "prepare_b_diagnostics": deterministic_value(prepared_b.diagnostics),
                    "compare_diagnostics": deterministic_value(comparison.diagnostics),
                }
            )
        finally:
            adapter.close()
    return results


def run_repeatability(
    *,
    repository_root: Path,
    historical_results_root: Path | None = None,
    output_path: Path | None = None,
    process_count: int = 3,
) -> dict[str, Any]:
    root = repository_root.resolve()
    historical = (
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
        / "repeatability_report.json"
    )
    cases = select_repeatability_cases(
        repository_root=root,
        historical_results_root=historical,
    )
    # A worker handles every selected case once.  Therefore every case is
    # independently reconstructed in each of three fresh processes.
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(_run_cases, cases) for _ in range(process_count)]
        runs = [future.result() for future in futures]

    mismatches = []
    for case_index, (label, pair) in enumerate(cases):
        expected = runs[0][case_index]
        for run_index in range(1, len(runs)):
            if runs[run_index][case_index] != expected:
                mismatches.append(
                    {
                        "label": label,
                        "pair_id": pair.pair_id,
                        "run_index": run_index,
                        "expected": expected,
                        "actual": runs[run_index][case_index],
                    }
                )
    report: dict[str, Any] = {
        "repeatability_schema": "gftt-harris-rootsift-geometric-repeatability-v1",
        "process_count": process_count,
        "case_count": len(cases),
        "comparison_count": process_count * len(cases),
        "comparison_policy": "exact equality after recursive removal of timing fields ending in _ms",
        "runtime": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
        },
        "selected_cases": [
            {"label": label, "dataset": pair.dataset, "protocol": pair.protocol, "pair_id": pair.pair_id}
            for label, pair in cases
        ],
        "runs": runs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "pass" if not mismatches and process_count >= 3 else "fail",
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


__all__ = ["run_repeatability", "select_repeatability_cases"]
