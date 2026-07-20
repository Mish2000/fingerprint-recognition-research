"""Historical parity gate for the restored SIFT geometric baseline.

Re-runs the restored implementation on pairs drawn from the six historical
bundles and requires exact equality of identity, outcome and deterministic
diagnostics.  Nothing here relaxes a comparison: the only fields excluded are
wall-clock timings and the one identity field that provably cannot reproduce
(``implementation_hash``), which is reported explicitly with its reason rather
than silently dropped.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.contract import BENCHMARK_CONTRACT_VERSION
from fingerprint_benchmark.hashing import file_sha256
from fingerprint_benchmark.manifest import read_pair_manifest
from fingerprint_benchmark.runner import _execute_pair, prepare_run_context

from .restored import (
    HISTORICAL_CONFIG_HASH,
    HISTORICAL_IMPLEMENTATION_HASH,
    HISTORICAL_SOURCE_COMMIT,
    RestoredSiftGeometricAdapter,
)


BUNDLE_CONFIG_HASH = HISTORICAL_CONFIG_HASH
DATASETS = ("sd300b", "sd300c")
PROTOCOLS = ("plain_self", "roll_self", "plain_roll")
SAMPLE_STRIDE_COUNT = 10

#: Compared for exact equality.  Every one of these is deterministic.
IDENTITY_COLUMNS = (
    "pair_id",
    "dataset",
    "protocol",
    "subject_id",
    "canonical_finger_position",
    "method",
    "method_version",
    "benchmark_contract_version",
    "result_schema_version",
    "config_hash",
    "manifest_sha256",
    "score_direction",
    "score_semantics",
)

#: Outcome fields compared for exact equality, including raw score as text.
OUTCOME_COLUMNS = ("status", "error_code", "error_message", "raw_score")

#: Reported but not required to match, with a recorded reason.  These are the
#: fields the restoration provably cannot reproduce; see the module docstring
#: and ``docs/sift_geometric_full.md``.
DECLARED_DIFFERENCE_COLUMNS = {
    "implementation_hash": (
        "provenance.py replaced the hardcoded sift_geometric source-hash branch with the generic "
        "implementation_source_paths capability, and contract.py gained the "
        "ImplementationSourceProvider protocol. Both feed implementation_hash. Algorithm identity "
        "is carried by config_hash, which does reproduce exactly."
    ),
}


class ParityError(RuntimeError):
    """Raised when the parity gate cannot be evaluated at all."""


@dataclass(frozen=True)
class PairComparison:
    pair_id: str
    dataset: str
    protocol: str
    matched: bool
    mismatches: list[dict[str, Any]]
    selection_reason: str
    expected: dict[str, Any]
    actual: dict[str, Any]


def is_deterministic_diagnostic_key(key: str) -> bool:
    """Timings are the only excluded diagnostics; everything else is compared."""

    return not key.endswith("_ms")


def deterministic_diagnostics(raw: str) -> dict[str, Any]:
    """Parse a diagnostics cell and drop wall-clock timing keys only."""

    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ParityError("Diagnostics payload must be a JSON object.")
    return {key: value for key, value in payload.items() if is_deterministic_diagnostic_key(key)}


def select_sample_indices(rows: list[Mapping[str, str]]) -> dict[int, str]:
    """Deterministic stride sample plus one of each outcome class if present.

    Stride follows the frozen rule ``index_i = floor(i * N / 10)`` for
    ``i = 0..9``.  Coverage additions are appended only when that class exists in
    the bundle and the stride missed it.
    """

    total = len(rows)
    if total == 0:
        raise ParityError("Historical bundle contains no rows.")
    selected: dict[int, str] = {}
    for i in range(SAMPLE_STRIDE_COUNT):
        index = (i * total) // SAMPLE_STRIDE_COUNT
        selected.setdefault(index, f"stride_{i}")

    def first_matching(predicate: Any) -> int | None:
        for index, row in enumerate(rows):
            if predicate(row):
                return index
        return None

    coverage = (
        (
            "positive_score",
            lambda row: row["status"] == "ok" and row["raw_score"] not in ("", "0.0"),
        ),
        ("zero_score_ok", lambda row: row["status"] == "ok" and row["raw_score"] == "0.0"),
        ("technical_failure", lambda row: row["status"] != "ok"),
    )
    for reason, predicate in coverage:
        if any(predicate(rows[index]) for index in selected):
            continue
        index = first_matching(predicate)
        if index is not None:
            selected.setdefault(index, reason)
    return selected


def _read_bundle_rows(pairs_csv: Path) -> list[dict[str, str]]:
    limit = csv.field_size_limit()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    try:
        with pairs_csv.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    finally:
        csv.field_size_limit(limit)


def _compare_row(
    expected: Mapping[str, str],
    actual: Mapping[str, str],
    *,
    selection_reason: str,
) -> PairComparison:
    mismatches: list[dict[str, Any]] = []
    for column in IDENTITY_COLUMNS + OUTCOME_COLUMNS:
        if expected.get(column, "") != actual.get(column, ""):
            mismatches.append(
                {
                    "field": column,
                    "kind": "column",
                    "expected": expected.get(column, ""),
                    "actual": actual.get(column, ""),
                }
            )

    expected_score, actual_score = expected.get("raw_score", ""), actual.get("raw_score", "")
    if expected_score and actual_score and float(expected_score) != float(actual_score):
        mismatches.append(
            {
                "field": "raw_score_numeric",
                "kind": "numeric",
                "expected": float(expected_score),
                "actual": float(actual_score),
            }
        )

    for column in ("prepare_a_diagnostics", "prepare_b_diagnostics", "compare_diagnostics"):
        expected_diag = deterministic_diagnostics(expected.get(column, ""))
        actual_diag = deterministic_diagnostics(actual.get(column, ""))
        for key in sorted(set(expected_diag) | set(actual_diag)):
            if expected_diag.get(key) != actual_diag.get(key):
                mismatches.append(
                    {
                        "field": f"{column}.{key}",
                        "kind": "diagnostic",
                        "expected": expected_diag.get(key),
                        "actual": actual_diag.get(key),
                    }
                )

    compare_expected = deterministic_diagnostics(expected.get("compare_diagnostics", ""))
    compare_actual = deterministic_diagnostics(actual.get("compare_diagnostics", ""))
    summary_keys = (
        "matches_submitted_to_geometry",
        "geometric_inlier_count",
        "geometry_failure_reason",
    )
    return PairComparison(
        pair_id=expected["pair_id"],
        dataset=expected["dataset"],
        protocol=expected["protocol"],
        matched=not mismatches,
        mismatches=mismatches,
        selection_reason=selection_reason,
        expected={
            **{column: expected.get(column, "") for column in OUTCOME_COLUMNS},
            **{key: compare_expected.get(key) for key in summary_keys},
        },
        actual={
            **{column: actual.get(column, "") for column in OUTCOME_COLUMNS},
            **{key: compare_actual.get(key) for key in summary_keys},
        },
    )


def run_bundle_parity(
    *,
    dataset: str,
    protocol: str,
    historical_results_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Replay one historical bundle's sample through the restored code."""

    bundle = (
        historical_results_root
        / dataset
        / protocol
        / "sift_geometric"
        / BENCHMARK_CONTRACT_VERSION
        / BUNDLE_CONFIG_HASH
    )
    pairs_csv = bundle / "pairs.csv"
    metadata_path = bundle / "run_metadata.json"
    if not pairs_csv.is_file() or not metadata_path.is_file():
        raise ParityError(f"Historical bundle is incomplete: {bundle}")

    historical_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        historical_metadata.get("method") != "sift_geometric"
        or historical_metadata.get("method_version") != "sift-geometric-v1"
    ):
        raise ParityError(f"Bundle is not a sift-geometric-v1 oracle: {bundle}")

    manifest_path = repository_root / "protocols" / dataset / f"{protocol}.csv"
    manifest_sha256 = file_sha256(manifest_path)
    if historical_metadata["manifest"]["sha256"] != manifest_sha256:
        raise ParityError(
            f"Manifest {manifest_path} changed since the historical run; parity is not evaluable."
        )

    historical_rows = _read_bundle_rows(pairs_csv)
    selection = select_sample_indices(historical_rows)
    pairs_by_id = {pair.pair_id: pair for pair in read_pair_manifest(manifest_path)}

    adapter = RestoredSiftGeometricAdapter()
    try:
        context = prepare_run_context(
            expected_dataset=dataset,
            expected_protocol=protocol,
            manifest_path=manifest_path,
            adapter=adapter,
            results_root=repository_root / "results",
        )
        if context.spec.config_hash != BUNDLE_CONFIG_HASH:
            raise ParityError(
                f"Restored config_hash {context.spec.config_hash} does not match the historical "
                f"bundle identity {BUNDLE_CONFIG_HASH}."
            )
        comparisons: list[PairComparison] = []
        for index in sorted(selection):
            expected_row = historical_rows[index]
            pair = pairs_by_id.get(expected_row["pair_id"])
            if pair is None:
                raise ParityError(f"Manifest has no pair {expected_row['pair_id']!r}.")
            actual_row = _execute_pair(
                pair,
                adapter,
                run_spec=context.spec,
                method_metadata=context.method_metadata,
            )
            comparisons.append(
                _compare_row(expected_row, actual_row, selection_reason=selection[index])
            )
    finally:
        adapter.close()

    return {
        "dataset": dataset,
        "protocol": protocol,
        "historical_bundle_path": str(bundle),
        "historical_pairs_csv_sha256": file_sha256(pairs_csv),
        "historical_run_metadata_sha256": file_sha256(metadata_path),
        "historical_row_count": len(historical_rows),
        "historical_config_hash": historical_metadata["config_hash"],
        "restored_config_hash": context.spec.config_hash,
        "config_hash_matches": historical_metadata["config_hash"] == context.spec.config_hash,
        "historical_implementation_hash": historical_metadata["implementation_hash"],
        "restored_implementation_hash": context.spec.implementation_hash,
        "manifest_sha256": manifest_sha256,
        "pair_count": len(comparisons),
        "matched_pair_count": sum(1 for item in comparisons if item.matched),
        "status": "pass" if all(item.matched for item in comparisons) else "fail",
        "pairs": [
            {
                "pair_id": item.pair_id,
                "selection_reason": item.selection_reason,
                "matched": item.matched,
                "expected": item.expected,
                "actual": item.actual,
                "mismatches": item.mismatches,
            }
            for item in comparisons
        ],
    }


def run_parity(
    *,
    historical_results_root: Path,
    repository_root: Path,
    output_path: Path,
    historical_source_commit: str = HISTORICAL_SOURCE_COMMIT,
    current_commit: str | None = None,
) -> dict[str, Any]:
    """Run the full six-bundle parity gate and persist the report."""

    bundles = [
        run_bundle_parity(
            dataset=dataset,
            protocol=protocol,
            historical_results_root=historical_results_root,
            repository_root=repository_root,
        )
        for dataset in DATASETS
        for protocol in PROTOCOLS
    ]
    total_pairs = sum(bundle["pair_count"] for bundle in bundles)
    mismatches = [
        {"dataset": bundle["dataset"], "protocol": bundle["protocol"], **pair}
        for bundle in bundles
        for pair in bundle["pairs"]
        if not pair["matched"]
    ]
    report = {
        "parity_schema": "sift-geometric-restoration-parity-v1",
        "method": "sift_geometric",
        "method_version": "sift-geometric-v1",
        "historical_source_commit": historical_source_commit,
        "current_commit_before_restoration": current_commit,
        "historical_implementation_hash": HISTORICAL_IMPLEMENTATION_HASH,
        "declared_differences": DECLARED_DIFFERENCE_COLUMNS,
        "comparison_policy": {
            "tolerance": "none; exact equality required",
            "excluded_fields": [
                "wall-clock timing columns",
                "diagnostics keys ending in _ms",
                "implementation_hash (declared difference with recorded reason)",
            ],
            "sample_rule": "index_i = floor(i * N / 10) for i in 0..9, plus one pair per outcome class",
        },
        "runtime": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
        },
        "bundle_count": len(bundles),
        "total_pair_count": total_pairs,
        "matched_pair_count": sum(bundle["matched_pair_count"] for bundle in bundles),
        "mismatch_count": len(mismatches),
        "config_hash_reproduced_in_all_bundles": all(
            bundle["config_hash_matches"] for bundle in bundles
        ),
        "status": "pass" if not mismatches and total_pairs >= 60 else "fail",
        "mismatches": mismatches,
        "bundles": bundles,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    output_path.write_text(payload, encoding="utf-8", newline="\n")
    report["report_path"] = str(output_path)
    report["report_sha256"] = file_sha256(output_path)
    return report


def repeatability_check(
    *,
    pair_specs: Iterable[tuple[str, str, str]],
    repository_root: Path,
    repeats: int = 1,
) -> list[dict[str, Any]]:
    """Re-run named pairs in this process; call from separate processes to prove
    cross-process repeatability."""

    results: list[dict[str, Any]] = []
    for dataset, protocol, pair_id in pair_specs:
        manifest_path = repository_root / "protocols" / dataset / f"{protocol}.csv"
        pairs_by_id = {pair.pair_id: pair for pair in read_pair_manifest(manifest_path)}
        pair = pairs_by_id[pair_id]
        adapter = RestoredSiftGeometricAdapter()
        try:
            context = prepare_run_context(
                expected_dataset=dataset,
                expected_protocol=protocol,
                manifest_path=manifest_path,
                adapter=adapter,
                results_root=repository_root / "results",
            )
            for _ in range(repeats):
                row = _execute_pair(
                    pair,
                    adapter,
                    run_spec=context.spec,
                    method_metadata=context.method_metadata,
                )
                results.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "pair_id": pair_id,
                        "status": row["status"],
                        "error_code": row["error_code"],
                        "raw_score": row["raw_score"],
                        "compare_diagnostics": deterministic_diagnostics(
                            row["compare_diagnostics"]
                        ),
                        "prepare_a_diagnostics": deterministic_diagnostics(
                            row["prepare_a_diagnostics"]
                        ),
                        "prepare_b_diagnostics": deterministic_diagnostics(
                            row["prepare_b_diagnostics"]
                        ),
                    }
                )
        finally:
            adapter.close()
    return results
