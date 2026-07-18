"""Exact next-subject SourceAFIS impostor experiment requested by the supervisor.

The existing per-dataset self-accepted genuine protocol is immutable input.  This
module derives one deterministic wrong PLAIN-to-ROLL pair per genuine identity,
runs only those two manifests through the frozen benchmark-v2 cold-pair runner,
and emits the deliberately narrow supervisor report.

The research manifest retains both subjects and all pairing provenance.  A
lossless benchmark projection is stored beside it because benchmark-v2's frozen
manifest contract has ten columns and must not be changed (changing it would
also change the frozen implementation hash).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT
from fingerprint_data_discovery.canonical_fingers import canonical_finger_position

from .bundle import create_candidate_directory, discard_candidate_directory
from .contract import BENCHMARK_CONTRACT_VERSION, BenchmarkRunSpec
from .derived_protocol import (
    DATASETS,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SERVICE_URL,
    DEFAULT_SIDECAR_JAR,
    DEFAULT_PROTOCOL_ROOT as GENUINE_PROTOCOL_RELATIVE_ROOT,
    DEFAULT_RUN_ROOT as GENUINE_RUN_RELATIVE_ROOT,
    INCLUDED_COLUMNS,
    METHOD,
    SOURCEAFIS_THRESHOLD,
    DerivedProtocolError,
    PrimaryBundle,
    _load_derived_bundle,
    _publish_immutable_bytes,
    _publish_immutable_candidate,
    _startup_dict,
    load_and_validate_protocol as load_genuine_protocol,
    load_primary_bundle,
)
from .hashing import canonical_json_bytes, file_sha256
from .io import write_csv_atomic, write_json_atomic
from .manifest import MANIFEST_COLUMNS, PairRecord, read_pair_manifest
from .runner import (
    METADATA_FILENAME,
    RESULT_FILENAME,
    prepare_run_context,
    read_result_rows,
    run_benchmark_manifest,
    validate_result_bundle,
)
from .shared_accuracy_integrity import (
    capture_snapshot,
    compare_snapshot,
    publish_immutable_bytes,
    publish_immutable_json,
    read_snapshot,
)
from .sourceafis_adapter import SourceAfisAdapter
from .sourceafis_client import SourceAfisSidecarClient, validate_health
from .sourceafis_sidecar import ManagedSourceAfisSidecar


NEGATIVE_PROTOCOL = "plain_roll_next_subject_impostor"
NEGATIVE_NAMESPACE = "negative_next_subject"
NEGATIVE_SCHEMA_VERSION = "sourceafis-next-subject-impostor-v1"
NEGATIVE_RESULT_SCHEMA_VERSION = "sourceafis-next-subject-result-summary-v1"
REPORT_SCHEMA_VERSION = "sourceafis-exact-requested-supervisor-report-v1"
SHARED_ACCURACY_NAMESPACE = "sourceafis_sift_v1"

DEFAULT_NEGATIVE_PROTOCOL_ROOT = (
    Path("results")
    / "derived_protocols"
    / "sourceafis_per_dataset_self_accept_t40_v1"
    / NEGATIVE_NAMESPACE
)
DEFAULT_NEGATIVE_RUN_ROOT = (
    Path("results")
    / "derived_protocol_runs"
    / "sourceafis_per_dataset_self_accept_t40_v1"
    / NEGATIVE_NAMESPACE
)
DEFAULT_SUPERVISOR_ROOT = (
    Path("results")
    / "supervisor_reports"
    / "sourceafis_exact_requested_report_v1"
)
DEFAULT_SHARED_ACCURACY_ROOT = Path("results") / "shared_accuracy" / SHARED_ACCURACY_NAMESPACE

ADVISORY_EXPECTED_NEGATIVE_COUNTS = {"sd300b": 8593, "sd300c": 8614}
ADVISORY_EXPECTED_SINGLE_FINGER_COUNTS = {"plain_self": 8788, "roll_self": 8871}

NEGATIVE_MANIFEST_COLUMNS = [
    "pair_id",
    "negative_pair_id",
    "dataset",
    "protocol",
    "pair_label",
    "subject_id",
    "canonical_finger_position",
    "subject_id_a",
    "subject_id_b",
    "ppi",
    "path_a",
    "path_b",
    "raw_frgp_a",
    "raw_frgp_b",
    "source_plain_pair_id",
    "source_roll_pair_id",
    "plain_group_index",
    "roll_group_index",
    "shift",
]

PAIRING_MAP_COLUMNS = [
    "negative_pair_id",
    "dataset",
    "protocol",
    "pair_label",
    "canonical_finger_position",
    "subject_id_a",
    "subject_id_b",
    "ppi",
    "path_a",
    "path_b",
    "raw_frgp_a",
    "raw_frgp_b",
    "source_plain_pair_id",
    "source_roll_pair_id",
    "plain_group_index",
    "roll_group_index",
    "shift",
]

NEGATIVE_SUMMARY_COLUMNS = [
    "dataset",
    "protocol",
    "pair_label",
    "shift",
    "pair_count",
    "source_genuine_pair_count",
    "canonical_group_count",
    "different_subject_count",
    "same_canonical_finger_count",
    "unique_plain_count",
    "unique_roll_count",
    "duplicate_negative_pair_count",
    "genuine_contamination_count",
    "same_image_both_sides_count",
    "advisory_expected_count",
    "advisory_count_delta",
    "manifest_sha256",
    "benchmark_manifest_sha256",
    "pairing_map_sha256",
]

REUSE_DETAIL_COLUMNS = [
    "negative_pair_id",
    "dataset",
    "canonical_finger_position",
    "subject_id_a",
    "subject_id_b",
    "ppi",
    "path_a",
    "path_b",
    "shared_split",
    "shared_accuracy_pair_id",
    "raw_score",
    "status",
    "error_code",
    "prepare_a_diagnostics_json",
    "prepare_b_diagnostics_json",
    "compare_diagnostics_json",
    "shared_score_file",
]

NEGATIVE_RESULT_COLUMNS = [
    "dataset",
    "threshold",
    "total_wrong_pairs",
    "status_ok",
    "failures",
    "false_matches",
    "correct_non_matches",
    "false_match_percentage",
    "correct_non_match_percentage",
    "score_zero",
    "positive_below_threshold",
    "mean_score",
    "median_score",
    "min_score",
    "max_score",
    "mean_method_compare_ms",
    "median_method_compare_ms",
    "p95_method_compare_ms",
    "mean_total_ms",
    "median_total_ms",
    "p95_total_ms",
]

REPRO_DETAIL_COLUMNS = [
    "dataset",
    "negative_pair_id",
    "shared_accuracy_pair_id",
    "shared_raw_score",
    "cold_pair_raw_score",
    "raw_score_text_equal",
    "raw_score_abs_delta",
    "status_equal",
    "error_code_equal",
    "prepare_a_diagnostics_equal",
    "prepare_b_diagnostics_equal",
    "compare_diagnostics_equal",
    "exact_reproducibility",
]

REPORT_SECTION_TITLES = (
    "Single-finger record counts",
    "PLAIN self-comparisons",
    "ROLL self-comparisons",
    "Cleaned genuine PLAIN-ROLL pairs",
    "SourceAFIS results on genuine pairs",
    "SourceAFIS results on wrong pairs",
)


class NegativeProtocolError(ValueError):
    """Raised when an exact requested-protocol invariant cannot be proven."""


@dataclass(frozen=True)
class SharedReuseIdentity:
    method: str
    implementation_hash: str
    config_hash: str
    sourceafis_version: str
    sidecar_jar_sha256: str


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return str(value)


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(expected_columns):
                raise NegativeProtocolError(
                    f"CSV schema mismatch in {path}: expected {list(expected_columns)}, got {reader.fieldnames}."
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise NegativeProtocolError(f"Cannot read CSV {path}: {exc}") from exc
    if any(None in row for row in rows):
        raise NegativeProtocolError(f"CSV has unnamed extra values: {path}")
    return rows


def _require_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise NegativeProtocolError(f"Unsupported dataset: {dataset}")


def _pair_sort_key(pair: PairRecord) -> tuple[str, str]:
    return pair.subject_id, pair.pair_id


def build_next_subject_rows(
    dataset: str,
    genuine_pairs: Sequence[PairRecord],
) -> list[dict[str, Any]]:
    """Build a stable circular shift by one inside every canonical finger."""

    _require_dataset(dataset)
    if not genuine_pairs:
        raise NegativeProtocolError(f"Genuine population is empty for {dataset}.")
    groups: dict[int, list[PairRecord]] = defaultdict(list)
    seen_identity: set[tuple[str, int]] = set()
    for pair in genuine_pairs:
        if pair.dataset != dataset or pair.protocol != "plain_roll":
            raise NegativeProtocolError(
                f"Wrong genuine source record {pair.pair_id}: {pair.dataset}/{pair.protocol}."
            )
        identity = (pair.subject_id, pair.canonical_finger_position)
        if identity in seen_identity:
            raise NegativeProtocolError(f"Multiple captures for single-finger identity {identity}.")
        seen_identity.add(identity)
        groups[pair.canonical_finger_position].append(pair)

    rows: list[dict[str, Any]] = []
    for finger in sorted(groups):
        ordered = sorted(groups[finger], key=_pair_sort_key)
        if len(ordered) < 2:
            raise NegativeProtocolError(
                f"Canonical finger {finger} in {dataset} cannot be circularly shifted across subjects."
            )
        subjects = [pair.subject_id for pair in ordered]
        if len(subjects) != len(set(subjects)):
            raise NegativeProtocolError(
                f"Canonical finger {finger} in {dataset} contains repeated subjects."
            )
        for plain_index, plain in enumerate(ordered):
            roll_index = (plain_index + 1) % len(ordered)
            roll = ordered[roll_index]
            negative_pair_id = (
                f"{dataset}_{NEGATIVE_PROTOCOL}_{finger:02d}_"
                f"{plain.subject_id}_{roll.subject_id}"
            )
            rows.append(
                {
                    "pair_id": negative_pair_id,
                    "negative_pair_id": negative_pair_id,
                    "dataset": dataset,
                    "protocol": NEGATIVE_PROTOCOL,
                    "pair_label": "impostor",
                    "subject_id": plain.subject_id,
                    "canonical_finger_position": finger,
                    "subject_id_a": plain.subject_id,
                    "subject_id_b": roll.subject_id,
                    "ppi": plain.ppi,
                    "path_a": str(plain.path_a),
                    "path_b": str(roll.path_b),
                    "raw_frgp_a": plain.raw_frgp_a,
                    "raw_frgp_b": roll.raw_frgp_b,
                    "source_plain_pair_id": plain.pair_id,
                    "source_roll_pair_id": roll.pair_id,
                    "plain_group_index": plain_index,
                    "roll_group_index": roll_index,
                    "shift": 1,
                }
            )
    return rows


def benchmark_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project the research manifest losslessly onto frozen benchmark-v2 columns."""

    return [
        {
            "pair_id": row["negative_pair_id"],
            "dataset": row["dataset"],
            "protocol": row["protocol"],
            "subject_id": row["subject_id_a"],
            "canonical_finger_position": int(row["canonical_finger_position"]),
            "ppi": int(row["ppi"]),
            "raw_frgp_a": int(row["raw_frgp_a"]),
            "raw_frgp_b": int(row["raw_frgp_b"]),
            "path_a": row["path_a"],
            "path_b": row["path_b"],
        }
        for row in rows
    ]


def validate_next_subject_rows(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    genuine_pairs: Sequence[PairRecord],
) -> dict[str, Any]:
    """Prove every construction invariant against the frozen genuine population."""

    _require_dataset(dataset)
    genuine_by_id = {pair.pair_id: pair for pair in genuine_pairs}
    if len(genuine_by_id) != len(genuine_pairs):
        raise NegativeProtocolError(f"Duplicate genuine source pair ids in {dataset}.")
    expected = build_next_subject_rows(dataset, genuine_pairs)
    normalized = [{name: _csv_value(row.get(name)) for name in NEGATIVE_MANIFEST_COLUMNS} for row in rows]
    expected_normalized = [
        {name: _csv_value(row.get(name)) for name in NEGATIVE_MANIFEST_COLUMNS} for row in expected
    ]
    if normalized != expected_normalized:
        raise NegativeProtocolError(f"Negative manifest is not the exact deterministic shift for {dataset}.")

    pair_ids: set[str] = set()
    directed_paths: set[tuple[str, str]] = set()
    plain_sources: list[str] = []
    roll_sources: list[str] = []
    contamination = 0
    same_image = 0
    for row in rows:
        pair_id = str(row["negative_pair_id"])
        if pair_id in pair_ids:
            raise NegativeProtocolError(f"Duplicate negative_pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if str(row["dataset"]) != dataset or str(row["protocol"]) != NEGATIVE_PROTOCOL:
            raise NegativeProtocolError(f"Dataset/protocol mismatch in {pair_id}.")
        if str(row["pair_label"]) != "impostor" or int(row["shift"]) != 1:
            raise NegativeProtocolError(f"Pair label/shift mismatch in {pair_id}.")
        if str(row["subject_id_a"]) == str(row["subject_id_b"]):
            contamination += 1
        source_a = genuine_by_id[str(row["source_plain_pair_id"])]
        source_b = genuine_by_id[str(row["source_roll_pair_id"])]
        finger = int(row["canonical_finger_position"])
        if source_a.canonical_finger_position != finger or source_b.canonical_finger_position != finger:
            raise NegativeProtocolError(f"Cross-finger pair in {pair_id}.")
        if str(source_a.path_a) != str(row["path_a"]) or str(source_b.path_b) != str(row["path_b"]):
            raise NegativeProtocolError(f"Source path provenance mismatch in {pair_id}.")
        if source_a.ppi != int(row["ppi"]) or source_b.ppi != int(row["ppi"]):
            raise NegativeProtocolError(f"PPI mismatch in {pair_id}.")
        path_key = (str(row["path_a"]), str(row["path_b"]))
        if path_key in directed_paths:
            raise NegativeProtocolError(f"Duplicate negative path pair in {pair_id}.")
        directed_paths.add(path_key)
        if path_key[0].casefold() == path_key[1].casefold():
            same_image += 1
        plain_sources.append(source_a.pair_id)
        roll_sources.append(source_b.pair_id)

    genuine_ids = set(genuine_by_id)
    if len(rows) != len(genuine_pairs):
        raise NegativeProtocolError(f"Negative/genuine count mismatch for {dataset}.")
    if set(plain_sources) != genuine_ids or len(plain_sources) != len(set(plain_sources)):
        raise NegativeProtocolError(f"Not every PLAIN source is used exactly once for {dataset}.")
    if set(roll_sources) != genuine_ids or len(roll_sources) != len(set(roll_sources)):
        raise NegativeProtocolError(f"Not every ROLL source is used exactly once for {dataset}.")
    if contamination or same_image:
        raise NegativeProtocolError(
            f"Genuine/image contamination for {dataset}: subjects={contamination}, images={same_image}."
        )
    return {
        "dataset": dataset,
        "protocol": NEGATIVE_PROTOCOL,
        "pair_label": "impostor",
        "shift": 1,
        "pair_count": len(rows),
        "source_genuine_pair_count": len(genuine_pairs),
        "canonical_group_count": len({pair.canonical_finger_position for pair in genuine_pairs}),
        "different_subject_count": len(rows),
        "same_canonical_finger_count": len(rows),
        "unique_plain_count": len(set(plain_sources)),
        "unique_roll_count": len(set(roll_sources)),
        "duplicate_negative_pair_count": 0,
        "genuine_contamination_count": 0,
        "same_image_both_sides_count": 0,
        "only_single_finger_eligible_inputs": True,
        "stable_subject_sorting": True,
        "circular_wrap_around": True,
        "datasets_constructed_independently": True,
    }


def _pairing_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{name: row[name] for name in PAIRING_MAP_COLUMNS} for row in rows]


def prepare_negative_protocol(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    negative_protocol_root: Path | None = None,
) -> dict[str, Any]:
    """Derive and immutably publish both independent negative manifests."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    final_root = (negative_protocol_root or project_root / DEFAULT_NEGATIVE_PROTOCOL_ROOT).resolve()
    genuine_root = (project_root / GENUINE_PROTOCOL_RELATIVE_ROOT).resolve()
    genuine = load_genuine_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=genuine_root,
        validate_sources=True,
    )
    candidate = create_candidate_directory(final_root)
    try:
        summaries: list[dict[str, Any]] = []
        for dataset in DATASETS:
            source_manifest = genuine_root / dataset / "plain_roll.csv"
            genuine_pairs = read_pair_manifest(source_manifest)
            rows = build_next_subject_rows(dataset, genuine_pairs)
            validation = validate_next_subject_rows(dataset, rows, genuine_pairs)
            dataset_dir = candidate / dataset
            manifest_path = dataset_dir / f"{NEGATIVE_PROTOCOL}.csv"
            benchmark_path = dataset_dir / "benchmark_manifest.csv"
            pairing_path = dataset_dir / "pairing_map.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            write_csv_atomic(rows, manifest_path, NEGATIVE_MANIFEST_COLUMNS)
            write_csv_atomic(benchmark_projection(rows), benchmark_path, MANIFEST_COLUMNS)
            write_csv_atomic(_pairing_rows(rows), pairing_path, PAIRING_MAP_COLUMNS)
            benchmark_pairs = read_pair_manifest(benchmark_path)
            if benchmark_projection(rows) != [
                {
                    "pair_id": pair.pair_id,
                    "dataset": pair.dataset,
                    "protocol": pair.protocol,
                    "subject_id": pair.subject_id,
                    "canonical_finger_position": pair.canonical_finger_position,
                    "ppi": pair.ppi,
                    "raw_frgp_a": pair.raw_frgp_a,
                    "raw_frgp_b": pair.raw_frgp_b,
                    "path_a": str(pair.path_a),
                    "path_b": str(pair.path_b),
                }
                for pair in benchmark_pairs
            ]:
                raise NegativeProtocolError(f"Benchmark projection is not lossless for {dataset}.")
            advisory = ADVISORY_EXPECTED_NEGATIVE_COUNTS[dataset]
            summary = {
                **validation,
                "advisory_expected_count": advisory,
                "advisory_count_delta": len(rows) - advisory,
                "source_genuine_manifest": str(source_manifest),
                "source_genuine_manifest_sha256": file_sha256(source_manifest),
                "manifest_relative_path": f"{dataset}/{NEGATIVE_PROTOCOL}.csv",
                "benchmark_manifest_relative_path": f"{dataset}/benchmark_manifest.csv",
                "pairing_map_relative_path": f"{dataset}/pairing_map.csv",
                "manifest_sha256": file_sha256(manifest_path),
                "benchmark_manifest_sha256": file_sha256(benchmark_path),
                "pairing_map_sha256": file_sha256(pairing_path),
            }
            summaries.append(summary)

        payload = {
            "schema_version": NEGATIVE_SCHEMA_VERSION,
            "namespace": NEGATIVE_NAMESPACE,
            "source_namespace": genuine["summary"]["namespace"],
            "source_protocol_summary_sha256": genuine["protocol_summary_sha256"],
            "method": METHOD,
            "method_version": "3.18.1",
            "threshold": SOURCEAFIS_THRESHOLD,
            "construction": "next subject within the same canonical finger position, circular shift by one",
            "grouping_unit": "canonical_finger_position",
            "stable_sort": ["subject_id", "pair_id"],
            "plain_group_index_base": 0,
            "roll_group_index_base": 0,
            "datasets_constructed_independently": True,
            "research_manifest_and_frozen_benchmark_projection": True,
            "datasets": summaries,
        }
        write_json_atomic(payload, candidate / "negative_protocol_summary.json")
        write_csv_atomic(
            [
                {column: summary[column] for column in NEGATIVE_SUMMARY_COLUMNS}
                for summary in summaries
            ],
            candidate / "negative_protocol_summary.csv",
            NEGATIVE_SUMMARY_COLUMNS,
        )
        _publish_immutable_candidate(candidate, final_root)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)
    return load_and_validate_negative_protocol(
        project_root=project_root,
        data_root=data_root,
        negative_protocol_root=final_root,
    )


def load_and_validate_negative_protocol(
    *,
    project_root: Path,
    data_root: Path,
    negative_protocol_root: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    root = negative_protocol_root.resolve()
    summary_path = root / "negative_protocol_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NegativeProtocolError(f"Cannot read negative protocol summary: {exc}") from exc
    if summary.get("schema_version") != NEGATIVE_SCHEMA_VERSION:
        raise NegativeProtocolError("Negative protocol schema mismatch.")
    if summary.get("namespace") != NEGATIVE_NAMESPACE or summary.get("threshold") != SOURCEAFIS_THRESHOLD:
        raise NegativeProtocolError("Negative protocol namespace/threshold mismatch.")
    genuine_root = project_root / GENUINE_PROTOCOL_RELATIVE_ROOT
    genuine = load_genuine_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=genuine_root,
        validate_sources=True,
    )
    if summary.get("source_protocol_summary_sha256") != genuine["protocol_summary_sha256"]:
        raise NegativeProtocolError("Frozen genuine protocol summary changed.")

    reports: list[dict[str, Any]] = []
    by_dataset = {item["dataset"]: item for item in summary["datasets"]}
    if set(by_dataset) != set(DATASETS):
        raise NegativeProtocolError("Negative protocol must contain exactly SD300b and SD300c.")
    for dataset in DATASETS:
        frozen = by_dataset[dataset]
        public_path = root / dataset / f"{NEGATIVE_PROTOCOL}.csv"
        benchmark_path = root / dataset / "benchmark_manifest.csv"
        pairing_path = root / dataset / "pairing_map.csv"
        for path, key in (
            (public_path, "manifest_sha256"),
            (benchmark_path, "benchmark_manifest_sha256"),
            (pairing_path, "pairing_map_sha256"),
        ):
            if file_sha256(path) != frozen[key]:
                raise NegativeProtocolError(f"Frozen negative artifact changed: {path}")
        rows = _read_csv(public_path, NEGATIVE_MANIFEST_COLUMNS)
        pairing = _read_csv(pairing_path, PAIRING_MAP_COLUMNS)
        if pairing != [{name: row[name] for name in PAIRING_MAP_COLUMNS} for row in rows]:
            raise NegativeProtocolError(f"Pairing map differs from research manifest for {dataset}.")
        genuine_pairs = read_pair_manifest(genuine_root / dataset / "plain_roll.csv")
        validation = validate_next_subject_rows(dataset, rows, genuine_pairs)
        projected = benchmark_projection(rows)
        benchmark_pairs = read_pair_manifest(benchmark_path)
        observed_projection = [
            {
                "pair_id": pair.pair_id,
                "dataset": pair.dataset,
                "protocol": pair.protocol,
                "subject_id": pair.subject_id,
                "canonical_finger_position": pair.canonical_finger_position,
                "ppi": pair.ppi,
                "raw_frgp_a": pair.raw_frgp_a,
                "raw_frgp_b": pair.raw_frgp_b,
                "path_a": str(pair.path_a),
                "path_b": str(pair.path_b),
            }
            for pair in benchmark_pairs
        ]
        if projected != observed_projection:
            raise NegativeProtocolError(f"Benchmark projection mismatch for {dataset}.")
        if len(rows) != int(frozen["pair_count"]):
            raise NegativeProtocolError(f"Frozen pair count mismatch for {dataset}.")
        reports.append({**validation, "manifest_path": str(public_path), "benchmark_manifest_path": str(benchmark_path)})
    return {
        "root": str(root),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "summary": summary,
        "datasets": reports,
    }


def make_negative_manifest_validator(
    *,
    negative_protocol_root: Path,
    dataset: str,
    expected_manifest_sha256: str,
):
    public_path = negative_protocol_root / dataset / f"{NEGATIVE_PROTOCOL}.csv"
    benchmark_path = negative_protocol_root / dataset / "benchmark_manifest.csv"

    def validate(manifest_path: Path, data_root: Path) -> dict[str, Any]:
        if manifest_path.resolve() != benchmark_path.resolve():
            raise NegativeProtocolError(f"Unexpected execution manifest: {manifest_path}")
        if file_sha256(manifest_path) != expected_manifest_sha256:
            raise NegativeProtocolError(f"Execution manifest hash mismatch for {dataset}.")
        rows = _read_csv(public_path, NEGATIVE_MANIFEST_COLUMNS)
        pairs = read_pair_manifest(manifest_path)
        if len(rows) != len(pairs):
            raise NegativeProtocolError(f"Research/execution count mismatch for {dataset}.")
        expected_projection = benchmark_projection(rows)
        actual_projection = [
            {
                "pair_id": pair.pair_id,
                "dataset": pair.dataset,
                "protocol": pair.protocol,
                "subject_id": pair.subject_id,
                "canonical_finger_position": pair.canonical_finger_position,
                "ppi": pair.ppi,
                "raw_frgp_a": pair.raw_frgp_a,
                "raw_frgp_b": pair.raw_frgp_b,
                "path_a": str(pair.path_a),
                "path_b": str(pair.path_b),
            }
            for pair in pairs
        ]
        if expected_projection != actual_projection:
            raise NegativeProtocolError(f"Execution projection differs for {dataset}.")
        resolved_data_root = data_root.resolve()
        for pair in pairs:
            for image in (pair.path_a, pair.path_b):
                resolved = image.resolve()
                try:
                    resolved.relative_to(resolved_data_root)
                except ValueError as exc:
                    raise NegativeProtocolError(f"Image escapes read-only data root: {image}") from exc
                if not resolved.is_file():
                    raise NegativeProtocolError(f"Image is missing: {image}")
        return {
            "schema_version": NEGATIVE_SCHEMA_VERSION,
            "dataset": dataset,
            "pair_count": len(pairs),
            "research_manifest_sha256": file_sha256(public_path),
            "benchmark_manifest_sha256": file_sha256(manifest_path),
            "projection_exact": True,
            "all_images_under_read_only_data_root": True,
            "all_images_exist": True,
        }

    return validate


def _shared_reuse_identity(shared_root: Path) -> tuple[SharedReuseIdentity, dict[str, Any]]:
    path = shared_root / "provenance" / "method_runtime.json"
    runtime = json.loads(path.read_text(encoding="utf-8"))
    sourceafis = runtime["sourceafis"]
    identity = SharedReuseIdentity(
        method=sourceafis["method"],
        implementation_hash=sourceafis["implementation_hash"],
        config_hash=sourceafis["frozen_config_hash"],
        sourceafis_version=sourceafis["method_version"],
        sidecar_jar_sha256=sourceafis["sidecar_jar_sha256"],
    )
    if identity != SharedReuseIdentity(
        method="sourceafis",
        implementation_hash=sourceafis["implementation_hash_components"] and sourceafis["implementation_hash"],
        config_hash=sourceafis["frozen_config_hash"],
        sourceafis_version="3.18.1",
        sidecar_jar_sha256=sourceafis["implementation_hash_components"]["sidecar_jar_sha256"],
    ):
        raise NegativeProtocolError("Shared SourceAFIS runtime provenance is internally inconsistent.")
    return identity, runtime


def _reuse_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, str, str, int]:
    return (
        str(row["dataset"]),
        int(row["ppi"]),
        str(row["path_a"]),
        str(row["path_b"]),
        str(row["subject_id_a"]),
        str(row["subject_id_b"]),
        int(row["canonical_finger_position"]),
    )


def lookup_exact_reuse(
    negative_rows: Sequence[Mapping[str, Any]],
    shared_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return exact path/subject/finger reuse matches and missing count."""

    index: dict[tuple[str, int, str, str, str, str, int], Mapping[str, Any]] = {}
    for row in shared_rows:
        key = _reuse_key(row)
        if key in index:
            raise NegativeProtocolError(f"Duplicate exact shared-accuracy reuse key: {key}")
        index[key] = row
    overlaps: list[dict[str, Any]] = []
    for row in negative_rows:
        found = index.get(_reuse_key(row))
        if found is None:
            continue
        overlaps.append(
            {
                "negative_pair_id": row["negative_pair_id"],
                "dataset": row["dataset"],
                "canonical_finger_position": row["canonical_finger_position"],
                "subject_id_a": row["subject_id_a"],
                "subject_id_b": row["subject_id_b"],
                "ppi": row["ppi"],
                "path_a": row["path_a"],
                "path_b": row["path_b"],
                "shared_split": found["split"],
                "shared_accuracy_pair_id": found["accuracy_pair_id"],
                "raw_score": found["raw_score"],
                "status": found["status"],
                "error_code": found["error_code"],
                "prepare_a_diagnostics_json": found["prepare_a_diagnostics_json"],
                "prepare_b_diagnostics_json": found["prepare_b_diagnostics_json"],
                "compare_diagnostics_json": found["compare_diagnostics_json"],
                "shared_score_file": found.get("shared_score_file", ""),
            }
        )
    return overlaps, len(negative_rows) - len(overlaps)


def _load_compatible_shared_rows(
    *,
    shared_root: Path,
    dataset: str,
    identity: SharedReuseIdentity,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    expected_header: list[str] | None = None
    for split in ("development", "evaluation"):
        directory = shared_root / "scores" / "sourceafis" / dataset / split
        metadata_path = directory / "run_metadata.json"
        score_path = directory / "impostor.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "method": identity.method,
            "implementation_hash": identity.implementation_hash,
            "frozen_config_hash": identity.config_hash,
            "method_version": identity.sourceafis_version,
            "dataset": dataset,
            "split": split,
        }
        actual_metadata = {
            "method": metadata["method"],
            "implementation_hash": metadata["implementation_hash"],
            "frozen_config_hash": metadata["frozen_config_hash"],
            "method_version": "3.18.1",
            "dataset": metadata["dataset"],
            "split": metadata["split"],
        }
        if actual_metadata != expected_metadata:
            raise NegativeProtocolError(
                f"Shared score provenance mismatch in {metadata_path}: {actual_metadata} != {expected_metadata}"
            )
        expected_hash = metadata["score_files"]["impostor"]["sha256"]
        if file_sha256(score_path) != expected_hash:
            raise NegativeProtocolError(f"Shared impostor score hash mismatch: {score_path}")
        with score_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if expected_header is None:
                expected_header = list(reader.fieldnames or [])
            elif reader.fieldnames != expected_header:
                raise NegativeProtocolError("Shared SourceAFIS score schemas differ across splits.")
            for row in reader:
                if row["pair_label"] != "impostor":
                    raise NegativeProtocolError(f"Non-impostor row in {score_path}.")
                exact_row_identity = (
                    row["method"] == identity.method
                    and row["method_version"] == identity.sourceafis_version
                    and row["implementation_hash"] == identity.implementation_hash
                    and row["frozen_config_hash"] == identity.config_hash
                    and row["dataset"] == dataset
                    and row["split"] == split
                )
                if not exact_row_identity:
                    raise NegativeProtocolError(f"Shared score row provenance mismatch: {row['accuracy_pair_id']}")
                rows.append({**row, "shared_score_file": str(score_path.resolve())})
    return rows


def reuse_preflight(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    negative_protocol_root: Path | None = None,
    negative_run_root: Path | None = None,
    shared_accuracy_root: Path | None = None,
    sidecar_jar: Path | None = None,
    service_url: str = DEFAULT_SERVICE_URL,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Bind the current cold-pair implementation and find exact shared scores."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (negative_protocol_root or project_root / DEFAULT_NEGATIVE_PROTOCOL_ROOT).resolve()
    run_root = (negative_run_root or project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve()
    shared_root = (shared_accuracy_root or project_root / DEFAULT_SHARED_ACCURACY_ROOT).resolve()
    jar_path = (sidecar_jar or project_root / DEFAULT_SIDECAR_JAR).resolve()
    protocol = load_and_validate_negative_protocol(
        project_root=project_root,
        data_root=data_root,
        negative_protocol_root=protocol_root,
    )
    identity, shared_runtime = _shared_reuse_identity(shared_root)
    if file_sha256(jar_path) != identity.sidecar_jar_sha256:
        raise NegativeProtocolError("Current JAR does not match shared-accuracy JAR provenance.")

    current_conditions: list[dict[str, Any]] = []
    all_overlaps: list[dict[str, Any]] = []
    with ManagedSourceAfisSidecar(jar_path, service_url, timeout_seconds=timeout_seconds) as sidecar:
        startup = _startup_dict(sidecar.startup)
        client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
        try:
            health = client.health()
            validate_health(health)
            adapter = SourceAfisAdapter(client, health=health)
            for dataset in DATASETS:
                manifest = protocol_root / dataset / "benchmark_manifest.csv"
                context = prepare_run_context(
                    manifest_path=manifest,
                    expected_dataset=dataset,
                    expected_protocol=NEGATIVE_PROTOCOL,
                    adapter=adapter,
                    results_root=run_root,
                    startup_validation=startup,
                )
                current_identity = SharedReuseIdentity(
                    method=context.spec.method,
                    implementation_hash=context.spec.implementation_hash,
                    config_hash=context.spec.config_hash,
                    sourceafis_version=context.spec.method_version,
                    sidecar_jar_sha256=startup["jar_sha256"],
                )
                if current_identity != identity:
                    raise NegativeProtocolError(
                        f"Current runtime does not exactly match shared scores for {dataset}: "
                        f"{current_identity} != {identity}"
                    )
                negative_rows = _read_csv(
                    protocol_root / dataset / f"{NEGATIVE_PROTOCOL}.csv",
                    NEGATIVE_MANIFEST_COLUMNS,
                )
                shared_rows = _load_compatible_shared_rows(
                    shared_root=shared_root,
                    dataset=dataset,
                    identity=identity,
                )
                overlaps, missing = lookup_exact_reuse(negative_rows, shared_rows)
                all_overlaps.extend(overlaps)
                current_conditions.append(
                    {
                        "dataset": dataset,
                        "negative_pair_count": len(negative_rows),
                        "shared_compatible_score_count": len(shared_rows),
                        "overlap_count": len(overlaps),
                        "missing_count": missing,
                        "config_hash": context.spec.config_hash,
                        "implementation_hash": context.spec.implementation_hash,
                        "sidecar_jar_sha256": startup["jar_sha256"],
                        "sourceafis_version": context.spec.method_version,
                        "exact_provenance_match": True,
                    }
                )
        finally:
            client.close()

    report = {
        "schema_version": "sourceafis-negative-exact-reuse-preflight-v1",
        "protocol_summary_sha256": protocol["summary_sha256"],
        "method": identity.method,
        "sourceafis_version": identity.sourceafis_version,
        "config_hash": identity.config_hash,
        "implementation_hash": identity.implementation_hash,
        "sidecar_jar_sha256": identity.sidecar_jar_sha256,
        "shared_method_runtime_sha256": file_sha256(shared_root / "provenance" / "method_runtime.json"),
        "shared_runtime_sourceafis": shared_runtime["sourceafis"],
        "match_key": [
            "dataset",
            "ppi",
            "path_a",
            "path_b",
            "subject_id_a",
            "subject_id_b",
            "canonical_finger_position",
        ],
        "conditions": current_conditions,
        "overlap_count": len(all_overlaps),
        "missing_count": sum(item["missing_count"] for item in current_conditions),
        "full_cold_pair_run_still_required": True,
        "passed": True,
    }
    _publish_immutable_bytes(run_root / "reuse_preflight.json", _json_bytes(report))
    _publish_immutable_bytes(
        run_root / "reuse_overlap.csv",
        _csv_bytes(all_overlaps, REUSE_DETAIL_COLUMNS),
    )
    return {**report, "path": str(run_root / "reuse_preflight.json")}


def run_negative_protocol(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    negative_protocol_root: Path | None = None,
    negative_run_root: Path | None = None,
    bundle_staging_root: Path | None = None,
    sidecar_jar: Path | None = None,
    service_url: str = DEFAULT_SERVICE_URL,
    timeout_seconds: float = 120.0,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Run exactly two serial cold-pair manifests using one frozen JAR."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (negative_protocol_root or project_root / DEFAULT_NEGATIVE_PROTOCOL_ROOT).resolve()
    run_root = (negative_run_root or project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve()
    staging_root = (
        bundle_staging_root or project_root / ".sourceafis-negative-bundle-staging"
    ).resolve()
    jar_path = (sidecar_jar or project_root / DEFAULT_SIDECAR_JAR).resolve()
    protocol = load_and_validate_negative_protocol(
        project_root=project_root,
        data_root=data_root,
        negative_protocol_root=protocol_root,
    )
    preflight_path = run_root / "reuse_preflight.json"
    if not preflight_path.is_file():
        raise NegativeProtocolError("Exact reuse preflight must run before cold-pair execution.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True or preflight.get("protocol_summary_sha256") != protocol["summary_sha256"]:
        raise NegativeProtocolError("Exact reuse preflight is absent, failed, or stale.")
    if file_sha256(jar_path) != preflight["sidecar_jar_sha256"]:
        raise NegativeProtocolError("Execution JAR differs from exact reuse preflight.")

    frozen_by_dataset = {item["dataset"]: item for item in protocol["summary"]["datasets"]}
    runs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        frozen = frozen_by_dataset[dataset]
        manifest = protocol_root / dataset / "benchmark_manifest.csv"
        validator = make_negative_manifest_validator(
            negative_protocol_root=protocol_root,
            dataset=dataset,
            expected_manifest_sha256=frozen["benchmark_manifest_sha256"],
        )
        with ManagedSourceAfisSidecar(jar_path, service_url, timeout_seconds=timeout_seconds) as sidecar:
            startup = _startup_dict(sidecar.startup)
            client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
            try:
                health = client.health()
                validate_health(health)
                startup_validation = {
                    **startup,
                    "health": health.raw,
                    "health_requests_before_pair_execution": client.health_request_count,
                    "negative_protocol_namespace": NEGATIVE_NAMESPACE,
                    "negative_protocol_summary_path": protocol["summary_path"],
                    "negative_protocol_summary_sha256": protocol["summary_sha256"],
                    "research_manifest_path": str(
                        protocol_root / dataset / f"{NEGATIVE_PROTOCOL}.csv"
                    ),
                    "research_manifest_sha256": frozen["manifest_sha256"],
                    "serial_dataset_order": list(DATASETS),
                    "cross_pair_template_cache": False,
                }
                adapter = SourceAfisAdapter(client, health=health)
                context = prepare_run_context(
                    manifest_path=manifest,
                    expected_dataset=dataset,
                    expected_protocol=NEGATIVE_PROTOCOL,
                    adapter=adapter,
                    results_root=run_root,
                    startup_validation=startup_validation,
                    bundle_directory=staging_root / dataset / preflight["config_hash"],
                )
                exact = (
                    context.spec.config_hash == preflight["config_hash"]
                    and context.spec.implementation_hash == preflight["implementation_hash"]
                    and context.spec.method_version == preflight["sourceafis_version"]
                    and startup["jar_sha256"] == preflight["sidecar_jar_sha256"]
                )
                if not exact:
                    raise NegativeProtocolError(f"Runtime identity drift before {dataset} execution.")
                metadata = run_benchmark_manifest(
                    manifest_path=manifest,
                    adapter=adapter,
                    expected_dataset=dataset,
                    expected_protocol=NEGATIVE_PROTOCOL,
                    results_root=run_root,
                    startup_validation=startup_validation,
                    data_root=data_root,
                    dedicated_validator=validator,
                    skip_existing=skip_existing,
                    bundle_directory=context.bundle_directory,
                    progress_callback=lambda completed, total, d=dataset: print(
                        f"[{d}/{NEGATIVE_PROTOCOL}] {completed}/{total} measured cold pairs",
                        file=sys.stderr,
                        flush=True,
                    ),
                )
            finally:
                client.close()
        normal_bundle = (
            run_root
            / dataset
            / NEGATIVE_PROTOCOL
            / METHOD
            / BENCHMARK_CONTRACT_VERSION
            / metadata["config_hash"]
        ).resolve()
        bundle = _windows_extended_path(normal_bundle)
        if bundle.exists():
            raise NegativeProtocolError(f"Final negative bundle already exists: {normal_bundle}")
        bundle.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(context.bundle_directory, bundle)
        except OSError as exc:
            raise NegativeProtocolError(
                f"Cannot promote validated staging bundle to {normal_bundle}: {exc}"
            ) from exc
        relocated_spec_data = dict(metadata["run_spec"])
        relocated_spec_data["manifest_path"] = Path(relocated_spec_data["manifest_path"])
        relocated_spec = BenchmarkRunSpec(**relocated_spec_data)
        validate_result_bundle(
            bundle,
            manifest_records=read_pair_manifest(manifest),
            run_spec=relocated_spec,
            score_direction=metadata["score_direction"],
            score_semantics=metadata["score_semantics"],
        )
        runs.append(
            {
                "dataset": dataset,
                "pair_count": metadata["result"]["row_count"],
                "success_count": metadata["success_count"],
                "failure_count": sum(metadata["failure_counts"].values()),
                "bundle_path": str(normal_bundle),
                "bundle_staged_then_atomically_promoted": True,
                "pairs_sha256": metadata["result"]["sha256"],
                "score_payload_sha256": metadata["result"]["score_payload_sha256"],
                "config_hash": metadata["config_hash"],
                "implementation_hash": metadata["implementation_hash"],
                "sidecar_jar_sha256": startup["jar_sha256"],
                "timing_mode": metadata["timing_mode"],
                "execution_wall_ms": metadata["execution_wall_ms"],
            }
        )
    report = {
        "schema_version": "sourceafis-negative-cold-pair-run-summary-v1",
        "protocol_summary_sha256": protocol["summary_sha256"],
        "method": METHOD,
        "method_version": "3.18.1",
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "timing_mode": "cold_pair",
        "serial_execution": True,
        "fresh_jvm_per_dataset": True,
        "cross_pair_template_cache": False,
        "sidecar_jar_sha256": file_sha256(jar_path),
        "runs": runs,
        "self_rerun_count": 0,
        "genuine_plain_roll_rerun_count": 0,
        "sift_rerun_count": 0,
        "primary_artifacts_overwritten": False,
    }
    _publish_immutable_bytes(run_root / "run_summary.json", _json_bytes(report))
    return {**report, "path": str(run_root / "run_summary.json")}


def _load_negative_bundle(
    *,
    run_root: Path,
    protocol_root: Path,
    dataset: str,
) -> PrimaryBundle:
    contract_root = _windows_extended_path(
        run_root / dataset / NEGATIVE_PROTOCOL / METHOD / BENCHMARK_CONTRACT_VERSION
    )
    bundles = sorted(
        metadata.parent
        for metadata in contract_root.glob(f"*/{METADATA_FILENAME}")
        if (metadata.parent / RESULT_FILENAME).is_file()
    )
    if len(bundles) != 1:
        raise NegativeProtocolError(f"Expected one negative bundle for {dataset}, found {len(bundles)}.")
    bundle = bundles[0].resolve()
    metadata = json.loads((bundle / METADATA_FILENAME).read_text(encoding="utf-8"))
    raw_spec = dict(metadata["run_spec"])
    raw_spec["manifest_path"] = Path(raw_spec["manifest_path"])
    spec = BenchmarkRunSpec(**raw_spec)
    manifest = (protocol_root / dataset / "benchmark_manifest.csv").resolve()
    if spec.manifest_path.resolve() != manifest or spec.expected_protocol != NEGATIVE_PROTOCOL:
        raise NegativeProtocolError(f"Negative bundle points to wrong manifest/protocol for {dataset}.")
    pairs = read_pair_manifest(manifest)
    validate_result_bundle(
        bundle,
        manifest_records=pairs,
        run_spec=spec,
        score_direction=metadata["score_direction"],
        score_semantics=metadata["score_semantics"],
    )
    rows = read_result_rows(bundle / RESULT_FILENAME)
    return PrimaryBundle(dataset, NEGATIVE_PROTOCOL, manifest, bundle, pairs, rows, metadata)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def negative_decision_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    dataset: str,
) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    scores = [float(row["raw_score"]) for row in ok]
    false_matches = sum(score >= SOURCEAFIS_THRESHOLD for score in scores)
    correct = sum(score < SOURCEAFIS_THRESHOLD for score in scores)
    compare_times = [float(row["method_compare_ms"]) for row in ok if row["method_compare_ms"]]
    total_times = [float(row["total_ms"]) for row in rows if row["total_ms"]]
    total = len(rows)
    return {
        "dataset": dataset,
        "threshold": SOURCEAFIS_THRESHOLD,
        "total_wrong_pairs": total,
        "status_ok": len(ok),
        "failures": total - len(ok),
        "false_matches": false_matches,
        "correct_non_matches": correct,
        "false_match_percentage": 100.0 * false_matches / total if total else None,
        "correct_non_match_percentage": 100.0 * correct / total if total else None,
        "score_zero": sum(score == 0.0 for score in scores),
        "positive_below_threshold": sum(0.0 < score < SOURCEAFIS_THRESHOLD for score in scores),
        "mean_score": _mean(scores),
        "median_score": _median(scores),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "mean_method_compare_ms": _mean(compare_times),
        "median_method_compare_ms": _median(compare_times),
        "p95_method_compare_ms": _p95(compare_times),
        "mean_total_ms": _mean(total_times),
        "median_total_ms": _median(total_times),
        "p95_total_ms": _p95(total_times),
    }


def _json_diagnostics_equal(left: str, right: str) -> bool:
    try:
        return json.loads(left or "{}") == json.loads(right or "{}")
    except json.JSONDecodeError as exc:
        raise NegativeProtocolError(f"Invalid deterministic diagnostics JSON: {exc}") from exc


def compare_overlap_rows(
    overlap_rows: Sequence[Mapping[str, str]],
    cold_rows: Sequence[Mapping[str, str]],
    *,
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cold_by_id = {row["pair_id"]: row for row in cold_rows}
    details: list[dict[str, Any]] = []
    for shared in overlap_rows:
        if shared["dataset"] != dataset:
            continue
        pair_id = shared["negative_pair_id"]
        cold = cold_by_id.get(pair_id)
        if cold is None:
            raise NegativeProtocolError(f"Overlap pair missing from cold result: {pair_id}")
        shared_score = shared["raw_score"]
        cold_score = cold["raw_score"]
        delta = (
            abs(float(shared_score) - float(cold_score))
            if shared_score != "" and cold_score != ""
            else None
        )
        row = {
            "dataset": dataset,
            "negative_pair_id": pair_id,
            "shared_accuracy_pair_id": shared["shared_accuracy_pair_id"],
            "shared_raw_score": shared_score,
            "cold_pair_raw_score": cold_score,
            "raw_score_text_equal": shared_score == cold_score,
            "raw_score_abs_delta": delta,
            "status_equal": shared["status"] == cold["status"],
            "error_code_equal": shared["error_code"] == cold["error_code"],
            "prepare_a_diagnostics_equal": _json_diagnostics_equal(
                shared["prepare_a_diagnostics_json"], cold["prepare_a_diagnostics"]
            ),
            "prepare_b_diagnostics_equal": _json_diagnostics_equal(
                shared["prepare_b_diagnostics_json"], cold["prepare_b_diagnostics"]
            ),
            "compare_diagnostics_equal": _json_diagnostics_equal(
                shared["compare_diagnostics_json"], cold["compare_diagnostics"]
            ),
        }
        row["exact_reproducibility"] = all(
            row[name]
            for name in (
                "raw_score_text_equal",
                "status_equal",
                "error_code_equal",
                "prepare_a_diagnostics_equal",
                "prepare_b_diagnostics_equal",
                "compare_diagnostics_equal",
            )
        )
        details.append(row)
    deltas = [row["raw_score_abs_delta"] for row in details if row["raw_score_abs_delta"] is not None]
    summary = {
        "dataset": dataset,
        "overlap_count": len(details),
        "exact_score_match_count": sum(row["raw_score_text_equal"] for row in details),
        "mismatch_count": sum(not row["exact_reproducibility"] for row in details),
        "max_absolute_score_delta": max(deltas) if deltas else None,
        "status_match_count": sum(row["status_equal"] for row in details),
        "error_code_match_count": sum(row["error_code_equal"] for row in details),
        "deterministic_diagnostics_match_count": sum(
            row["prepare_a_diagnostics_equal"]
            and row["prepare_b_diagnostics_equal"]
            and row["compare_diagnostics_equal"]
            for row in details
        ),
        "timing_equality_required": False,
        "passed": all(row["exact_reproducibility"] for row in details),
    }
    return details, summary


def analyze_negative_results(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    negative_protocol_root: Path | None = None,
    negative_run_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (negative_protocol_root or project_root / DEFAULT_NEGATIVE_PROTOCOL_ROOT).resolve()
    run_root = (negative_run_root or project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve()
    protocol = load_and_validate_negative_protocol(
        project_root=project_root,
        data_root=data_root,
        negative_protocol_root=protocol_root,
    )
    overlap_rows = _read_csv(run_root / "reuse_overlap.csv", REUSE_DETAIL_COLUMNS)
    decisions: list[dict[str, Any]] = []
    repro_summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        bundle = _load_negative_bundle(run_root=run_root, protocol_root=protocol_root, dataset=dataset)
        decisions.append(negative_decision_summary(bundle.rows, dataset=dataset))
        details, repro = compare_overlap_rows(overlap_rows, bundle.rows, dataset=dataset)
        _publish_immutable_bytes(
            run_root / "reproducibility" / dataset / "exact_score_reproducibility.csv",
            _csv_bytes(details, REPRO_DETAIL_COLUMNS),
        )
        repro_summaries.append(repro)
    result_report = {
        "schema_version": NEGATIVE_RESULT_SCHEMA_VERSION,
        "threshold": SOURCEAFIS_THRESHOLD,
        "decision_rule": "match iff status == ok and raw_score >= 40; non_match iff status == ok and raw_score < 40",
        "datasets": decisions,
    }
    repro_report = {
        "schema_version": "sourceafis-negative-exact-score-reproducibility-v1",
        "exact_equality_required": True,
        "timing_equality_required": False,
        "datasets": repro_summaries,
        "overall": {
            "overlap_count": sum(row["overlap_count"] for row in repro_summaries),
            "exact_score_match_count": sum(row["exact_score_match_count"] for row in repro_summaries),
            "mismatch_count": sum(row["mismatch_count"] for row in repro_summaries),
            "max_absolute_score_delta": max(
                (row["max_absolute_score_delta"] or 0.0) for row in repro_summaries
            ),
            "passed": all(row["passed"] for row in repro_summaries),
        },
    }
    _publish_immutable_bytes(run_root / "negative_results_summary.json", _json_bytes(result_report))
    _publish_immutable_bytes(
        run_root / "negative_results_summary.csv",
        _csv_bytes(decisions, NEGATIVE_RESULT_COLUMNS),
    )
    _publish_immutable_bytes(run_root / "reproducibility_summary.json", _json_bytes(repro_report))
    _publish_immutable_bytes(
        run_root / "reproducibility_summary.csv",
        _csv_bytes(repro_summaries, list(repro_summaries[0].keys())),
    )
    if not repro_report["overall"]["passed"]:
        raise NegativeProtocolError("Exact shared-score reproducibility failed.")
    return {
        "protocol_summary_sha256": protocol["summary_sha256"],
        "negative_results": result_report,
        "reproducibility": repro_report,
    }


def self_decision_summary(rows: Sequence[Mapping[str, str]], *, dataset: str, protocol: str) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    scores = [float(row["raw_score"]) for row in ok]
    matches = sum(score >= SOURCEAFIS_THRESHOLD for score in scores)
    non_matches = sum(score < SOURCEAFIS_THRESHOLD for score in scores)
    compare = [float(row["method_compare_ms"]) for row in ok if row["method_compare_ms"]]
    return {
        "dataset": dataset,
        "protocol": protocol,
        "total": len(rows),
        "status_ok": len(ok),
        "failures": len(rows) - len(ok),
        "matches": matches,
        "non_matches": non_matches,
        "score_zero": sum(score == 0.0 for score in scores),
        "positive_below_threshold": sum(0.0 < score < SOURCEAFIS_THRESHOLD for score in scores),
        "removed_before_derived_protocol": len(rows) - matches,
        "match_percentage": 100.0 * matches / len(rows) if rows else None,
        "mean_method_compare_ms": _mean(compare),
    }


def genuine_decision_summary(rows: Sequence[Mapping[str, str]], *, dataset: str) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    scores = [float(row["raw_score"]) for row in ok]
    matches = sum(score >= SOURCEAFIS_THRESHOLD for score in scores)
    non_matches = sum(score < SOURCEAFIS_THRESHOLD for score in scores)
    compare = [float(row["method_compare_ms"]) for row in ok if row["method_compare_ms"]]
    total_times = [float(row["total_ms"]) for row in rows if row["total_ms"]]
    return {
        "dataset": dataset,
        "total_pairs": len(rows),
        "status_ok": len(ok),
        "failures": len(rows) - len(ok),
        "matches": matches,
        "non_matches": non_matches,
        "match_percentage": 100.0 * matches / len(rows) if rows else None,
        "non_match_percentage": 100.0 * non_matches / len(rows) if rows else None,
        "score_zero": sum(score == 0.0 for score in scores),
        "positive_below_threshold": sum(0.0 < score < SOURCEAFIS_THRESHOLD for score in scores),
        "mean_score": _mean(scores),
        "median_score": _median(scores),
        "mean_method_compare_ms": _mean(compare),
        "median_method_compare_ms": _median(compare),
        "p95_method_compare_ms": _p95(compare),
        "mean_total_ms": _mean(total_times),
        "median_total_ms": _median(total_times),
        "p95_total_ms": _p95(total_times),
    }


def _ground_truth_counts(pairs: Sequence[PairRecord]) -> dict[str, int]:
    same = sum(
        pair.subject_id in pair.path_a.name
        and pair.subject_id in pair.path_b.name
        and canonical_finger_position("plain", pair.raw_frgp_a)
        == pair.canonical_finger_position
        and canonical_finger_position("roll", pair.raw_frgp_b)
        == pair.canonical_finger_position
        for pair in pairs
    )
    return {"same_subject_same_finger_count": same, "wrong_count": len(pairs) - same}


def collect_existing_source_summaries(
    *,
    project_root: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Read existing primary self and derived genuine artifacts; never rerun them."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    genuine_protocol_root = project_root / GENUINE_PROTOCOL_RELATIVE_ROOT
    genuine_run_root = project_root / GENUINE_RUN_RELATIVE_ROOT
    records: dict[str, Any] = {}
    self_summaries: dict[str, dict[str, Any]] = {}
    genuine_summaries: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        plain = load_primary_bundle(project_root, data_root, dataset, "plain_self", validate_source_manifest=True)
        roll = load_primary_bundle(project_root, data_root, dataset, "roll_self", validate_source_manifest=True)
        records[dataset] = {
            "plain_single_finger_records": len(plain.pairs),
            "roll_single_finger_records": len(roll.pairs),
            "plain_advisory_expected": ADVISORY_EXPECTED_SINGLE_FINGER_COUNTS["plain_self"],
            "roll_advisory_expected": ADVISORY_EXPECTED_SINGLE_FINGER_COUNTS["roll_self"],
        }
        self_summaries[dataset] = {
            "plain_self": self_decision_summary(plain.rows, dataset=dataset, protocol="plain_self"),
            "roll_self": self_decision_summary(roll.rows, dataset=dataset, protocol="roll_self"),
        }
        genuine_bundle = _load_derived_bundle(
            run_root=genuine_run_root,
            protocol_root=genuine_protocol_root,
            dataset=dataset,
        )
        genuine_pairs = read_pair_manifest(genuine_protocol_root / dataset / "plain_roll.csv")
        ground_truth = _ground_truth_counts(genuine_pairs)
        if ground_truth["wrong_count"] != 0:
            raise NegativeProtocolError(f"Derived genuine ground truth is contaminated for {dataset}.")
        genuine_summaries[dataset] = {
            **ground_truth,
            **genuine_decision_summary(genuine_bundle.rows, dataset=dataset),
        }
    return {
        "record_counts": records,
        "self": self_summaries,
        "genuine": genuine_summaries,
        "self_rerun_count": 0,
        "genuine_rerun_count": 0,
    }


def _section_dataset(section: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    return section["datasets"][dataset]


def build_report_sections(
    source: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> list[dict[str, Any]]:
    negative_by_dataset = {
        row["dataset"]: row for row in negative["negative_results"]["datasets"]
    }
    return [
        {
            "number": 1,
            "title": REPORT_SECTION_TITLES[0],
            "datasets": {
                dataset: {
                    "plain_single_finger_records": source["record_counts"][dataset]["plain_single_finger_records"],
                    "roll_single_finger_records": source["record_counts"][dataset]["roll_single_finger_records"],
                }
                for dataset in DATASETS
            },
        },
        {
            "number": 2,
            "title": REPORT_SECTION_TITLES[1],
            "datasets": {
                dataset: {
                    "total": source["self"][dataset]["plain_self"]["total"],
                    "matches": source["self"][dataset]["plain_self"]["matches"],
                    "non_matches_removed_from_derived_protocol": source["self"][dataset]["plain_self"]["removed_before_derived_protocol"],
                    "match_percentage": source["self"][dataset]["plain_self"]["match_percentage"],
                    "mean_comparison_time_ms": source["self"][dataset]["plain_self"]["mean_method_compare_ms"],
                }
                for dataset in DATASETS
            },
        },
        {
            "number": 3,
            "title": REPORT_SECTION_TITLES[2],
            "datasets": {
                dataset: {
                    "total": source["self"][dataset]["roll_self"]["total"],
                    "matches": source["self"][dataset]["roll_self"]["matches"],
                    "non_matches_removed_from_derived_protocol": source["self"][dataset]["roll_self"]["removed_before_derived_protocol"],
                    "match_percentage": source["self"][dataset]["roll_self"]["match_percentage"],
                    "mean_comparison_time_ms": source["self"][dataset]["roll_self"]["mean_method_compare_ms"],
                }
                for dataset in DATASETS
            },
        },
        {
            "number": 4,
            "title": REPORT_SECTION_TITLES[3],
            "datasets": {
                dataset: {
                    "pair_count": source["genuine"][dataset]["total_pairs"],
                    "ground_truth_same_subject_same_finger_count": source["genuine"][dataset]["same_subject_same_finger_count"],
                    "ground_truth_wrong_count": source["genuine"][dataset]["wrong_count"],
                }
                for dataset in DATASETS
            },
        },
        {
            "number": 5,
            "title": REPORT_SECTION_TITLES[4],
            "datasets": {
                dataset: {
                    "matched_by_sourceafis": source["genuine"][dataset]["matches"],
                    "not_matched_by_sourceafis": source["genuine"][dataset]["non_matches"],
                    "match_percentage": source["genuine"][dataset]["match_percentage"],
                    "mean_comparison_time_ms": source["genuine"][dataset]["mean_method_compare_ms"],
                }
                for dataset in DATASETS
            },
        },
        {
            "number": 6,
            "title": REPORT_SECTION_TITLES[5],
            "pairing_method": "next subject within the same canonical finger position, circular shift by one",
            "datasets": {
                dataset: {
                    "pair_count": negative_by_dataset[dataset]["total_wrong_pairs"],
                    "incorrectly_matched_by_sourceafis": negative_by_dataset[dataset]["false_matches"],
                    "correctly_rejected_by_sourceafis": negative_by_dataset[dataset]["correct_non_matches"],
                    "false_match_percentage": negative_by_dataset[dataset]["false_match_percentage"],
                    "mean_comparison_time_ms": negative_by_dataset[dataset]["mean_method_compare_ms"],
                }
                for dataset in DATASETS
            },
        },
    ]


def _render_markdown(sections: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# SourceAFIS exact requested supervisor report", ""]
    for section in sections:
        lines.extend([f"## {section['number']}. {section['title']}", ""])
        if section["number"] == 6:
            lines.extend([f"Pairing method: `{section['pairing_method']}`", ""])
        metric_names = list(_section_dataset(section, DATASETS[0]).keys())
        lines.extend(["| Dataset | " + " | ".join(metric_names) + " |", "|---|" + "---:|" * len(metric_names)])
        for dataset in DATASETS:
            values = _section_dataset(section, dataset)
            rendered = []
            for name in metric_names:
                value = values[name]
                if isinstance(value, float):
                    rendered.append(f"{value:.9f}")
                else:
                    rendered.append(str(value))
            lines.append(f"| {dataset.upper()} | " + " | ".join(rendered) + " |")
        lines.append("")
    return "\n".join(lines)


def _long_report_rows(sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in sections:
        for dataset in DATASETS:
            for metric, value in _section_dataset(section, dataset).items():
                rows.append(
                    {
                        "section_number": section["number"],
                        "section": section["title"],
                        "dataset": dataset,
                        "metric": metric,
                        "value": value,
                    }
                )
    return rows


def write_supervisor_report(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    negative_run_root: Path | None = None,
    supervisor_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    run_root = (negative_run_root or project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve()
    final_root = (supervisor_root or project_root / DEFAULT_SUPERVISOR_ROOT).resolve()
    source = collect_existing_source_summaries(project_root=project_root, data_root=data_root)
    negative = analyze_negative_results(
        project_root=project_root,
        data_root=data_root,
        negative_run_root=run_root,
    )
    sections = build_report_sections(source, negative)
    if tuple(section["title"] for section in sections) != REPORT_SECTION_TITLES:
        raise NegativeProtocolError("Supervisor report section allowlist/order mismatch.")
    forbidden = ("roc", "det", "auc", "eer", "confidence", "calibrat", "sift", "fusion", "ranking")
    serialized_sections = json.dumps(sections, ensure_ascii=True).casefold()
    if any(term in serialized_sections for term in forbidden):
        raise NegativeProtocolError("Supervisor report contains a forbidden topic.")

    candidate = create_candidate_directory(final_root)
    try:
        (candidate / "supervisor_report.md").write_text(
            _render_markdown(sections), encoding="utf-8", newline="\n"
        )
        write_csv_atomic(
            _long_report_rows(sections),
            candidate / "supervisor_report.csv",
            ["section_number", "section", "dataset", "metric", "value"],
        )
        write_json_atomic(
            {"schema_version": REPORT_SCHEMA_VERSION, "sections": sections},
            candidate / "supervisor_report.json",
        )
        _publish_immutable_candidate(candidate, final_root)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)
    return {
        "root": str(final_root),
        "sections": sections,
        "files": {
            path.name: file_sha256(path)
            for path in sorted(final_root.iterdir())
            if path.is_file()
        },
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _windows_extended_path(path: Path) -> Path:
    """Return the same absolute path through Win32's long-path namespace."""

    resolved = path.resolve()
    value = str(resolved)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return Path("\\\\?\\" + value)
    return resolved


def enumerate_relevant_protected_paths(project_root: Path) -> dict[Path, str]:
    """Enumerate only protected artifacts relevant to this exact experiment."""

    project_root = project_root.resolve()
    exclusions = [
        (project_root / DEFAULT_NEGATIVE_PROTOCOL_ROOT).resolve(),
        (project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve(),
        (project_root / DEFAULT_SUPERVISOR_ROOT).resolve(),
    ]
    roots = [
        (project_root / "protocols", "base_manifests"),
        (project_root / "results" / "sd300b", "sourceafis_primary_results"),
        (project_root / "results" / "sd300c", "sourceafis_primary_results"),
        (project_root / "results" / "sourceafis", "sourceafis_audits"),
        (project_root / "results" / "cohorts", "sourceafis_cohorts"),
        (project_root / GENUINE_PROTOCOL_RELATIVE_ROOT, "sourceafis_derived_genuine_protocol"),
        (project_root / GENUINE_RUN_RELATIVE_ROOT, "sourceafis_derived_genuine_results"),
        (project_root / DEFAULT_SHARED_ACCURACY_ROOT, "shared_accuracy_artifacts"),
        (project_root / "results" / "derived_protocols" / "sift_geometric_per_dataset_self_accept_v1", "sift_results"),
        (project_root / "results" / "derived_protocol_runs" / "sift_geometric_per_dataset_self_accept_v1", "sift_results"),
        (project_root / "results" / "sift_geometric", "sift_results"),
        (project_root / "apps" / "sourceafis-sidecar", "sourceafis_sidecar"),
        (project_root / "src" / "fingerprint_benchmark" / "sift", "sift_code"),
    ]
    paths: dict[Path, str] = {}
    for root, category in roots:
        root = root.resolve()
        if not root.is_dir():
            raise NegativeProtocolError(f"Required protected root is missing: {root}")
        for path in sorted(item.resolve() for item in root.rglob("*") if item.is_file()):
            if any(_is_within(path, exclusion) for exclusion in exclusions):
                continue
            previous = paths.get(path)
            if previous is not None and previous != category:
                # A nested root is classified by its more specific category.
                continue
            paths[path] = category
    benchmark_root = project_root / "src" / "fingerprint_benchmark"
    frozen_files = (
        "bundle.py",
        "contract.py",
        "hashing.py",
        "io.py",
        "manifest.py",
        "preflight.py",
        "provenance.py",
        "runner.py",
        "sourceafis_adapter.py",
        "sourceafis_client.py",
        "sourceafis_sidecar.py",
        "sift_derived_protocol.py",
    )
    for filename in frozen_files:
        path = (benchmark_root / filename).resolve()
        if not path.is_file():
            raise NegativeProtocolError(f"Required protected implementation file is missing: {path}")
        paths[path] = "frozen_implementation"
    return dict(sorted(paths.items(), key=lambda item: item[0].as_posix().casefold()))


def capture_integrity_before(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    negative_run_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_root = (negative_run_root or project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve()
    paths = enumerate_relevant_protected_paths(project_root)
    return capture_snapshot(paths, run_root / "integrity" / "protected_before.jsonl")


def verify_integrity_after(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    negative_run_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_root = (negative_run_root or project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve()
    before_path = run_root / "integrity" / "protected_before.jsonl"
    before_records, _ = read_snapshot(before_path)
    frozen_existing_paths = {
        Path(record["path"]): record["category"] for record in before_records
    }
    report, records, footer = compare_snapshot(before_path, frozen_existing_paths)
    report["inventory_policy"] = (
        "rehash_exact_existing_file_inventory_frozen_before_execution; new output roots excluded"
    )
    after_payloads = [
        {"header": {"schema_version": "shared-accuracy-protected-snapshot-v1", "hash_algorithm": "sha256"}},
        *records,
        {"footer": footer},
    ]
    after_bytes = (
        "\n".join(
            json.dumps(item, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for item in after_payloads
        )
        + "\n"
    ).encode("utf-8")
    after_path = run_root / "integrity" / "protected_after.jsonl"
    report_path = run_root / "integrity" / "protected_artifact_integrity.json"
    publish_immutable_bytes(after_path, after_bytes)
    report["after_snapshot_path"] = str(after_path.resolve())
    report["after_snapshot_sha256"] = file_sha256(after_path)
    publish_immutable_json(report_path, report)
    if report["protected_artifacts_unchanged"] is not True:
        raise NegativeProtocolError(
            f"Protected artifacts changed; mismatch_count={report['mismatch_count']}."
        )
    return {**report, "report_path": str(report_path), "report_sha256": file_sha256(report_path)}


def finalize_artifact_manifest(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    negative_protocol_root: Path | None = None,
    negative_run_root: Path | None = None,
    supervisor_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    protocol_root = (negative_protocol_root or project_root / DEFAULT_NEGATIVE_PROTOCOL_ROOT).resolve()
    run_root = (negative_run_root or project_root / DEFAULT_NEGATIVE_RUN_ROOT).resolve()
    report_root = (supervisor_root or project_root / DEFAULT_SUPERVISOR_ROOT).resolve()
    integrity_path = run_root / "integrity" / "protected_artifact_integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("protected_artifacts_unchanged") is not True:
        raise NegativeProtocolError("Cannot finalize without protected-artifact equality.")
    manifest_path = report_root / "artifact_manifest.json"
    files: list[dict[str, Any]] = []
    for label, root in (
        ("negative_protocol", protocol_root),
        ("negative_runs", run_root),
        ("supervisor_report", report_root),
    ):
        scan_root = _windows_extended_path(root)
        for path in sorted(item for item in scan_root.rglob("*") if item.is_file()):
            display_value = str(path)
            if display_value.startswith("\\\\?\\"):
                display_value = display_value[4:]
            display_path = Path(display_value)
            if display_path.resolve() == manifest_path.resolve():
                continue
            files.append(
                {
                    "category": label,
                    "path": str(display_path),
                    "relative_path": path.relative_to(scan_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    implementation_files = []
    for relative in (
        Path("src/fingerprint_benchmark/sourceafis_negative_protocol.py"),
        Path("tests/test_sourceafis_negative_protocol.py"),
    ):
        path = (project_root / relative).resolve()
        implementation_files.append(
            {"path": str(path), "sha256": file_sha256(path), "size": path.stat().st_size}
        )
    tree_digest = __import__("hashlib").sha256()
    for record in files:
        tree_digest.update(canonical_json_bytes(record))
        tree_digest.update(b"\n")
    payload = {
        "schema_version": "sourceafis-exact-requested-artifact-manifest-v1",
        "hash_algorithm": "sha256",
        "output_file_count": len(files),
        "output_tree_sha256": tree_digest.hexdigest(),
        "files": files,
        "implementation_files": implementation_files,
        "protected_artifacts_unchanged": True,
        "protected_tree_sha256": integrity["before"]["tree_sha256"],
        "sift_invoked": False,
        "sift_modified": False,
    }
    _publish_immutable_bytes(manifest_path, _json_bytes(payload))
    return {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "output_file_count": len(files),
        "output_tree_sha256": payload["output_tree_sha256"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exact requested SourceAFIS next-subject experiment.")
    parser.add_argument(
        "command",
        choices=("prepare", "integrity-before", "preflight", "run", "analyze", "report", "integrity-after", "finalize"),
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--negative-protocol-root", type=Path)
    parser.add_argument("--negative-run-root", type=Path)
    parser.add_argument("--bundle-staging-root", type=Path)
    parser.add_argument("--supervisor-root", type=Path)
    parser.add_argument("--shared-accuracy-root", type=Path)
    parser.add_argument("--sidecar-jar", type=Path)
    parser.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "project_root": args.project_root,
        "data_root": args.data_root,
    }
    try:
        if args.command == "prepare":
            result = prepare_negative_protocol(
                **common,
                negative_protocol_root=args.negative_protocol_root,
            )
        elif args.command == "integrity-before":
            result = capture_integrity_before(
                project_root=args.project_root,
                negative_run_root=args.negative_run_root,
            )
        elif args.command == "preflight":
            result = reuse_preflight(
                **common,
                negative_protocol_root=args.negative_protocol_root,
                negative_run_root=args.negative_run_root,
                bundle_staging_root=args.bundle_staging_root,
                shared_accuracy_root=args.shared_accuracy_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "run":
            result = run_negative_protocol(
                **common,
                negative_protocol_root=args.negative_protocol_root,
                negative_run_root=args.negative_run_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
                skip_existing=args.skip_existing,
            )
        elif args.command == "analyze":
            result = analyze_negative_results(
                **common,
                negative_protocol_root=args.negative_protocol_root,
                negative_run_root=args.negative_run_root,
            )
        elif args.command == "report":
            result = write_supervisor_report(
                **common,
                negative_run_root=args.negative_run_root,
                supervisor_root=args.supervisor_root,
            )
        elif args.command == "integrity-after":
            result = verify_integrity_after(
                project_root=args.project_root,
                negative_run_root=args.negative_run_root,
            )
        elif args.command == "finalize":
            result = finalize_artifact_manifest(
                project_root=args.project_root,
                negative_protocol_root=args.negative_protocol_root,
                negative_run_root=args.negative_run_root,
                supervisor_root=args.supervisor_root,
            )
        else:  # pragma: no cover
            raise NegativeProtocolError(f"Unsupported command: {args.command}")
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except (NegativeProtocolError, DerivedProtocolError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
