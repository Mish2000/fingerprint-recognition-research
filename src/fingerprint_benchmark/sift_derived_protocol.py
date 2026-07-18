"""Per-dataset SIFT self-accept-then-plain-roll derived protocol.

The six primary SIFT bundles, frozen development artifacts, base manifests,
SourceAFIS artifacts, and datasets are immutable inputs.  This module only
publishes method-specific derived manifests and derived plain-roll reruns.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from .bundle import (
    BundlePublicationError,
    create_candidate_directory,
    discard_candidate_directory,
    publish_candidate_directory,
)
from .contract import BENCHMARK_CONTRACT_VERSION, BenchmarkRunSpec
from .hashing import canonical_json_bytes, file_sha256, stable_hash
from .io import write_csv_atomic, write_json_atomic
from .manifest import MANIFEST_COLUMNS, PairRecord, read_pair_manifest
from .preflight import validator_for
from .runner import (
    METADATA_FILENAME,
    RESULT_FILENAME,
    _execute_pair,
    _run_warm_up,
    prepare_run_context,
    read_result_rows,
    run_benchmark_manifest,
    validate_result_bundle,
)
from .sift.adapter import SiftGeometricAdapter
from .sift.config import METHOD_NAME, METHOD_VERSION, SiftGeometricConfig


PROTOCOL_NAMESPACE = "sift_geometric_per_dataset_self_accept_v1"
PROTOCOL_SCHEMA_VERSION = "sift-per-dataset-self-accept-derived-protocol-v1"
PREFLIGHT_SCHEMA_VERSION = "sift-derived-runtime-preflight-v1"
REPORT_SCHEMA_VERSION = "sift-derived-report-v1"
DATASETS = ("sd300b", "sd300c")
PROTOCOLS = ("plain_self", "roll_self", "plain_roll")
SELF_PROTOCOLS = ("plain_self", "roll_self")

EXPECTED_CONFIG_FILE_SHA256 = "f9f0623ae89752d09c5933d49dc80acc5803863cc8dc7109efb98b96d282f01f"
EXPECTED_DECISION_FILE_SHA256 = "13e9e29d918f95783d68eecb70f6aa857009ac902417d3ac8d59dcf59b7a98fa"

DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
DEFAULT_PROTOCOL_ROOT = Path("results") / "derived_protocols" / PROTOCOL_NAMESPACE
DEFAULT_RUN_ROOT = Path("results") / "derived_protocol_runs" / PROTOCOL_NAMESPACE
DEFAULT_CONFIG = Path("results") / "sift_geometric" / "development" / "sift_geometric_config.json"
DEFAULT_DECISION = Path("results") / "sift_geometric" / "development" / "decision_rule.json"

Identity = tuple[str, int]

INCLUDED_COLUMNS = [
    "dataset",
    "subject_id",
    "canonical_finger_position",
    "plain_self_pair_id",
    "roll_self_pair_id",
    "base_plain_roll_pair_id",
    "plain_self_decision",
    "roll_self_decision",
    "plain_self_status",
    "roll_self_status",
    "plain_self_raw_score",
    "roll_self_raw_score",
    "plain_self_keypoint_count_a",
    "plain_self_keypoint_count_b",
    "plain_self_candidate_match_count",
    "plain_self_geometric_inlier_count",
    "plain_self_inlier_ratio",
    "plain_self_geometric_model_status",
    "roll_self_keypoint_count_a",
    "roll_self_keypoint_count_b",
    "roll_self_candidate_match_count",
    "roll_self_geometric_inlier_count",
    "roll_self_inlier_ratio",
    "roll_self_geometric_model_status",
]
EXCLUDED_COLUMNS = [*INCLUDED_COLUMNS, "reason_flags"]

PROTOCOL_SUMMARY_COLUMNS = [
    "dataset",
    "threshold",
    "plain_self_total_count",
    "plain_self_accepted_count",
    "roll_self_total_count",
    "roll_self_accepted_count",
    "base_plain_roll_count",
    "identity_universe_count",
    "included_identity_count",
    "excluded_identity_count",
    "derived_manifest_sha256",
]

DETERMINISTIC_DIAGNOSTIC_FIELDS = (
    "keypoint_count_a",
    "keypoint_count_b",
    "candidate_match_count",
    "geometric_inlier_count",
    "inlier_ratio",
    "geometric_model_status",
)


class SiftDerivedProtocolError(ValueError):
    """Raised when the frozen SIFT-derived protocol cannot be proven safe."""


@dataclass(frozen=True)
class PrimaryBundle:
    dataset: str
    protocol: str
    manifest_path: Path
    bundle_path: Path
    pairs: list[PairRecord]
    rows: list[dict[str, str]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetSelection:
    dataset: str
    threshold: float
    included_identities: tuple[Identity, ...]
    included_rows: list[dict[str, str]]
    excluded_rows: list[dict[str, str]]
    reason_counts: dict[str, int]
    reason_overlap_counts: dict[str, int]
    plain_self_accepted_count: int
    roll_self_accepted_count: int
    identity_universe_count: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiftDerivedProtocolError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SiftDerivedProtocolError(f"JSON artifact must contain an object: {path}")
    return payload


def _identity(pair: PairRecord) -> Identity:
    return pair.subject_id, pair.canonical_finger_position


def _threshold(decision_rule: Mapping[str, Any], dataset: str) -> float:
    try:
        value = float(decision_rule["thresholds_by_dataset"][dataset]["primary_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SiftDerivedProtocolError(f"Frozen decision rule lacks a valid threshold for {dataset}.") from exc
    if not math.isfinite(value):
        raise SiftDerivedProtocolError(f"Frozen decision threshold is non-finite for {dataset}.")
    return value


def validate_frozen_artifacts(
    project_root: Path,
    *,
    config_path: Path | None = None,
    decision_path: Path | None = None,
) -> dict[str, Any]:
    """Validate both required file hashes and the embedded decision-rule hash."""

    config_path = (config_path or project_root / DEFAULT_CONFIG).resolve()
    decision_path = (decision_path or project_root / DEFAULT_DECISION).resolve()
    config_sha = file_sha256(config_path)
    decision_sha = file_sha256(decision_path)
    if config_sha != EXPECTED_CONFIG_FILE_SHA256:
        raise SiftDerivedProtocolError(
            f"Frozen SIFT config hash mismatch: expected {EXPECTED_CONFIG_FILE_SHA256}, got {config_sha}."
        )
    if decision_sha != EXPECTED_DECISION_FILE_SHA256:
        raise SiftDerivedProtocolError(
            f"Frozen SIFT decision-rule hash mismatch: expected {EXPECTED_DECISION_FILE_SHA256}, got {decision_sha}."
        )
    config_payload = _read_json(config_path)
    config = SiftGeometricConfig(**config_payload)
    decision = _read_json(decision_path)
    embedded_hash = str(decision.get("decision_rule_hash", ""))
    unhashed = {key: value for key, value in decision.items() if key != "decision_rule_hash"}
    calculated_hash = stable_hash(unhashed)
    if embedded_hash != calculated_hash:
        raise SiftDerivedProtocolError(
            f"Embedded decision-rule hash mismatch: expected {embedded_hash}, calculated {calculated_hash}."
        )
    if decision.get("acceptance_operator") != "raw_score >= dataset_threshold":
        raise SiftDerivedProtocolError("Frozen SIFT acceptance operator changed.")
    thresholds = {dataset: _threshold(decision, dataset) for dataset in DATASETS}
    return {
        "config_path": str(config_path),
        "config_file_sha256": config_sha,
        "decision_rule_path": str(decision_path),
        "decision_rule_file_sha256": decision_sha,
        "embedded_decision_rule_hash": embedded_hash,
        "config": config,
        "config_payload": config_payload,
        "decision_rule": decision,
        "thresholds": thresholds,
    }


def load_primary_bundle(
    project_root: Path,
    data_root: Path,
    dataset: str,
    protocol: str,
    *,
    validate_source_manifest: bool = True,
) -> PrimaryBundle:
    """Load and fully validate one immutable primary SIFT bundle."""

    if dataset not in DATASETS or protocol not in PROTOCOLS:
        raise SiftDerivedProtocolError(f"Unsupported primary condition: {dataset}/{protocol}")
    root = (
        project_root
        / "results"
        / dataset
        / protocol
        / METHOD_NAME
        / BENCHMARK_CONTRACT_VERSION
    )
    bundles = sorted(
        metadata.parent
        for metadata in root.glob(f"*/{METADATA_FILENAME}")
        if (metadata.parent / RESULT_FILENAME).is_file()
    )
    if len(bundles) != 1:
        raise SiftDerivedProtocolError(
            f"Expected exactly one primary SIFT bundle for {dataset}/{protocol}, found {len(bundles)}."
        )
    bundle = bundles[0].resolve()
    metadata = _read_json(bundle / METADATA_FILENAME)
    raw_spec = dict(metadata.get("run_spec", {}))
    if not raw_spec:
        raise SiftDerivedProtocolError(f"Primary metadata lacks run_spec: {bundle}")
    raw_spec["manifest_path"] = Path(raw_spec["manifest_path"])
    spec = BenchmarkRunSpec(**raw_spec)
    expected_manifest = (project_root / "protocols" / dataset / f"{protocol}.csv").resolve()
    if (
        spec.expected_dataset != dataset
        or spec.expected_protocol != protocol
        or spec.method != METHOD_NAME
        or spec.method_version != METHOD_VERSION
    ):
        raise SiftDerivedProtocolError(f"Primary run identity mismatch for {dataset}/{protocol}.")
    if spec.manifest_path.resolve() != expected_manifest:
        raise SiftDerivedProtocolError(f"Primary run points to the wrong manifest: {bundle}")
    if file_sha256(expected_manifest) != spec.manifest_sha256:
        raise SiftDerivedProtocolError(f"Primary manifest SHA-256 mismatch for {dataset}/{protocol}.")
    if validate_source_manifest:
        validator_for(dataset, protocol)(expected_manifest, data_root)
    pairs = read_pair_manifest(expected_manifest)
    validate_result_bundle(
        bundle,
        manifest_records=pairs,
        run_spec=spec,
        score_direction=metadata["score_direction"],
        score_semantics=metadata["score_semantics"],
    )
    rows = read_result_rows(bundle / RESULT_FILENAME)
    if [pair.pair_id for pair in pairs] != [row["pair_id"] for row in rows]:
        raise SiftDerivedProtocolError(f"Pair-ID alignment failure for {dataset}/{protocol}.")
    identities: set[Identity] = set()
    for pair, row in zip(pairs, rows, strict=True):
        identity = _identity(pair)
        if identity in identities:
            raise SiftDerivedProtocolError(f"Duplicate identity in {dataset}/{protocol}: {identity}")
        identities.add(identity)
        if row["subject_id"] != identity[0] or int(row["canonical_finger_position"]) != identity[1]:
            raise SiftDerivedProtocolError(f"Result identity mismatch for {pair.pair_id}.")
    return PrimaryBundle(dataset, protocol, expected_manifest, bundle, pairs, rows, metadata)


def load_six_primary_bundles(
    project_root: Path,
    data_root: Path,
    frozen: Mapping[str, Any],
    *,
    validate_source_manifests: bool = True,
) -> dict[tuple[str, str], PrimaryBundle]:
    """Validate the six bundles and their uniform frozen implementation/config."""

    bundles = {
        (dataset, protocol): load_primary_bundle(
            project_root,
            data_root,
            dataset,
            protocol,
            validate_source_manifest=validate_source_manifests,
        )
        for dataset in DATASETS
        for protocol in PROTOCOLS
    }
    config_hashes = {bundle.metadata["config_hash"] for bundle in bundles.values()}
    implementation_hashes = {bundle.metadata["implementation_hash"] for bundle in bundles.values()}
    if len(config_hashes) != 1:
        raise SiftDerivedProtocolError(f"Primary SIFT config hashes are not uniform: {sorted(config_hashes)}")
    if len(implementation_hashes) != 1:
        raise SiftDerivedProtocolError(
            f"Primary SIFT implementation hashes are not uniform: {sorted(implementation_hashes)}"
        )
    for condition, bundle in bundles.items():
        effective = bundle.metadata.get("config", {})
        for key, expected in frozen["config_payload"].items():
            if effective.get(key) != expected:
                raise SiftDerivedProtocolError(
                    f"Primary effective config differs from frozen config at {condition}/{key}."
                )
        if bundle.metadata.get("external_runtime", {}).get("opencv_version") is None:
            raise SiftDerivedProtocolError(f"Primary OpenCV provenance is missing for {condition}.")
    return bundles


def _index_bundle(bundle: PrimaryBundle) -> dict[Identity, tuple[PairRecord, dict[str, str]]]:
    output: dict[Identity, tuple[PairRecord, dict[str, str]]] = {}
    for pair, row in zip(bundle.pairs, bundle.rows, strict=True):
        identity = _identity(pair)
        if identity in output:
            raise SiftDerivedProtocolError(f"Duplicate identity in {bundle.dataset}/{bundle.protocol}: {identity}")
        output[identity] = (pair, row)
    return output


def _raw_score(row: Mapping[str, str], label: str) -> float:
    raw = row.get("raw_score", "")
    if raw == "":
        raise SiftDerivedProtocolError(f"Successful row lacks raw_score: {label}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise SiftDerivedProtocolError(f"Invalid raw_score in {label}: {raw!r}") from exc
    if not math.isfinite(value):
        raise SiftDerivedProtocolError(f"Non-finite raw_score in {label}.")
    return value


def frozen_decision(row: Mapping[str, str], threshold: float) -> str:
    """Apply only the frozen SIFT status-plus-threshold decision."""

    if row.get("status") != "ok":
        return "rejected"
    return "accepted" if _raw_score(row, str(row.get("pair_id", "row"))) >= threshold else "rejected"


def _compare_diagnostics(row: Mapping[str, str]) -> dict[str, Any]:
    raw = row.get("compare_diagnostics", "")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SiftDerivedProtocolError(f"Invalid compare diagnostics for {row.get('pair_id')}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SiftDerivedProtocolError(f"Compare diagnostics are not an object for {row.get('pair_id')}.")
    return payload


def deterministic_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project SIFT diagnostics onto fields guaranteed deterministic by this protocol."""

    success = payload.get("geometry_success")
    failure = payload.get("geometry_failure_reason")
    if success is True:
        model_status: Any = "success"
    elif success is False:
        model_status = failure or "failure"
    else:
        model_status = None
    return {
        "keypoint_count_a": payload.get("keypoint_count_a"),
        "keypoint_count_b": payload.get("keypoint_count_b"),
        "candidate_match_count": payload.get("matches_submitted_to_geometry"),
        "geometric_inlier_count": payload.get("geometric_inlier_count"),
        "inlier_ratio": payload.get("inlier_ratio"),
        "geometric_model_status": model_status,
    }


def full_deterministic_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add residual summaries required by the full primary/rerun comparison."""

    return {
        **deterministic_diagnostics(payload),
        "residual_destination_pixels": payload.get("residual_destination_pixels"),
        "residual_reference_pixels": payload.get("residual_reference_pixels"),
    }


def _provenance_fields(prefix: str, item: tuple[PairRecord, dict[str, str]] | None) -> dict[str, str]:
    if item is None:
        return {
            f"{prefix}_pair_id": "",
            f"{prefix}_decision": "",
            f"{prefix}_status": "",
            f"{prefix}_raw_score": "",
            **{f"{prefix}_{field}": "" for field in DETERMINISTIC_DIAGNOSTIC_FIELDS},
        }
    pair, row = item
    diagnostics = deterministic_diagnostics(_compare_diagnostics(row))
    return {
        f"{prefix}_pair_id": pair.pair_id,
        f"{prefix}_decision": "",  # assigned by the caller with the dataset threshold
        f"{prefix}_status": row["status"],
        f"{prefix}_raw_score": row["raw_score"],
        **{
            f"{prefix}_{field}": "" if diagnostics[field] is None else str(diagnostics[field])
            for field in DETERMINISTIC_DIAGNOSTIC_FIELDS
        },
    }


def select_dataset_identities(
    dataset: str,
    plain_self: PrimaryBundle,
    roll_self: PrimaryBundle,
    base_plain_roll: PrimaryBundle,
    *,
    threshold: float,
) -> DatasetSelection:
    """Filter one dataset independently using only its two frozen self decisions."""

    if dataset not in DATASETS:
        raise SiftDerivedProtocolError(f"Wrong dataset: {dataset}")
    for bundle, expected_protocol in (
        (plain_self, "plain_self"),
        (roll_self, "roll_self"),
        (base_plain_roll, "plain_roll"),
    ):
        if bundle.dataset != dataset or bundle.protocol != expected_protocol:
            raise SiftDerivedProtocolError(
                f"Wrong bundle supplied for {dataset}/{expected_protocol}: {bundle.dataset}/{bundle.protocol}"
            )
    plain = _index_bundle(plain_self)
    roll = _index_bundle(roll_self)
    plain_roll = _index_bundle(base_plain_roll)
    universe = set(plain) | set(roll) | set(plain_roll)
    included_set: set[Identity] = set()
    excluded_rows: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()

    for identity in sorted(universe):
        plain_item = plain.get(identity)
        roll_item = roll.get(identity)
        pair_item = plain_roll.get(identity)
        reasons: list[str] = []
        if plain_item is None:
            reasons.append("missing_plain_self_identity")
        elif plain_item[1]["status"] != "ok":
            reasons.append("plain_self_non_ok")
        elif frozen_decision(plain_item[1], threshold) != "accepted":
            reasons.append("plain_self_rejected")
        if roll_item is None:
            reasons.append("missing_roll_self_identity")
        elif roll_item[1]["status"] != "ok":
            reasons.append("roll_self_non_ok")
        elif frozen_decision(roll_item[1], threshold) != "accepted":
            reasons.append("roll_self_rejected")
        if pair_item is None:
            reasons.append("missing_plain_roll_pair")

        common = {
            "dataset": dataset,
            "subject_id": identity[0],
            "canonical_finger_position": str(identity[1]),
            **_provenance_fields("plain_self", plain_item),
            **_provenance_fields("roll_self", roll_item),
            "base_plain_roll_pair_id": pair_item[0].pair_id if pair_item is not None else "",
        }
        if plain_item is not None:
            common["plain_self_decision"] = frozen_decision(plain_item[1], threshold)
        if roll_item is not None:
            common["roll_self_decision"] = frozen_decision(roll_item[1], threshold)
        common = {column: common.get(column, "") for column in INCLUDED_COLUMNS}
        if reasons:
            reason_counts.update(reasons)
            overlap_counts[";".join(reasons)] += 1
            excluded_rows.append({**common, "reason_flags": ";".join(reasons)})
        else:
            included_set.add(identity)

    included_identities = tuple(
        _identity(pair) for pair in base_plain_roll.pairs if _identity(pair) in included_set
    )
    if set(included_identities) != included_set:
        raise SiftDerivedProtocolError(f"Eligible identity completeness failure for {dataset}.")
    included_rows: list[dict[str, str]] = []
    for identity in included_identities:
        plain_item = plain[identity]
        roll_item = roll[identity]
        pair_item = plain_roll[identity]
        common = {
            "dataset": dataset,
            "subject_id": identity[0],
            "canonical_finger_position": str(identity[1]),
            **_provenance_fields("plain_self", plain_item),
            **_provenance_fields("roll_self", roll_item),
            "base_plain_roll_pair_id": pair_item[0].pair_id,
        }
        common["plain_self_decision"] = frozen_decision(plain_item[1], threshold)
        common["roll_self_decision"] = frozen_decision(roll_item[1], threshold)
        included_rows.append({column: common.get(column, "") for column in INCLUDED_COLUMNS})
    return DatasetSelection(
        dataset=dataset,
        threshold=threshold,
        included_identities=included_identities,
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        reason_counts=dict(sorted(reason_counts.items())),
        reason_overlap_counts=dict(sorted(overlap_counts.items())),
        plain_self_accepted_count=sum(
            item[1]["status"] == "ok" and frozen_decision(item[1], threshold) == "accepted"
            for item in plain.values()
        ),
        roll_self_accepted_count=sum(
            item[1]["status"] == "ok" and frozen_decision(item[1], threshold) == "accepted"
            for item in roll.values()
        ),
        identity_universe_count=len(universe),
    )


def _physical_manifest_rows(path: Path) -> tuple[bytes, list[tuple[str, bytes, list[str]]]]:
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if not lines:
        raise SiftDerivedProtocolError(f"Manifest is empty: {path}")

    def parse(raw: bytes, line_number: int) -> list[str]:
        try:
            decoded = raw.decode("utf-8")
            records = list(csv.reader(io.StringIO(decoded, newline="")))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise SiftDerivedProtocolError(f"Cannot parse {path}:{line_number}: {exc}") from exc
        if len(records) != 1:
            raise SiftDerivedProtocolError(f"Multiline CSV rows are unsupported in {path}.")
        return records[0]

    header = parse(lines[0], 1)
    if header != MANIFEST_COLUMNS:
        raise SiftDerivedProtocolError(f"Manifest schema mismatch in {path}: {header}")
    rows: list[tuple[str, bytes, list[str]]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(lines[1:], start=2):
        values = parse(raw, line_number)
        if len(values) != len(MANIFEST_COLUMNS):
            raise SiftDerivedProtocolError(f"Manifest field-count mismatch at {path}:{line_number}.")
        pair_id = values[0]
        if pair_id in seen:
            raise SiftDerivedProtocolError(f"Duplicate pair_id {pair_id!r} in {path}.")
        seen.add(pair_id)
        rows.append((pair_id, raw, values))
    if len(rows) != len(read_pair_manifest(path)):
        raise SiftDerivedProtocolError(f"Physical/logical row-count mismatch in {path}.")
    return lines[0], rows


def filter_manifest_bytes(base_manifest: Path, included_pair_ids: set[str]) -> bytes:
    """Return a byte-exact, source-ordered row subset of a base manifest."""

    header, rows = _physical_manifest_rows(base_manifest)
    known = {pair_id for pair_id, _, _ in rows}
    missing = included_pair_ids - known
    if missing:
        raise SiftDerivedProtocolError(f"Selected pairs are absent from base manifest: {sorted(missing)[:5]}")
    selected = [raw for pair_id, raw, _ in rows if pair_id in included_pair_ids]
    if len(selected) != len(included_pair_ids):
        raise SiftDerivedProtocolError("Derived manifest selection count mismatch.")
    return header + b"".join(selected)


def validate_exact_manifest_subset(
    derived_manifest: Path,
    base_manifest: Path,
    *,
    expected_dataset: str,
    expected_pair_ids: Sequence[str],
) -> dict[str, Any]:
    """Prove exact schema, rows, source order, identities, orientation, and completeness."""

    base_header, base_rows = _physical_manifest_rows(base_manifest)
    derived_header, derived_rows = _physical_manifest_rows(derived_manifest)
    if derived_header != base_header:
        raise SiftDerivedProtocolError("Derived manifest header bytes differ from its base manifest.")
    base_lookup = {pair_id: (index, raw, values) for index, (pair_id, raw, values) in enumerate(base_rows)}
    actual_pair_ids: list[str] = []
    source_indexes: list[int] = []
    identities: set[Identity] = set()
    for pair_id, raw, values in derived_rows:
        source = base_lookup.get(pair_id)
        if source is None or source[1] != raw:
            raise SiftDerivedProtocolError(f"Derived row is not a byte-exact base row: {pair_id}")
        source_indexes.append(source[0])
        actual_pair_ids.append(pair_id)
        if values[1] != expected_dataset or values[2] != "plain_roll":
            raise SiftDerivedProtocolError(f"Wrong dataset/protocol in derived row {pair_id}.")
        identity = (values[3], int(values[4]))
        if identity in identities:
            raise SiftDerivedProtocolError(f"Duplicate identity in derived manifest: {identity}")
        identities.add(identity)
        path_a, path_b = Path(values[8]), Path(values[9])
        if "plain" not in str(path_a).lower() or "roll" not in str(path_b).lower():
            raise SiftDerivedProtocolError(f"Derived pair is not A=PLAIN/B=ROLL: {pair_id}")
    if source_indexes != sorted(source_indexes):
        raise SiftDerivedProtocolError("Derived manifest does not preserve source order.")
    if actual_pair_ids != list(expected_pair_ids):
        raise SiftDerivedProtocolError("Derived manifest is incomplete or contains an ineligible pair.")
    pairs = read_pair_manifest(derived_manifest)
    expected_ppi = 1000 if expected_dataset == "sd300b" else 2000
    if any(pair.ppi != expected_ppi for pair in pairs):
        raise SiftDerivedProtocolError(f"Wrong PPI in {expected_dataset} derived manifest.")
    return {
        "validation_mode": "exact_ordered_byte_for_row_subset",
        "schema_exact": True,
        "base_manifest_path": str(base_manifest.resolve()),
        "base_manifest_sha256": file_sha256(base_manifest),
        "derived_manifest_path": str(derived_manifest.resolve()),
        "derived_manifest_sha256": file_sha256(derived_manifest),
        "pair_count": len(actual_pair_ids),
        "identity_count": len(identities),
        "pair_ids_unique": len(actual_pair_ids) == len(set(actual_pair_ids)),
        "identities_unique": len(actual_pair_ids) == len(identities),
        "source_order_preserved": True,
        "source_rows_byte_exact": True,
        "eligible_identity_completeness": True,
        "ineligible_identity_exclusion": True,
        "plain_roll_outcome_used_for_selection": False,
        "orientation": "A=plain,B=roll",
    }


def make_derived_manifest_validator(
    *,
    base_manifest: Path,
    expected_dataset: str,
    expected_manifest_sha256: str,
    expected_pair_ids: Sequence[str],
) -> Callable[[Path, Path], dict[str, Any]]:
    def validate(derived_manifest: Path, data_root: Path) -> dict[str, Any]:
        if file_sha256(derived_manifest) != expected_manifest_sha256:
            raise SiftDerivedProtocolError("Derived manifest SHA-256 changed after publication.")
        source_report = validator_for(expected_dataset, "plain_roll")(base_manifest, data_root)
        report = validate_exact_manifest_subset(
            derived_manifest,
            base_manifest,
            expected_dataset=expected_dataset,
            expected_pair_ids=expected_pair_ids,
        )
        report["source_manifest_validator_result"] = (
            asdict(source_report) if hasattr(source_report, "__dataclass_fields__") else source_report
        )
        return report

    return validate


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _publish_immutable_candidate(candidate: Path, final: Path) -> None:
    if final.exists():
        if _directory_hashes(candidate) != _directory_hashes(final):
            raise BundlePublicationError(f"Immutable artifact exists with different content: {final}")
        discard_candidate_directory(candidate)
        return
    publish_candidate_directory(candidate, final)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SiftDerivedProtocolError(f"Immutable artifact already exists with different bytes: {path}")
        return
    _write_bytes(path, payload)


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise SiftDerivedProtocolError(
                    f"CSV schema mismatch in {path}: expected {columns}, got {reader.fieldnames}."
                )
            rows = list(reader)
    except OSError as exc:
        raise SiftDerivedProtocolError(f"Cannot read CSV {path}: {exc}") from exc
    if any(None in row for row in rows):
        raise SiftDerivedProtocolError(f"CSV has unnamed extra fields: {path}")
    return rows


def _stringify(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: "" if value is None else ("true" if value is True else "false" if value is False else str(value))
        for key, value in row.items()
    }


def _source_hashes(bundles: Mapping[tuple[str, str], PrimaryBundle]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for (dataset, protocol), bundle in sorted(bundles.items()):
        hashes[f"{dataset}/{protocol}/manifest"] = file_sha256(bundle.manifest_path)
        hashes[f"{dataset}/{protocol}/pairs"] = file_sha256(bundle.bundle_path / RESULT_FILENAME)
        hashes[f"{dataset}/{protocol}/metadata"] = file_sha256(bundle.bundle_path / METADATA_FILENAME)
    return hashes


def prepare_protocol(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
) -> dict[str, Any]:
    """Validate all frozen inputs, derive both datasets, and publish atomically."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    final_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    frozen = validate_frozen_artifacts(project_root)
    bundles = load_six_primary_bundles(project_root, data_root, frozen)
    candidate = create_candidate_directory(final_root)
    try:
        dataset_summaries: list[dict[str, Any]] = []
        for dataset in DATASETS:
            threshold = frozen["thresholds"][dataset]
            selection = select_dataset_identities(
                dataset,
                bundles[(dataset, "plain_self")],
                bundles[(dataset, "roll_self")],
                bundles[(dataset, "plain_roll")],
                threshold=threshold,
            )
            included_pair_ids = [row["base_plain_roll_pair_id"] for row in selection.included_rows]
            base = bundles[(dataset, "plain_roll")]
            dataset_dir = candidate / dataset
            manifest = dataset_dir / "plain_roll.csv"
            _write_bytes(manifest, filter_manifest_bytes(base.manifest_path, set(included_pair_ids)))
            write_csv_atomic(selection.included_rows, dataset_dir / "included_identities.csv", INCLUDED_COLUMNS)
            write_csv_atomic(selection.excluded_rows, dataset_dir / "excluded_identities.csv", EXCLUDED_COLUMNS)
            validation = validate_exact_manifest_subset(
                manifest,
                base.manifest_path,
                expected_dataset=dataset,
                expected_pair_ids=included_pair_ids,
            )
            summary = {
                "dataset": dataset,
                "threshold": threshold,
                "identity_key": ["subject_id", "canonical_finger_position"],
                "selection_rule": "plain_self status=ok and frozen decision=accepted; roll_self status=ok and frozen decision=accepted; base plain_roll pair exists",
                "plain_roll_score_used_for_selection": False,
                "plain_self_total_count": len(bundles[(dataset, "plain_self")].rows),
                "plain_self_accepted_count": selection.plain_self_accepted_count,
                "roll_self_total_count": len(bundles[(dataset, "roll_self")].rows),
                "roll_self_accepted_count": selection.roll_self_accepted_count,
                "base_plain_roll_count": len(base.pairs),
                "identity_universe_count": selection.identity_universe_count,
                "included_identity_count": len(selection.included_identities),
                "excluded_identity_count": len(selection.excluded_rows),
                "exclusion_reason_counts": selection.reason_counts,
                "exclusion_reason_overlap_counts": selection.reason_overlap_counts,
                "derived_manifest_relative_path": f"{dataset}/plain_roll.csv",
                "derived_manifest_sha256": validation["derived_manifest_sha256"],
                "included_identities_sha256": file_sha256(dataset_dir / "included_identities.csv"),
                "excluded_identities_sha256": file_sha256(dataset_dir / "excluded_identities.csv"),
                "validation": validation,
            }
            summary["validation"]["derived_manifest_path"] = str(
                (final_root / dataset / "plain_roll.csv").resolve()
            )
            dataset_summaries.append(summary)

        config_hashes = {bundle.metadata["config_hash"] for bundle in bundles.values()}
        implementation_hashes = {bundle.metadata["implementation_hash"] for bundle in bundles.values()}
        payload = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "namespace": PROTOCOL_NAMESPACE,
            "method": METHOD_NAME,
            "method_version": METHOD_VERSION,
            "datasets_filtered_independently": True,
            "cross_dataset_identity_equality_required": False,
            "plain_roll_score_used_for_selection": False,
            "base_rows_preserved_byte_exactly": True,
            "base_row_order_preserved": True,
            "deterministic_output": True,
            "frozen_config_path": frozen["config_path"],
            "frozen_config_file_sha256": frozen["config_file_sha256"],
            "frozen_decision_rule_path": frozen["decision_rule_path"],
            "frozen_decision_rule_file_sha256": frozen["decision_rule_file_sha256"],
            "embedded_decision_rule_hash": frozen["embedded_decision_rule_hash"],
            "thresholds_by_dataset": frozen["thresholds"],
            "primary_config_hash": next(iter(config_hashes)),
            "primary_implementation_hash": next(iter(implementation_hashes)),
            "source_artifacts_sha256": _source_hashes(bundles),
            "datasets": dataset_summaries,
        }
        write_json_atomic(payload, candidate / "protocol_summary.json")
        write_csv_atomic(
            [
                _stringify({column: summary[column] for column in PROTOCOL_SUMMARY_COLUMNS})
                for summary in dataset_summaries
            ],
            candidate / "protocol_summary.csv",
            PROTOCOL_SUMMARY_COLUMNS,
        )
        _publish_immutable_candidate(candidate, final_root)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)
    return load_and_validate_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=final_root,
        validate_sources=False,
    )


def load_and_validate_protocol(
    *,
    project_root: Path,
    data_root: Path,
    protocol_root: Path,
    validate_sources: bool = True,
) -> dict[str, Any]:
    protocol_root = protocol_root.resolve()
    summary_path = protocol_root / "protocol_summary.json"
    summary = _read_json(summary_path)
    if summary.get("schema_version") != PROTOCOL_SCHEMA_VERSION or summary.get("namespace") != PROTOCOL_NAMESPACE:
        raise SiftDerivedProtocolError("Derived protocol summary schema/namespace mismatch.")
    if summary.get("plain_roll_score_used_for_selection") is not False:
        raise SiftDerivedProtocolError("Derived protocol illegally uses plain-roll outcomes for selection.")
    frozen = validate_frozen_artifacts(project_root)
    if (
        summary.get("frozen_config_file_sha256") != frozen["config_file_sha256"]
        or summary.get("frozen_decision_rule_file_sha256") != frozen["decision_rule_file_sha256"]
    ):
        raise SiftDerivedProtocolError("Frozen config/decision provenance changed after publication.")
    bundles = (
        load_six_primary_bundles(project_root, data_root, frozen)
        if validate_sources
        else {}
    )
    reports: list[dict[str, Any]] = []
    summaries = {item["dataset"]: item for item in summary.get("datasets", [])}
    if set(summaries) != set(DATASETS):
        raise SiftDerivedProtocolError("Protocol summary does not contain exactly both datasets.")
    for dataset in DATASETS:
        dataset_summary = summaries[dataset]
        manifest = protocol_root / dataset / "plain_roll.csv"
        included_path = protocol_root / dataset / "included_identities.csv"
        excluded_path = protocol_root / dataset / "excluded_identities.csv"
        included = _read_csv(included_path, INCLUDED_COLUMNS)
        excluded = _read_csv(excluded_path, EXCLUDED_COLUMNS)
        expected_pair_ids = [row["base_plain_roll_pair_id"] for row in included]
        if file_sha256(manifest) != dataset_summary["derived_manifest_sha256"]:
            raise SiftDerivedProtocolError(f"Derived manifest hash mismatch for {dataset}.")
        if file_sha256(included_path) != dataset_summary["included_identities_sha256"]:
            raise SiftDerivedProtocolError(f"Included provenance hash mismatch for {dataset}.")
        if file_sha256(excluded_path) != dataset_summary["excluded_identities_sha256"]:
            raise SiftDerivedProtocolError(f"Excluded provenance hash mismatch for {dataset}.")
        base_manifest = project_root / "protocols" / dataset / "plain_roll.csv"
        report = validate_exact_manifest_subset(
            manifest,
            base_manifest,
            expected_dataset=dataset,
            expected_pair_ids=expected_pair_ids,
        )
        if validate_sources:
            selection = select_dataset_identities(
                dataset,
                bundles[(dataset, "plain_self")],
                bundles[(dataset, "roll_self")],
                bundles[(dataset, "plain_roll")],
                threshold=frozen["thresholds"][dataset],
            )
            if selection.included_rows != included or selection.excluded_rows != excluded:
                raise SiftDerivedProtocolError(f"Published eligibility/provenance is incomplete for {dataset}.")
            current_hashes = _source_hashes(bundles)
            if current_hashes != summary["source_artifacts_sha256"]:
                raise SiftDerivedProtocolError("A protected primary SIFT source artifact changed.")
        reports.append(report)
    return {
        "protocol_root": str(protocol_root),
        "protocol_summary_path": str(summary_path),
        "protocol_summary_sha256": file_sha256(summary_path),
        "datasets": reports,
        "summary": summary,
    }


def deterministic_sample_indices(row_count: int, sample_count: int = 30) -> list[int]:
    """Return floor(i*N/sample_count) for a documented, uniformly spaced sample."""

    if sample_count <= 0 or row_count < sample_count:
        raise SiftDerivedProtocolError(
            f"Cannot select {sample_count} unique samples from {row_count} rows."
        )
    indexes = [math.floor(index * row_count / sample_count) for index in range(sample_count)]
    if len(indexes) != len(set(indexes)):
        raise SiftDerivedProtocolError("Deterministic sample indexes are not unique.")
    return indexes


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _hardware_support() -> dict[str, dict[str, Any]]:
    # cv::CpuFeatures numeric values are part of OpenCV's public API.  Python
    # wheels do not consistently expose the CPU_* symbolic constants.
    features = {
        "SSE4_2": 7,
        "AVX2": 11,
        "FMA3": 12,
        "AVX512F": 13,
    }
    output: dict[str, dict[str, Any]] = {}
    for name, feature_id in features.items():
        try:
            supported: bool | None = bool(cv2.checkHardwareSupport(feature_id))
        except (AttributeError, cv2.error):
            supported = None
        output[name] = {"opencv_cpu_feature_id": feature_id, "supported": supported}
    return output


def runtime_environment_provenance() -> dict[str, Any]:
    build = cv2.getBuildInformation()
    return {
        "opencv_version": cv2.__version__,
        "opencv_build_information_sha256": hashlib.sha256(build.encode("utf-8")).hexdigest(),
        "use_optimized": bool(cv2.useOptimized()),
        "num_threads": int(cv2.getNumThreads()),
        "cpu_model": _cpu_model(),
        "instruction_sets": _hardware_support(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def _diagnostics_from_result_row(
    row: Mapping[str, str], *, include_residuals: bool = False
) -> dict[str, Any]:
    payload = _compare_diagnostics(row)
    return (
        full_deterministic_diagnostics(payload)
        if include_residuals
        else deterministic_diagnostics(payload)
    )


def _score_delta(primary: Mapping[str, str], derived: Mapping[str, str]) -> float | None:
    if primary.get("raw_score", "") == "" or derived.get("raw_score", "") == "":
        return None
    return abs(float(primary["raw_score"]) - float(derived["raw_score"]))


def compare_result_rows(
    primary: Mapping[str, str],
    derived: Mapping[str, str],
    *,
    threshold: float,
    include_residuals: bool = False,
) -> dict[str, Any]:
    """Compare one rerun row with its primary row under the frozen contract."""

    if primary["pair_id"] != derived["pair_id"]:
        raise SiftDerivedProtocolError("Result-row comparison requires identical pair_id values.")
    delta = _score_delta(primary, derived)
    primary_decision = frozen_decision(primary, threshold)
    derived_decision = frozen_decision(derived, threshold)
    primary_diagnostics = _diagnostics_from_result_row(
        primary, include_residuals=include_residuals
    )
    derived_diagnostics = _diagnostics_from_result_row(
        derived, include_residuals=include_residuals
    )
    score_text_equal = primary.get("raw_score", "") == derived.get("raw_score", "")
    score_numeric_equal = delta == 0.0 if delta is not None else score_text_equal
    status_equal = primary["status"] == derived["status"]
    error_equal = primary.get("error_code", "") == derived.get("error_code", "")
    decision_equal = primary_decision == derived_decision
    diagnostics_equal = primary_diagnostics == derived_diagnostics
    return {
        "pair_id": derived["pair_id"],
        "subject_id": derived["subject_id"],
        "canonical_finger_position": derived["canonical_finger_position"],
        "primary_raw_score": primary.get("raw_score", ""),
        "derived_raw_score": derived.get("raw_score", ""),
        "exact_score_text_equal": score_text_equal,
        "exact_score_numeric_equal": score_numeric_equal,
        "absolute_score_delta": delta,
        "primary_decision": primary_decision,
        "derived_decision": derived_decision,
        "decision_equal": decision_equal,
        "primary_status": primary["status"],
        "derived_status": derived["status"],
        "status_equal": status_equal,
        "primary_error_code": primary.get("error_code", ""),
        "derived_error_code": derived.get("error_code", ""),
        "error_code_equal": error_equal,
        "primary_diagnostics": primary_diagnostics,
        "derived_diagnostics": derived_diagnostics,
        "diagnostics_equal": diagnostics_equal,
        "passed": (
            score_text_equal
            and score_numeric_equal
            and decision_equal
            and status_equal
            and error_equal
            and diagnostics_equal
        ),
    }


def _primary_environment(bundle: PrimaryBundle) -> dict[str, Any]:
    runtime = bundle.metadata.get("external_runtime", {})
    build = str(runtime.get("opencv_build_information", ""))
    return {
        "opencv_version": runtime.get("opencv_version"),
        "opencv_build_information_sha256": hashlib.sha256(build.encode("utf-8")).hexdigest(),
        "use_optimized": runtime.get("opencv_optimized"),
        "num_threads": runtime.get("opencv_thread_count"),
        "numpy_version": runtime.get("numpy_version"),
    }


def runtime_preflight(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
    sample_count: int = 30,
) -> dict[str, Any]:
    """Run the mandatory exact 30+30 numerical gate without publishing bundles."""

    if sample_count != 30:
        raise SiftDerivedProtocolError("The SIFT runtime preflight requires exactly 30 pairs per dataset.")
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    protocol = load_and_validate_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=protocol_root,
        validate_sources=True,
    )
    frozen = validate_frozen_artifacts(project_root)
    dataset_reports: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        manifest = protocol_root / dataset / "plain_roll.csv"
        pairs = read_pair_manifest(manifest)
        indexes = deterministic_sample_indices(len(pairs), sample_count)
        sampled_pairs = [pairs[index] for index in indexes]
        primary = load_primary_bundle(
            project_root, data_root, dataset, "plain_roll", validate_source_manifest=False
        )
        primary_by_pair = {row["pair_id"]: row for row in primary.rows}
        adapter = SiftGeometricAdapter(frozen["config"])
        try:
            context = prepare_run_context(
                manifest_path=manifest,
                expected_dataset=dataset,
                expected_protocol="plain_roll",
                adapter=adapter,
                results_root=run_root,
                startup_validation={
                    "derived_protocol_namespace": PROTOCOL_NAMESPACE,
                    "runtime_preflight": True,
                },
            )
            current_environment = runtime_environment_provenance()
            primary_environment = _primary_environment(primary)
            environment_checks = {
                "config_hash_equal": context.spec.config_hash == primary.metadata["config_hash"],
                "implementation_hash_equal": (
                    context.spec.implementation_hash == primary.metadata["implementation_hash"]
                ),
                "opencv_version_equal": (
                    current_environment["opencv_version"] == primary_environment["opencv_version"]
                ),
                "opencv_build_information_hash_equal": (
                    current_environment["opencv_build_information_sha256"]
                    == primary_environment["opencv_build_information_sha256"]
                ),
                "use_optimized_equal": (
                    current_environment["use_optimized"] == primary_environment["use_optimized"]
                ),
                "num_threads_equal": (
                    current_environment["num_threads"] == primary_environment["num_threads"]
                ),
            }
            environment_passed = all(environment_checks.values())
            samples: list[dict[str, Any]] = []
            if environment_passed:
                _run_warm_up(sampled_pairs, adapter)
                for pair in sampled_pairs:
                    rerun = _execute_pair(
                        pair,
                        adapter,
                        run_spec=context.spec,
                        method_metadata=context.method_metadata,
                    )
                    sample = compare_result_rows(
                        primary_by_pair[pair.pair_id], rerun, threshold=frozen["thresholds"][dataset]
                    )
                    samples.append(sample)
            mismatches = [sample for sample in samples if not sample["passed"]]
            deltas = [
                sample["absolute_score_delta"]
                for sample in samples
                if sample["absolute_score_delta"] is not None
            ]
            passed = environment_passed and len(samples) == sample_count and not mismatches
            dataset_reports[dataset] = {
                "passed": passed,
                "sample_count": len(samples),
                "sample_indexes": indexes,
                "sample_pair_ids": [pair.pair_id for pair in sampled_pairs],
                "samples": samples,
                "max_absolute_score_delta": max(deltas) if deltas else None,
                "mismatch_count": len(mismatches) + (0 if environment_passed else 1),
                "mismatches": mismatches,
                "environment_passed": environment_passed,
                "environment_checks": environment_checks,
                "environment": current_environment,
                "primary_environment": primary_environment,
                "primary_config_hash": primary.metadata["config_hash"],
                "current_config_hash": context.spec.config_hash,
                "primary_implementation_hash": primary.metadata["implementation_hash"],
                "current_implementation_hash": context.spec.implementation_hash,
            }
        finally:
            adapter.close()

    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "namespace": PROTOCOL_NAMESPACE,
        "protocol_summary_sha256": protocol["protocol_summary_sha256"],
        "sample_selection": "floor(i * N / 30) for i=0..29, independently per dataset",
        "sample_count_per_dataset": sample_count,
        "config_file_sha256": frozen["config_file_sha256"],
        "decision_rule_file_sha256": frozen["decision_rule_file_sha256"],
        "exact_score_text_equality_required": True,
        "max_absolute_score_delta_required": 0.0,
        "decision_status_error_and_diagnostics_exact_equality_required": True,
        "timing_equality_required": False,
        "datasets": dataset_reports,
        "passed": all(item["passed"] for item in dataset_reports.values()),
    }
    output = run_root / "runtime_preflight.json"
    _publish_immutable_bytes(output, _json_bytes(report))
    result = {**report, "path": str(output), "sha256": file_sha256(output)}
    if result["passed"] is not True:
        raise SiftDerivedProtocolError(
            "SIFT runtime preflight failed; full derived runs are forbidden. "
            f"Artifact: {output}"
        )
    return result


def run_protocol(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
    execution_results_root: Path | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Run only the two derived plain-roll manifests after the mandatory gate."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    execution_root = (
        Path(os.path.abspath(execution_results_root))
        if execution_results_root is not None
        else run_root
    )
    if execution_root != run_root:
        if not execution_root.exists() or not os.path.samefile(execution_root, run_root):
            raise SiftDerivedProtocolError(
                "--execution-results-root must be a short path alias to the same physical directory as --run-root."
            )
    preflight_path = run_root / "runtime_preflight.json"
    if not preflight_path.is_file():
        raise SiftDerivedProtocolError("Mandatory runtime_preflight.json is missing.")
    preflight = _read_json(preflight_path)
    if preflight.get("passed") is not True or not all(
        preflight.get("datasets", {}).get(dataset, {}).get("passed") is True for dataset in DATASETS
    ):
        raise SiftDerivedProtocolError("Runtime preflight did not pass for both datasets.")
    protocol = load_and_validate_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=protocol_root,
        validate_sources=True,
    )
    if preflight.get("protocol_summary_sha256") != protocol["protocol_summary_sha256"]:
        raise SiftDerivedProtocolError("Runtime preflight belongs to a different derived protocol.")
    frozen = validate_frozen_artifacts(project_root)
    summaries = {item["dataset"]: item for item in protocol["summary"]["datasets"]}
    runs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        manifest = protocol_root / dataset / "plain_roll.csv"
        included = _read_csv(protocol_root / dataset / "included_identities.csv", INCLUDED_COLUMNS)
        expected_pair_ids = [row["base_plain_roll_pair_id"] for row in included]
        validator = make_derived_manifest_validator(
            base_manifest=project_root / "protocols" / dataset / "plain_roll.csv",
            expected_dataset=dataset,
            expected_manifest_sha256=summaries[dataset]["derived_manifest_sha256"],
            expected_pair_ids=expected_pair_ids,
        )
        primary = load_primary_bundle(
            project_root, data_root, dataset, "plain_roll", validate_source_manifest=False
        )
        adapter = SiftGeometricAdapter(frozen["config"])
        try:
            context = prepare_run_context(
                manifest_path=manifest,
                expected_dataset=dataset,
                expected_protocol="plain_roll",
                adapter=adapter,
                results_root=execution_root,
                startup_validation={
                    "derived_protocol_namespace": PROTOCOL_NAMESPACE,
                    "protocol_summary_sha256": protocol["protocol_summary_sha256"],
                    "runtime_preflight_sha256": file_sha256(preflight_path),
                },
            )
            if context.spec.config_hash != primary.metadata["config_hash"]:
                raise SiftDerivedProtocolError(f"SIFT config hash changed before full {dataset} run.")
            if context.spec.implementation_hash != primary.metadata["implementation_hash"]:
                raise SiftDerivedProtocolError(
                    f"SIFT implementation hash changed before full {dataset} run."
                )
            if (
                runtime_environment_provenance()["opencv_build_information_sha256"]
                != preflight["datasets"][dataset]["environment"]["opencv_build_information_sha256"]
            ):
                raise SiftDerivedProtocolError(f"OpenCV environment changed after {dataset} preflight.")
            metadata = run_benchmark_manifest(
                manifest_path=manifest,
                adapter=adapter,
                expected_dataset=dataset,
                expected_protocol="plain_roll",
                results_root=execution_root,
                startup_validation={
                    "derived_protocol_namespace": PROTOCOL_NAMESPACE,
                    "protocol_summary_sha256": protocol["protocol_summary_sha256"],
                    "runtime_preflight_sha256": file_sha256(preflight_path),
                },
                data_root=data_root,
                dedicated_validator=validator,
                skip_existing=skip_existing,
                progress_callback=lambda completed, total, d=dataset: print(
                    f"[{d}/derived SIFT plain_roll] {completed}/{total}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
        finally:
            adapter.close()
        bundle = (
            run_root
            / dataset
            / "plain_roll"
            / METHOD_NAME
            / BENCHMARK_CONTRACT_VERSION
            / metadata["config_hash"]
        )
        runs.append(
            {
                "dataset": dataset,
                "pair_count": metadata["result"]["row_count"],
                "manifest_sha256": summaries[dataset]["derived_manifest_sha256"],
                "bundle_path": str(bundle.resolve()),
                "pairs_sha256": metadata["result"]["sha256"],
                "score_payload_sha256": metadata["result"]["score_payload_sha256"],
                "config_hash": metadata["config_hash"],
                "implementation_hash": metadata["implementation_hash"],
            }
        )
    report = {
        "schema_version": "sift-derived-run-summary-v1",
        "namespace": PROTOCOL_NAMESPACE,
        "protocol_summary_sha256": protocol["protocol_summary_sha256"],
        "runtime_preflight_sha256": file_sha256(preflight_path),
        "runs": runs,
        "primary_artifacts_overwritten": False,
    }
    output = run_root / "run_summary.json"
    _publish_immutable_bytes(output, _json_bytes(report))
    return {**report, "path": str(output), "sha256": file_sha256(output)}


def load_derived_bundle(
    *, run_root: Path, protocol_root: Path, dataset: str
) -> PrimaryBundle:
    root = run_root / dataset / "plain_roll" / METHOD_NAME / BENCHMARK_CONTRACT_VERSION
    bundles = sorted(
        metadata.parent
        for metadata in root.glob(f"*/{METADATA_FILENAME}")
        if (metadata.parent / RESULT_FILENAME).is_file()
    )
    if len(bundles) != 1:
        raise SiftDerivedProtocolError(f"Expected one derived SIFT bundle for {dataset}, found {len(bundles)}.")
    bundle = bundles[0].resolve()
    metadata = _read_json(bundle / METADATA_FILENAME)
    raw_spec = dict(metadata["run_spec"])
    raw_spec["manifest_path"] = Path(raw_spec["manifest_path"])
    spec = BenchmarkRunSpec(**raw_spec)
    manifest = (protocol_root / dataset / "plain_roll.csv").resolve()
    if spec.manifest_path.resolve() != manifest:
        raise SiftDerivedProtocolError(f"Derived {dataset} bundle points to the wrong manifest.")
    pairs = read_pair_manifest(manifest)
    validate_result_bundle(
        bundle,
        manifest_records=pairs,
        run_spec=spec,
        score_direction=metadata["score_direction"],
        score_semantics=metadata["score_semantics"],
    )
    rows = read_result_rows(bundle / RESULT_FILENAME)
    if [row["pair_id"] for row in rows] != [pair.pair_id for pair in pairs]:
        raise SiftDerivedProtocolError(f"Derived rerun pair alignment failure for {dataset}.")
    return PrimaryBundle(dataset, "plain_roll", manifest, bundle, pairs, rows, metadata)


def compare_derived_to_primary(
    primary_rows: Sequence[Mapping[str, str]],
    derived_rows: Sequence[Mapping[str, str]],
    *,
    dataset: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = {row["pair_id"]: row for row in primary_rows}
    if len(primary) != len(primary_rows):
        raise SiftDerivedProtocolError(f"Duplicate pair_id in primary {dataset} results.")
    if len({row["pair_id"] for row in derived_rows}) != len(derived_rows):
        raise SiftDerivedProtocolError(f"Duplicate pair_id in derived {dataset} results.")
    details: list[dict[str, Any]] = []
    for row in derived_rows:
        expected = primary.get(row["pair_id"])
        if expected is None:
            raise SiftDerivedProtocolError(f"Derived pair is absent from primary result: {row['pair_id']}")
        compared = compare_result_rows(
            expected,
            row,
            threshold=threshold,
            include_residuals=True,
        )
        details.append({"dataset": dataset, **compared})
    deltas = [
        row["absolute_score_delta"]
        for row in details
        if row["absolute_score_delta"] is not None
    ]
    summary = {
        "dataset": dataset,
        "pair_count": len(details),
        "reproducible_pair_count": sum(bool(row["passed"]) for row in details),
        "mismatch_pair_count": sum(not bool(row["passed"]) for row in details),
        "score_text_mismatch_count": sum(not bool(row["exact_score_text_equal"]) for row in details),
        "score_numeric_mismatch_count": sum(not bool(row["exact_score_numeric_equal"]) for row in details),
        "decision_mismatch_count": sum(not bool(row["decision_equal"]) for row in details),
        "status_mismatch_count": sum(not bool(row["status_equal"]) for row in details),
        "error_code_mismatch_count": sum(not bool(row["error_code_equal"]) for row in details),
        "diagnostics_mismatch_count": sum(not bool(row["diagnostics_equal"]) for row in details),
        "max_absolute_score_delta": max(deltas) if deltas else None,
        "passed": all(bool(row["passed"]) for row in details),
    }
    return details, summary


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _numbers(rows: Sequence[Mapping[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row.get(column, "") != ""]


def decision_summary(
    rows: Sequence[Mapping[str, str]], *, dataset: str, threshold: float
) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    scores = [float(row["raw_score"]) for row in ok]
    accepted = sum(score >= threshold for score in scores)
    total = len(rows)
    diagnostics = [_diagnostics_from_result_row(row) for row in ok]
    failures = Counter(row["status"] for row in rows if row["status"] != "ok")
    geometry_failures = sum(
        diag["geometric_model_status"] not in ("success", None) for diag in diagnostics
    )
    result: dict[str, Any] = {
        "dataset": dataset,
        "threshold": threshold,
        "total_pairs": total,
        "status_ok": len(ok),
        "failure_count": total - len(ok),
        "failure_counts_by_stage": dict(sorted(failures.items())),
        "accepted_count": accepted,
        "rejected_count": total - accepted,
        "accepted_percentage": 100.0 * accepted / total if total else None,
        "rejected_percentage": 100.0 * (total - accepted) / total if total else None,
        "raw_score_mean": _mean(scores),
        "raw_score_median": _median(scores),
        "raw_score_min": min(scores) if scores else None,
        "raw_score_max": max(scores) if scores else None,
        "geometric_model_failure_count": geometry_failures,
    }
    for column, prefix in (
        ("prepare_a_ms", "prepare_a_ms"),
        ("prepare_b_ms", "prepare_b_ms"),
        ("compare_ms", "compare_ms"),
        ("total_ms", "total_ms"),
        ("method_compare_ms", "method_compare_ms"),
    ):
        values = _numbers(rows, column)
        result[f"{prefix}_mean"] = _mean(values)
        result[f"{prefix}_median"] = _median(values)
        result[f"{prefix}_p95"] = _p95(values)
    for key, prefix in (
        ("keypoint_count_a", "keypoints_a"),
        ("keypoint_count_b", "keypoints_b"),
        ("candidate_match_count", "candidate_matches"),
        ("geometric_inlier_count", "geometric_inliers"),
    ):
        values = [float(diag[key]) for diag in diagnostics if diag.get(key) is not None]
        result[f"{prefix}_mean"] = _mean(values)
        result[f"{prefix}_median"] = _median(values)
    return result


def _finger_count(rows: Sequence[Mapping[str, str]], positions: set[int]) -> int:
    return sum(int(row["canonical_finger_position"]) in positions for row in rows)


def _render_supervisor_tables(
    decisions: Mapping[str, Mapping[str, Any]],
    included: Mapping[str, Sequence[Mapping[str, str]]],
    protocol_summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = ["# Derived SIFT plain-roll supervisor tables", ""]
    for dataset in DATASETS:
        label = "SD300b" if dataset == "sd300b" else "SD300c"
        rows = included[dataset]
        decision = decisions[dataset]
        protocol = protocol_summaries[dataset]
        fields = [
            ("Subjects", len({row["subject_id"] for row in rows})),
            ("Anatomical identities", len(rows)),
            ("Thumb", _finger_count(rows, {1, 6})),
            ("Index", _finger_count(rows, {2, 7})),
            ("Middle", _finger_count(rows, {3, 8})),
            ("Ring", _finger_count(rows, {4, 9})),
            ("Little", _finger_count(rows, {5, 10})),
            ("plain_self accepted identities", protocol["plain_self_accepted_count"]),
            ("roll_self accepted identities", protocol["roll_self_accepted_count"]),
            ("Final paired identities", len(rows)),
            ("plain_roll accepted", decision["accepted_count"]),
            ("plain_roll rejected", decision["rejected_count"]),
            ("Accepted percentage", decision["accepted_percentage"]),
            ("Failures", decision["failure_count"]),
            ("Mean raw score", decision["raw_score_mean"]),
            ("Median raw score", decision["raw_score_median"]),
            ("Mean method compare time (ms)", decision["method_compare_ms_mean"]),
            ("Median method compare time (ms)", decision["method_compare_ms_median"]),
            ("P95 method compare time (ms)", decision["method_compare_ms_p95"]),
            (
                "Keypoints A/B mean; median",
                f"{decision['keypoints_a_mean']} / {decision['keypoints_b_mean']}; "
                f"{decision['keypoints_a_median']} / {decision['keypoints_b_median']}",
            ),
            (
                "Candidate matches mean; median",
                f"{decision['candidate_matches_mean']}; {decision['candidate_matches_median']}",
            ),
            (
                "Geometric inliers mean; median",
                f"{decision['geometric_inliers_mean']}; {decision['geometric_inliers_median']}",
            ),
        ]
        lines.extend(
            [
                f"## {label} derived SIFT plain_roll",
                "",
                "| Measure | Value |",
                "| --- | ---: |",
            ]
        )
        for name, value in fields:
            lines.append(f"| {name} | {value} |")
        lines.append("")
    lines.extend(
        [
            "Every included PLAIN identity passed its dataset-specific frozen SIFT `plain_self` decision.",
            "Every included ROLL identity passed its dataset-specific frozen SIFT `roll_self` decision.",
            "Only identities with both sides and a valid base `plain_roll` pair were included.",
            "A `plain_roll` rejection remains an experimental result and never removes the pair.",
            "",
        ]
    )
    return "\n".join(lines)


def _find_single_result_bundle(root: Path) -> Path:
    bundles = sorted(
        metadata.parent
        for metadata in root.glob(f"*/{METADATA_FILENAME}")
        if (metadata.parent / RESULT_FILENAME).is_file()
    )
    if len(bundles) != 1:
        raise SiftDerivedProtocolError(f"Expected one result bundle below {root}, found {len(bundles)}.")
    return bundles[0]


def _alignment_report(
    project_root: Path,
    sift_protocol_root: Path,
    sift_rows: Mapping[str, Sequence[Mapping[str, str]]],
    thresholds: Mapping[str, float],
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    sourceafis_protocol = (
        project_root
        / "results"
        / "derived_protocols"
        / "sourceafis_per_dataset_self_accept_t40_v1"
    )
    sourceafis_runs = (
        project_root
        / "results"
        / "derived_protocol_runs"
        / "sourceafis_per_dataset_self_accept_t40_v1"
    )
    detail: list[dict[str, str]] = []
    summaries: dict[str, Any] = {}
    lines = ["# SourceAFIS / SIFT derived-protocol alignment", ""]
    for dataset in DATASETS:
        source_manifest = sourceafis_protocol / dataset / "plain_roll.csv"
        sift_manifest = sift_protocol_root / dataset / "plain_roll.csv"
        source_pairs = read_pair_manifest(source_manifest)
        sift_pairs = read_pair_manifest(sift_manifest)
        source_identities = {_identity(pair) for pair in source_pairs}
        sift_identities = {_identity(pair) for pair in sift_pairs}
        union = source_identities | sift_identities
        for subject, finger in sorted(union):
            detail.append(
                {
                    "dataset": dataset,
                    "subject_id": subject,
                    "canonical_finger_position": str(finger),
                    "sourceafis_included": str((subject, finger) in source_identities).lower(),
                    "sift_included": str((subject, finger) in sift_identities).lower(),
                    "membership": (
                        "both"
                        if (subject, finger) in source_identities and (subject, finger) in sift_identities
                        else "sourceafis_only"
                        if (subject, finger) in source_identities
                        else "sift_only"
                    ),
                }
            )
        source_bundle = _find_single_result_bundle(
            sourceafis_runs
            / dataset
            / "plain_roll"
            / "sourceafis"
            / BENCHMARK_CONTRACT_VERSION
        )
        source_rows = read_result_rows(source_bundle / RESULT_FILENAME)
        source_accept = sum(
            row["status"] == "ok" and float(row["raw_score"]) >= 40.0 for row in source_rows
        )
        sift_accept = sum(
            row["status"] == "ok" and float(row["raw_score"]) >= thresholds[dataset]
            for row in sift_rows[dataset]
        )
        summary = {
            "sourceafis_derived_pairs": len(source_pairs),
            "sift_derived_pairs": len(sift_pairs),
            "intersection_identities": len(source_identities & sift_identities),
            "sourceafis_only_identities": len(source_identities - sift_identities),
            "sift_only_identities": len(sift_identities - source_identities),
            "sourceafis_accepted": source_accept,
            "sourceafis_rejected": len(source_rows) - source_accept,
            "sift_accepted": sift_accept,
            "sift_rejected": len(sift_rows[dataset]) - sift_accept,
        }
        summaries[dataset] = summary
        label = "SD300b" if dataset == "sd300b" else "SD300c"
        lines.extend(
            [
                f"## {label}",
                "",
                "| Measure | Value |",
                "| --- | ---: |",
                f"| SourceAFIS derived pairs | {summary['sourceafis_derived_pairs']} |",
                f"| SIFT derived pairs | {summary['sift_derived_pairs']} |",
                f"| Identities in both | {summary['intersection_identities']} |",
                f"| SourceAFIS-only identities | {summary['sourceafis_only_identities']} |",
                f"| SIFT-only identities | {summary['sift_only_identities']} |",
                f"| SourceAFIS accepted / rejected in its cohort | {source_accept} / {len(source_rows) - source_accept} |",
                f"| SIFT accepted / rejected in its cohort | {sift_accept} / {len(sift_rows[dataset]) - sift_accept} |",
                "",
            ]
        )
    lines.extend(
        [
            "These percentages and counts do not establish that either method is better: each derived cohort was selected by that method's own self decisions.",
            "A direct scientific comparison should use the primary full manifests, followed by a future common-intersection analysis.",
            "No new common cross-method cohort was created here.",
            "",
        ]
    )
    return "\n".join(lines), detail, summaries


def build_reports(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Validate reruns, compare exactly to primary, and publish all reports atomically."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    protocol = load_and_validate_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=protocol_root,
        validate_sources=True,
    )
    frozen = validate_frozen_artifacts(project_root)
    reports_root = run_root / "reports"
    candidate = create_candidate_directory(reports_root)
    try:
        decisions: dict[str, dict[str, Any]] = {}
        comparisons: dict[str, dict[str, Any]] = {}
        rows_by_dataset: dict[str, list[dict[str, str]]] = {}
        included_by_dataset: dict[str, list[dict[str, str]]] = {}
        protocol_summaries = {item["dataset"]: item for item in protocol["summary"]["datasets"]}
        for dataset in DATASETS:
            primary = load_primary_bundle(
                project_root, data_root, dataset, "plain_roll", validate_source_manifest=False
            )
            derived = load_derived_bundle(
                run_root=run_root, protocol_root=protocol_root, dataset=dataset
            )
            details, comparison = compare_derived_to_primary(
                primary.rows,
                derived.rows,
                dataset=dataset,
                threshold=frozen["thresholds"][dataset],
            )
            comparison_rows: list[dict[str, str]] = []
            for row in details:
                flat = dict(row)
                flat["primary_diagnostics_json"] = json.dumps(
                    flat.pop("primary_diagnostics"), ensure_ascii=True, sort_keys=True, separators=(",", ":")
                )
                flat["derived_diagnostics_json"] = json.dumps(
                    flat.pop("derived_diagnostics"), ensure_ascii=True, sort_keys=True, separators=(",", ":")
                )
                comparison_rows.append(_stringify(flat))
            write_csv_atomic(
                comparison_rows,
                candidate / dataset / "primary_reproducibility.csv",
                list(comparison_rows[0]) if comparison_rows else [],
            )
            decisions[dataset] = decision_summary(
                derived.rows, dataset=dataset, threshold=frozen["thresholds"][dataset]
            )
            comparisons[dataset] = comparison
            rows_by_dataset[dataset] = derived.rows
            included_by_dataset[dataset] = _read_csv(
                protocol_root / dataset / "included_identities.csv", INCLUDED_COLUMNS
            )

        decision_rows = []
        for dataset in DATASETS:
            row = dict(decisions[dataset])
            row["failure_counts_by_stage_json"] = json.dumps(
                row.pop("failure_counts_by_stage"), sort_keys=True, separators=(",", ":")
            )
            decision_rows.append(_stringify(row))
        write_csv_atomic(decision_rows, candidate / "decision_summary.csv", list(decision_rows[0]))
        write_json_atomic(
            {
                "schema_version": "sift-derived-decision-summary-v1",
                "datasets": decisions,
            },
            candidate / "decision_summary.json",
        )
        comparison_payload = {
            "schema_version": "sift-derived-exact-reproducibility-v1",
            "score_tolerance": 0.0,
            "timing_equality_required": False,
            "datasets": comparisons,
            "overall": {
                "pair_count": sum(item["pair_count"] for item in comparisons.values()),
                "mismatch_pair_count": sum(
                    item["mismatch_pair_count"] for item in comparisons.values()
                ),
                "max_absolute_score_delta": max(
                    item["max_absolute_score_delta"] or 0.0 for item in comparisons.values()
                ),
                "passed": all(item["passed"] for item in comparisons.values()),
            },
        }
        write_json_atomic(comparison_payload, candidate / "reproducibility_summary.json")
        comparison_csv = [_stringify(item) for item in comparisons.values()]
        write_csv_atomic(
            comparison_csv, candidate / "reproducibility_summary.csv", list(comparison_csv[0])
        )
        (candidate / "supervisor_tables.md").write_text(
            _render_supervisor_tables(decisions, included_by_dataset, protocol_summaries),
            encoding="utf-8",
            newline="\n",
        )
        alignment_markdown, alignment_rows, alignment_summary = _alignment_report(
            project_root,
            protocol_root,
            rows_by_dataset,
            frozen["thresholds"],
        )
        (candidate / "sourceafis_sift_protocol_alignment.md").write_text(
            alignment_markdown, encoding="utf-8", newline="\n"
        )
        write_csv_atomic(
            alignment_rows,
            candidate / "sourceafis_sift_identity_alignment.csv",
            [
                "dataset",
                "subject_id",
                "canonical_finger_position",
                "sourceafis_included",
                "sift_included",
                "membership",
            ],
        )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "namespace": PROTOCOL_NAMESPACE,
            "protocol_summary_sha256": protocol["protocol_summary_sha256"],
            "decision_summaries": decisions,
            "reproducibility": comparison_payload,
            "sourceafis_sift_alignment": alignment_summary,
            "primary_artifacts_overwritten": False,
        }
        write_json_atomic(report, candidate / "report_summary.json")
        hashes = _directory_hashes(candidate)
        write_json_atomic(
            {"schema_version": "sift-derived-report-hashes-v1", "files": hashes},
            candidate / "report_artifact_hashes.json",
        )
        _publish_immutable_candidate(candidate, reports_root)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)
    final = _read_json(reports_root / "report_summary.json")
    if final["reproducibility"]["overall"]["passed"] is not True:
        raise SiftDerivedProtocolError("Exact SIFT primary/rerun reproducibility failed.")
    return {
        **final,
        "reports_root": str(reports_root),
        "report_summary_sha256": file_sha256(reports_root / "report_summary.json"),
    }


def protected_artifact_paths(project_root: Path, data_root: Path) -> dict[Path, str]:
    """Enumerate every immutable dataset, base, method, and implementation file."""

    paths: dict[Path, str] = {}

    def add_tree(root: Path, category: str) -> None:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    paths[path] = category

    # This study is intentionally limited to the two NIST datasets used by
    # every manifest in scope.  Other collections below the shared data root
    # are unrelated and must not enter this protocol's integrity inventory.
    add_tree(data_root / "NIST" / "sd300b", "dataset_sd300b")
    add_tree(data_root / "NIST" / "sd300c", "dataset_sd300c")
    add_tree(project_root / "protocols", "base_manifests")
    add_tree(project_root / "results" / "sift_geometric", "sift_pilot_config_decision_and_previous_results")
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            add_tree(
                project_root / "results" / dataset / protocol / METHOD_NAME,
                "sift_primary_bundles",
            )
    results = project_root / "results"
    if results.exists():
        for path in results.rglob("*"):
            if (
                path.is_file()
                and PROTOCOL_NAMESPACE not in path.as_posix()
                and "sourceafis" in path.as_posix().lower()
            ):
                paths[path] = "sourceafis_artifacts_audits_and_cohorts"
    add_tree(project_root / "apps" / "sourceafis-sidecar", "sourceafis_implementation")
    add_tree(project_root / "src" / "fingerprint_benchmark" / "sift", "sift_implementation")
    support = (
        "runner.py",
        "contract.py",
        "bundle.py",
        "hashing.py",
        "io.py",
        "manifest.py",
        "preflight.py",
        "provenance.py",
    )
    for name in support:
        path = project_root / "src" / "fingerprint_benchmark" / name
        if path.is_file():
            paths[path.resolve()] = "benchmark_implementation"
    for root, pattern, category in (
        (project_root / "src" / "fingerprint_benchmark", "sourceafis*.py", "sourceafis_implementation"),
        (project_root / "tests", "test_sift*.py", "sift_tests"),
        (project_root / "tests", "test_sourceafis*.py", "sourceafis_tests"),
    ):
        if root.exists():
            for path in root.glob(pattern):
                paths[path] = category
    return dict(sorted(paths.items(), key=lambda item: str(item[0]).lower()))


def _snapshot_records(
    paths: Mapping[Path, str], *, progress_label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tree = hashlib.sha256()
    total_bytes = 0
    count = len(paths)
    for index, (path, category) in enumerate(paths.items(), start=1):
        stat = path.stat()
        digest = file_sha256(path)
        record = {
            "category": category,
            "path": path.as_posix(),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest,
        }
        records.append(record)
        tree.update(canonical_json_bytes({key: record[key] for key in ("category", "path", "size", "sha256")}))
        tree.update(b"\n")
        total_bytes += stat.st_size
        if index == 1 or index % 1000 == 0 or index == count:
            print(
                f"[{progress_label}] hashed {index}/{count} files ({total_bytes / (1024 ** 3):.2f} GiB)",
                file=sys.stderr,
                flush=True,
            )
    footer = {
        "file_count": len(records),
        "total_bytes": total_bytes,
        "tree_sha256": tree.hexdigest(),
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
    }
    return records, footer


def _snapshot_bytes(records: Sequence[Mapping[str, Any]], footer: Mapping[str, Any]) -> bytes:
    lines = [
        json.dumps(
            {
                "header": {
                    "schema_version": "sift-derived-protected-artifact-snapshot-v1",
                    "algorithm": "sha256",
                }
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    lines.extend(
        json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    lines.append(
        json.dumps({"footer": dict(footer)}, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _read_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                payload = json.loads(line)
                if index == 0:
                    if payload.get("header", {}).get("schema_version") != "sift-derived-protected-artifact-snapshot-v1":
                        raise SiftDerivedProtocolError(f"Wrong snapshot schema: {path}")
                elif "footer" in payload:
                    footer = payload["footer"]
                else:
                    records.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise SiftDerivedProtocolError(f"Cannot read protected snapshot {path}: {exc}") from exc
    if footer is None or footer.get("file_count") != len(records):
        raise SiftDerivedProtocolError(f"Protected snapshot footer is missing or inconsistent: {path}")
    return records, footer


def capture_protected_before(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    run_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    paths = protected_artifact_paths(project_root, data_root)
    records, footer = _snapshot_records(paths, progress_label="protected-before")
    output = run_root / "integrity" / "protected_before.jsonl"
    _publish_immutable_bytes(output, _snapshot_bytes(records, footer))
    return {**footer, "path": str(output), "sha256": file_sha256(output)}


def _artifact_manifest(protocol_root: Path, run_root: Path) -> dict[str, Any]:
    output_path = run_root / "artifact_manifest.json"
    files: dict[str, dict[str, Any]] = {}
    for label, root in (("derived_protocols", protocol_root), ("derived_runs", run_root)):
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.resolve() == output_path.resolve():
                continue
            relative = path.relative_to(root).as_posix()
            files[f"{label}/{relative}"] = {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return {
        "schema_version": "sift-derived-artifact-manifest-v1",
        "namespace": PROTOCOL_NAMESPACE,
        "files": files,
    }


def finalize_protected_after(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    before_path = run_root / "integrity" / "protected_before.jsonl"
    before_records, before_footer = _read_snapshot(before_path)
    current_paths = protected_artifact_paths(project_root, data_root)
    before_paths = {Path(record["path"]) for record in before_records}
    if set(current_paths) != before_paths:
        added = sorted(str(path) for path in set(current_paths) - before_paths)
        removed = sorted(str(path) for path in before_paths - set(current_paths))
        raise SiftDerivedProtocolError(
            f"Protected artifact path set changed; added={added[:5]}, removed={removed[:5]}."
        )
    after_records, after_footer = _snapshot_records(current_paths, progress_label="protected-after")
    before_by_path = {record["path"]: record for record in before_records}
    mismatches = []
    for record in after_records:
        before = before_by_path[record["path"]]
        if before["sha256"] != record["sha256"] or before["size"] != record["size"]:
            mismatches.append(
                {
                    "path": record["path"],
                    "before_sha256": before["sha256"],
                    "after_sha256": record["sha256"],
                    "before_size": before["size"],
                    "after_size": record["size"],
                }
            )
    after_path = run_root / "integrity" / "protected_after.jsonl"
    _publish_immutable_bytes(after_path, _snapshot_bytes(after_records, after_footer))
    passed = not mismatches and before_footer["tree_sha256"] == after_footer["tree_sha256"]
    integrity = {
        "schema_version": "sift-derived-protected-artifact-integrity-v1",
        "protected_artifacts_unchanged": passed,
        "before_snapshot_sha256": file_sha256(before_path),
        "after_snapshot_sha256": file_sha256(after_path),
        "before": before_footer,
        "after": after_footer,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    report_path = run_root / "integrity" / "protected_artifact_integrity.json"
    _publish_immutable_bytes(report_path, _json_bytes(integrity))
    artifact_manifest = _artifact_manifest(protocol_root, run_root)
    artifact_manifest["protected_artifacts_unchanged"] = passed
    artifact_manifest["protected_tree_sha256"] = before_footer["tree_sha256"]
    artifact_path = run_root / "artifact_manifest.json"
    _publish_immutable_bytes(artifact_path, _json_bytes(artifact_manifest))
    result = {
        **integrity,
        "integrity_report_path": str(report_path),
        "integrity_report_sha256": file_sha256(report_path),
        "artifact_manifest_path": str(artifact_path),
        "artifact_manifest_sha256": file_sha256(artifact_path),
        "artifact_count": len(artifact_manifest["files"]),
    }
    if not passed:
        raise SiftDerivedProtocolError(f"Protected artifacts changed: {mismatches[:3]}")
    return result


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--run-root", type=Path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIFT per-dataset self-accept derived protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("protect-before", "prepare", "preflight", "run", "report", "protect-after"):
        child = subparsers.add_parser(command)
        _add_paths(child)
        if command == "run":
            child.add_argument("--skip-existing", action="store_true")
            child.add_argument(
                "--execution-results-root",
                type=Path,
                help="Same physical --run-root using a Windows extended-length path or short alias; avoids MAX_PATH in atomic candidates.",
            )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        common = {
            "project_root": args.project_root,
            "data_root": args.data_root,
        }
        if args.command == "protect-before":
            result = capture_protected_before(
                **common,
                run_root=args.run_root,
            )
        elif args.command == "prepare":
            result = prepare_protocol(
                **common,
                protocol_root=args.protocol_root,
            )
        elif args.command == "preflight":
            result = runtime_preflight(
                **common,
                protocol_root=args.protocol_root,
                run_root=args.run_root,
            )
        elif args.command == "run":
            result = run_protocol(
                **common,
                protocol_root=args.protocol_root,
                run_root=args.run_root,
                execution_results_root=args.execution_results_root,
                skip_existing=args.skip_existing,
            )
        elif args.command == "report":
            result = build_reports(
                **common,
                protocol_root=args.protocol_root,
                run_root=args.run_root,
            )
        elif args.command == "protect-after":
            result = finalize_protected_after(
                **common,
                protocol_root=args.protocol_root,
                run_root=args.run_root,
            )
        else:
            raise SiftDerivedProtocolError(f"Unsupported command: {args.command}")
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
