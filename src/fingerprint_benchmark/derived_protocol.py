"""Per-dataset self-accept-then-plain-roll protocol for SourceAFIS.

This module deliberately lives beside, rather than inside, the primary
benchmark CLI.  The primary manifests and benchmark bundles are immutable
inputs.  Derived manifests and reruns are published under method-specific
namespaces and validated as exact ordered subsets of the base plain-roll
manifests.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from .bundle import (
    BundlePublicationError,
    create_candidate_directory,
    discard_candidate_directory,
    publish_candidate_directory,
)
from .contract import BENCHMARK_CONTRACT_VERSION, BenchmarkRunSpec
from .hashing import file_sha256
from .io import write_csv_atomic, write_json_atomic
from .manifest import MANIFEST_COLUMNS, PairRecord, read_pair_manifest
from .preflight import validator_for
from .runner import (
    METADATA_FILENAME,
    RESULT_FILENAME,
    prepare_run_context,
    read_result_rows,
    run_benchmark_manifest,
    validate_result_bundle,
)
from .sourceafis_adapter import SourceAfisAdapter
from .sourceafis_client import SourceAfisSidecarClient, validate_health
from .sourceafis_sidecar import ManagedSourceAfisSidecar, SidecarStartup


PROTOCOL_NAMESPACE = "sourceafis_per_dataset_self_accept_t40_v1"
PROTOCOL_SCHEMA_VERSION = "per-dataset-self-accept-derived-protocol-v1"
REPORT_SCHEMA_VERSION = "derived-sourceafis-report-v1"
SOURCEAFIS_THRESHOLD = 40.0
DATASETS = ("sd300b", "sd300c")
SELF_PROTOCOLS = ("plain_self", "roll_self")
PROTOCOLS = ("plain_self", "roll_self", "plain_roll")
METHOD = "sourceafis"

DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
DEFAULT_PROTOCOL_ROOT = (
    Path("results") / "derived_protocols" / PROTOCOL_NAMESPACE
)
DEFAULT_RUN_ROOT = (
    Path("results") / "derived_protocol_runs" / PROTOCOL_NAMESPACE
)
DEFAULT_SIDECAR_JAR = (
    Path("apps") / "sourceafis-sidecar" / "target" / "sourceafis-sidecar-0.2.0.jar"
)
DEFAULT_SERVICE_URL = "http://127.0.0.1:8765"
DEFAULT_THRESHOLD_AUDIT = (
    Path("results")
    / "sourceafis"
    / BENCHMARK_CONTRACT_VERSION
    / "threshold40_audit"
    / "threshold40_audit.json"
)

# Informational cross-checks from the agreed protocol.  These values never
# participate in selection or validation pass/fail.
ADVISORY_EXPECTED_INCLUDED_COUNTS = {"sd300b": 8593, "sd300c": 8614}

INCLUDED_COLUMNS = [
    "dataset",
    "subject_id",
    "canonical_finger_position",
    "plain_self_status",
    "plain_self_raw_score",
    "roll_self_status",
    "roll_self_raw_score",
    "base_plain_roll_pair_id",
]

EXCLUDED_COLUMNS = [
    *INCLUDED_COLUMNS,
    "reason_flags",
]

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
    "advisory_expected_included_count",
    "advisory_count_delta",
    "derived_manifest_sha256",
]

DECISION_SUMMARY_COLUMNS = [
    "dataset",
    "threshold",
    "total_pairs",
    "status_ok",
    "failures",
    "genuine_accepts",
    "false_non_matches",
    "score_zero",
    "positive_below_threshold",
    "accept_percentage",
    "reject_percentage",
    "mean_score",
    "median_score",
    "mean_method_compare_ms",
    "median_method_compare_ms",
    "p95_method_compare_ms",
    "mean_total_ms",
    "median_total_ms",
    "p95_total_ms",
]

REPRODUCIBILITY_COLUMNS = [
    "dataset",
    "pair_id",
    "subject_id",
    "canonical_finger_position",
    "primary_status",
    "derived_status",
    "status_equal",
    "primary_error_code",
    "derived_error_code",
    "error_code_equal",
    "primary_raw_score",
    "derived_raw_score",
    "raw_score_text_equal",
    "raw_score_abs_delta",
    "prepare_a_diagnostics_equal",
    "prepare_b_diagnostics_equal",
    "compare_diagnostics_equal",
    "reproducible",
]

Identity = tuple[str, int]


class DerivedProtocolError(ValueError):
    """Raised when the derived protocol cannot be proven safe."""


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
    included_identities: tuple[Identity, ...]
    included_rows: list[dict[str, str]]
    excluded_rows: list[dict[str, str]]
    reason_counts: dict[str, int]
    plain_self_accepted_count: int
    roll_self_accepted_count: int
    identity_universe_count: int


def identity_from_pair(pair: PairRecord) -> Identity:
    return pair.subject_id, pair.canonical_finger_position


def identity_sort_key(identity: Identity) -> tuple[str, int]:
    return identity


def load_primary_bundle(
    project_root: Path,
    data_root: Path,
    dataset: str,
    protocol: str,
    *,
    validate_source_manifest: bool = True,
) -> PrimaryBundle:
    """Load and fully validate one immutable primary SourceAFIS bundle."""

    if dataset not in DATASETS or protocol not in PROTOCOLS:
        raise DerivedProtocolError(f"Unsupported primary condition: {dataset}/{protocol}")
    contract_root = (
        project_root
        / "results"
        / dataset
        / protocol
        / METHOD
        / BENCHMARK_CONTRACT_VERSION
    )
    bundles = sorted(
        metadata.parent
        for metadata in contract_root.glob(f"*/{METADATA_FILENAME}")
        if (metadata.parent / RESULT_FILENAME).is_file()
    )
    if len(bundles) != 1:
        raise DerivedProtocolError(
            f"Expected exactly one primary bundle for {dataset}/{protocol}, found {len(bundles)}."
        )
    bundle_path = bundles[0].resolve()
    metadata_path = bundle_path / METADATA_FILENAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivedProtocolError(f"Cannot read primary metadata {metadata_path}: {exc}") from exc
    raw_spec = dict(metadata.get("run_spec", {}))
    if not raw_spec:
        raise DerivedProtocolError(f"Primary metadata lacks run_spec: {metadata_path}")
    raw_spec["manifest_path"] = Path(raw_spec["manifest_path"])
    spec = BenchmarkRunSpec(**raw_spec)
    expected_manifest = (project_root / "protocols" / dataset / f"{protocol}.csv").resolve()
    if spec.manifest_path.resolve() != expected_manifest:
        raise DerivedProtocolError(
            f"Primary run points to unexpected manifest for {dataset}/{protocol}: {spec.manifest_path}"
        )
    if file_sha256(expected_manifest) != spec.manifest_sha256:
        raise DerivedProtocolError(
            f"Primary manifest SHA-256 does not match provenance for {dataset}/{protocol}."
        )
    if validate_source_manifest:
        validator_for(dataset, protocol)(expected_manifest, data_root)
    pairs = read_pair_manifest(expected_manifest)
    validate_result_bundle(
        bundle_path,
        manifest_records=pairs,
        run_spec=spec,
        score_direction=metadata["score_direction"],
        score_semantics=metadata["score_semantics"],
    )
    rows = read_result_rows(bundle_path / RESULT_FILENAME)
    if [pair.pair_id for pair in pairs] != [row["pair_id"] for row in rows]:
        raise DerivedProtocolError(
            f"Primary result sequence is not aligned with its manifest for {dataset}/{protocol}."
        )
    return PrimaryBundle(
        dataset=dataset,
        protocol=protocol,
        manifest_path=expected_manifest,
        bundle_path=bundle_path,
        pairs=pairs,
        rows=rows,
        metadata=metadata,
    )


def validate_threshold_audit(project_root: Path, audit_path: Path) -> dict[str, Any]:
    """Validate the existing frozen threshold-40 audit and every recorded input hash."""

    audit_path = audit_path.resolve()
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivedProtocolError(f"Cannot read threshold audit {audit_path}: {exc}") from exc
    if audit.get("audit_schema_version") != "sourceafis-threshold-decision-audit-v1":
        raise DerivedProtocolError("Unexpected threshold audit schema.")
    if float(audit.get("sourceafis_threshold", math.nan)) != SOURCEAFIS_THRESHOLD:
        raise DerivedProtocolError("Existing threshold audit does not freeze threshold 40.0.")
    primary_validation = audit.get("primary_bundle_validation", {})
    required_primary_checks = (
        "all_six_source_bundles_validated_before_calculation_by_audit_validator",
        "all_six_result_schemas_and_metadata_valid",
        "all_six_results_match_manifests_by_ordered_pair_id_and_identity",
        "all_six_result_and_manifest_hashes_match_metadata",
        "all_six_score_payload_hashes_match_metadata",
    )
    if not all(primary_validation.get(key) is True for key in required_primary_checks):
        raise DerivedProtocolError("Existing threshold audit did not pass all primary-bundle checks.")
    if audit.get("validation", {}).get("protected_source_artifacts_unchanged_during_audit") is not True:
        raise DerivedProtocolError("Existing threshold audit lacks protected-artifact integrity proof.")
    input_hashes = audit.get("input_artifacts_sha256")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise DerivedProtocolError("Existing threshold audit lacks input SHA-256 provenance.")
    for relative, expected_hash in sorted(input_hashes.items()):
        path = (project_root / relative).resolve()
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise DerivedProtocolError(f"Threshold-audit input hash mismatch: {path}")
    return {
        "path": str(audit_path),
        "sha256": file_sha256(audit_path),
        "schema_version": audit["audit_schema_version"],
        "threshold": SOURCEAFIS_THRESHOLD,
        "verified_input_hash_count": len(input_hashes),
    }


def _result_rows_by_identity(bundle: PrimaryBundle) -> dict[Identity, dict[str, str]]:
    indexed: dict[Identity, dict[str, str]] = {}
    for pair, row in zip(bundle.pairs, bundle.rows, strict=True):
        identity = identity_from_pair(pair)
        if identity in indexed:
            raise DerivedProtocolError(
                f"Duplicate identity in {bundle.dataset}/{bundle.protocol}: {identity}"
            )
        if row["subject_id"] != pair.subject_id or int(row["canonical_finger_position"]) != identity[1]:
            raise DerivedProtocolError(
                f"Result identity mismatch in {bundle.dataset}/{bundle.protocol}: {pair.pair_id}"
            )
        indexed[identity] = row
    return indexed


def _pairs_by_identity(pairs: Sequence[PairRecord], label: str) -> dict[Identity, PairRecord]:
    indexed: dict[Identity, PairRecord] = {}
    for pair in pairs:
        identity = identity_from_pair(pair)
        if identity in indexed:
            raise DerivedProtocolError(f"Duplicate identity in {label}: {identity}")
        indexed[identity] = pair
    return indexed


def _score(row: Mapping[str, str], label: str) -> float:
    raw = row.get("raw_score", "")
    if raw == "":
        raise DerivedProtocolError(f"Successful row lacks raw_score in {label}.")
    value = float(raw)
    if not math.isfinite(value):
        raise DerivedProtocolError(f"Non-finite raw_score in {label}.")
    return value


def self_row_is_accepted(row: Mapping[str, str], threshold: float = SOURCEAFIS_THRESHOLD) -> bool:
    """Return the frozen self-test decision without consulting plain-roll scores."""

    return row.get("status") == "ok" and _score(row, str(row.get("pair_id", "self row"))) >= threshold


def select_dataset_identities(
    dataset: str,
    plain_self: PrimaryBundle,
    roll_self: PrimaryBundle,
    base_plain_roll: PrimaryBundle,
    *,
    threshold: float = SOURCEAFIS_THRESHOLD,
) -> DatasetSelection:
    """Select one dataset independently using only its two self results and pair availability."""

    if dataset not in DATASETS:
        raise DerivedProtocolError(f"Wrong dataset: {dataset}")
    for bundle, protocol in (
        (plain_self, "plain_self"),
        (roll_self, "roll_self"),
        (base_plain_roll, "plain_roll"),
    ):
        if bundle.dataset != dataset or bundle.protocol != protocol:
            raise DerivedProtocolError(
                f"Wrong bundle supplied for {dataset}/{protocol}: {bundle.dataset}/{bundle.protocol}"
            )

    plain_rows = _result_rows_by_identity(plain_self)
    roll_rows = _result_rows_by_identity(roll_self)
    plain_roll_pairs = _pairs_by_identity(base_plain_roll.pairs, f"{dataset}/plain_roll")
    universe = set(plain_rows) | set(roll_rows) | set(plain_roll_pairs)
    included_set: set[Identity] = set()
    excluded_rows: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()

    for identity in sorted(universe, key=identity_sort_key):
        plain = plain_rows.get(identity)
        roll = roll_rows.get(identity)
        pair = plain_roll_pairs.get(identity)
        reasons: list[str] = []
        if plain is None:
            reasons.append("missing_plain_self_identity")
        elif plain["status"] != "ok":
            reasons.append("plain_self_non_ok")
        elif _score(plain, f"{dataset}/plain_self {identity}") < threshold:
            reasons.append("plain_self_below_40")
        if roll is None:
            reasons.append("missing_roll_self_identity")
        elif roll["status"] != "ok":
            reasons.append("roll_self_non_ok")
        elif _score(roll, f"{dataset}/roll_self {identity}") < threshold:
            reasons.append("roll_self_below_40")
        if pair is None:
            reasons.append("missing_plain_roll_pair")

        common = {
            "dataset": dataset,
            "subject_id": identity[0],
            "canonical_finger_position": str(identity[1]),
            "plain_self_status": plain["status"] if plain is not None else "",
            "plain_self_raw_score": plain["raw_score"] if plain is not None else "",
            "roll_self_status": roll["status"] if roll is not None else "",
            "roll_self_raw_score": roll["raw_score"] if roll is not None else "",
            "base_plain_roll_pair_id": pair.pair_id if pair is not None else "",
        }
        if reasons:
            reason_counts.update(reasons)
            excluded_rows.append({**common, "reason_flags": ";".join(reasons)})
        else:
            included_set.add(identity)

    included_identities = tuple(
        identity_from_pair(pair)
        for pair in base_plain_roll.pairs
        if identity_from_pair(pair) in included_set
    )
    if set(included_identities) != included_set:
        raise DerivedProtocolError(f"Derived completeness failure for {dataset}.")
    included_rows = []
    for identity in included_identities:
        pair = plain_roll_pairs[identity]
        included_rows.append(
            {
                "dataset": dataset,
                "subject_id": identity[0],
                "canonical_finger_position": str(identity[1]),
                "plain_self_status": plain_rows[identity]["status"],
                "plain_self_raw_score": plain_rows[identity]["raw_score"],
                "roll_self_status": roll_rows[identity]["status"],
                "roll_self_raw_score": roll_rows[identity]["raw_score"],
                "base_plain_roll_pair_id": pair.pair_id,
            }
        )
    return DatasetSelection(
        dataset=dataset,
        included_identities=included_identities,
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        reason_counts=dict(sorted(reason_counts.items())),
        plain_self_accepted_count=sum(self_row_is_accepted(row, threshold) for row in plain_rows.values()),
        roll_self_accepted_count=sum(self_row_is_accepted(row, threshold) for row in roll_rows.values()),
        identity_universe_count=len(universe),
    )


def _physical_manifest_rows(path: Path) -> tuple[bytes, list[tuple[str, bytes, list[str]]]]:
    """Return physical CSV rows while preserving every original row byte."""

    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if not lines:
        raise DerivedProtocolError(f"Manifest is empty: {path}")

    def parse_line(raw_line: bytes, line_number: int) -> list[str]:
        try:
            decoded = raw_line.decode("utf-8")
            parsed = list(csv.reader(io.StringIO(decoded, newline="")))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise DerivedProtocolError(f"Cannot parse physical CSV line {line_number} in {path}: {exc}") from exc
        if len(parsed) != 1:
            raise DerivedProtocolError(
                f"Manifest uses multiline CSV records, which cannot be filtered byte-exactly: {path}"
            )
        return parsed[0]

    header = parse_line(lines[0], 1)
    if header != MANIFEST_COLUMNS:
        raise DerivedProtocolError(f"Manifest schema mismatch in {path}: {header}")
    rows: list[tuple[str, bytes, list[str]]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(lines[1:], start=2):
        values = parse_line(raw_line, line_number)
        if len(values) != len(MANIFEST_COLUMNS):
            raise DerivedProtocolError(f"Manifest field count mismatch at {path}:{line_number}")
        pair_id = values[0]
        if pair_id in seen:
            raise DerivedProtocolError(f"Duplicate pair_id {pair_id!r} in {path}")
        seen.add(pair_id)
        rows.append((pair_id, raw_line, values))
    if len(rows) != len(read_pair_manifest(path)):
        raise DerivedProtocolError(f"Physical/logical CSV row mismatch in {path}")
    return lines[0], rows


def filter_manifest_bytes(base_manifest: Path, included_pair_ids: set[str]) -> bytes:
    """Filter rows while retaining the exact header, row bytes, and source order."""

    header, rows = _physical_manifest_rows(base_manifest)
    known = {pair_id for pair_id, _, _ in rows}
    missing = included_pair_ids - known
    if missing:
        raise DerivedProtocolError(
            f"Selected pair IDs are absent from the base manifest: {sorted(missing)[:5]}"
        )
    selected = [raw for pair_id, raw, _ in rows if pair_id in included_pair_ids]
    if len(selected) != len(included_pair_ids):
        raise DerivedProtocolError("Derived manifest selection count mismatch.")
    return header + b"".join(selected)


def validate_exact_manifest_subset(
    derived_manifest: Path,
    base_manifest: Path,
    *,
    expected_dataset: str,
    expected_pair_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prove exact schema, row bytes, source order, identity uniqueness, and completeness."""

    base_header, base_rows = _physical_manifest_rows(base_manifest)
    derived_header, derived_rows = _physical_manifest_rows(derived_manifest)
    if derived_header != base_header:
        raise DerivedProtocolError("Derived manifest header bytes differ from the base manifest.")
    base_lookup = {pair_id: (index, raw, values) for index, (pair_id, raw, values) in enumerate(base_rows)}
    actual_pair_ids: list[str] = []
    source_indexes: list[int] = []
    identities: set[Identity] = set()
    for pair_id, raw, values in derived_rows:
        source = base_lookup.get(pair_id)
        if source is None or source[1] != raw:
            raise DerivedProtocolError(f"Derived row is not byte-exact source row: {pair_id}")
        source_indexes.append(source[0])
        actual_pair_ids.append(pair_id)
        dataset = values[1]
        protocol = values[2]
        if dataset != expected_dataset or protocol != "plain_roll":
            raise DerivedProtocolError(f"Wrong dataset/protocol in derived row {pair_id}.")
        identity = (values[3], int(values[4]))
        if identity in identities:
            raise DerivedProtocolError(f"Duplicate identity in derived manifest: {identity}")
        identities.add(identity)
    if source_indexes != sorted(source_indexes):
        raise DerivedProtocolError("Derived manifest does not preserve base row order.")
    if expected_pair_ids is not None and actual_pair_ids != list(expected_pair_ids):
        raise DerivedProtocolError("Derived manifest is incomplete or contains an unexpected pair.")
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
    }


def make_derived_manifest_validator(
    *,
    base_manifest: Path,
    expected_dataset: str,
    expected_manifest_sha256: str,
    expected_pair_ids: Sequence[str],
    source_validator: Callable[[Path, Path], Any] | None = None,
) -> Callable[[Path, Path], dict[str, Any]]:
    """Create a runner validator bound to a frozen derived manifest and source manifest."""

    def validate(derived_manifest: Path, data_root: Path) -> dict[str, Any]:
        if file_sha256(derived_manifest) != expected_manifest_sha256:
            raise DerivedProtocolError("Derived manifest SHA-256 changed after publication.")
        validator = source_validator or validator_for(expected_dataset, "plain_roll")
        source_report = validator(base_manifest, data_root)
        report = validate_exact_manifest_subset(
            derived_manifest,
            base_manifest,
            expected_dataset=expected_dataset,
            expected_pair_ids=expected_pair_ids,
        )
        pairs = read_pair_manifest(derived_manifest)
        expected_ppi = 1000 if expected_dataset == "sd300b" else 2000
        for pair in pairs:
            if pair.ppi != expected_ppi:
                raise DerivedProtocolError(f"Wrong PPI in derived pair {pair.pair_id}.")
            if "plain" not in pair.path_a.name.lower() or "roll" not in pair.path_b.name.lower():
                raise DerivedProtocolError(f"Derived pair sides are not A=plain/B=roll: {pair.pair_id}")
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
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _directory_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted((item for item in root.rglob("*") if item.is_file()))
    }


def _publish_immutable_candidate(candidate: Path, final: Path) -> None:
    if final.exists():
        if _directory_file_hashes(candidate) != _directory_file_hashes(final):
            raise BundlePublicationError(
                f"Immutable derived protocol already exists with different content: {final}"
            )
        discard_candidate_directory(candidate)
        return
    publish_candidate_directory(candidate, final)


def prepare_protocol(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    sidecar_jar: Path | None = None,
    threshold_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all source inputs, derive both datasets, and publish atomically."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    final_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    jar_path = (sidecar_jar or project_root / DEFAULT_SIDECAR_JAR).resolve()
    audit_path = (threshold_audit_path or project_root / DEFAULT_THRESHOLD_AUDIT).resolve()
    if not jar_path.is_file():
        raise DerivedProtocolError(f"SourceAFIS sidecar JAR is missing: {jar_path}")
    threshold_audit = validate_threshold_audit(project_root, audit_path)

    bundles: dict[tuple[str, str], PrimaryBundle] = {}
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            bundles[(dataset, protocol)] = load_primary_bundle(
                project_root,
                data_root,
                dataset,
                protocol,
                validate_source_manifest=True,
            )

    candidate = create_candidate_directory(final_root)
    try:
        dataset_summaries: list[dict[str, Any]] = []
        all_source_hashes: dict[str, str] = {}
        for dataset in DATASETS:
            plain = bundles[(dataset, "plain_self")]
            roll = bundles[(dataset, "roll_self")]
            base = bundles[(dataset, "plain_roll")]
            selection = select_dataset_identities(dataset, plain, roll, base)
            included_pair_ids = [row["base_plain_roll_pair_id"] for row in selection.included_rows]
            manifest_payload = filter_manifest_bytes(base.manifest_path, set(included_pair_ids))
            dataset_dir = candidate / dataset
            derived_manifest = dataset_dir / "plain_roll.csv"
            _write_bytes(derived_manifest, manifest_payload)
            write_csv_atomic(selection.included_rows, dataset_dir / "included_identities.csv", INCLUDED_COLUMNS)
            write_csv_atomic(selection.excluded_rows, dataset_dir / "excluded_identities.csv", EXCLUDED_COLUMNS)
            validation = validate_exact_manifest_subset(
                derived_manifest,
                base.manifest_path,
                expected_dataset=dataset,
                expected_pair_ids=included_pair_ids,
            )
            validation["derived_manifest_path"] = str(
                (final_root / dataset / "plain_roll.csv").resolve()
            )
            expected = ADVISORY_EXPECTED_INCLUDED_COUNTS[dataset]
            source_hashes = {
                "plain_self_manifest_sha256": file_sha256(plain.manifest_path),
                "plain_self_pairs_sha256": file_sha256(plain.bundle_path / RESULT_FILENAME),
                "plain_self_metadata_sha256": file_sha256(plain.bundle_path / METADATA_FILENAME),
                "roll_self_manifest_sha256": file_sha256(roll.manifest_path),
                "roll_self_pairs_sha256": file_sha256(roll.bundle_path / RESULT_FILENAME),
                "roll_self_metadata_sha256": file_sha256(roll.bundle_path / METADATA_FILENAME),
                "base_plain_roll_manifest_sha256": file_sha256(base.manifest_path),
                "primary_plain_roll_pairs_sha256": file_sha256(base.bundle_path / RESULT_FILENAME),
                "primary_plain_roll_metadata_sha256": file_sha256(base.bundle_path / METADATA_FILENAME),
            }
            for label, digest in source_hashes.items():
                all_source_hashes[f"{dataset}/{label}"] = digest
            summary = {
                "dataset": dataset,
                "threshold": SOURCEAFIS_THRESHOLD,
                "identity_key": ["subject_id", "canonical_finger_position"],
                "selection_rule": "status == ok and plain_self >= 40 and roll_self >= 40 and base plain_roll pair exists",
                "plain_roll_score_used_for_selection": False,
                "plain_self_total_count": len(plain.rows),
                "plain_self_accepted_count": selection.plain_self_accepted_count,
                "roll_self_total_count": len(roll.rows),
                "roll_self_accepted_count": selection.roll_self_accepted_count,
                "base_plain_roll_count": len(base.pairs),
                "identity_universe_count": selection.identity_universe_count,
                "included_identity_count": len(selection.included_identities),
                "excluded_identity_count": len(selection.excluded_rows),
                "exclusion_reason_counts": selection.reason_counts,
                "advisory_expected_included_count": expected,
                "advisory_count_delta": len(selection.included_identities) - expected,
                "derived_manifest_relative_path": f"{dataset}/plain_roll.csv",
                "derived_manifest_sha256": validation["derived_manifest_sha256"],
                "included_identities_sha256": file_sha256(dataset_dir / "included_identities.csv"),
                "excluded_identities_sha256": file_sha256(dataset_dir / "excluded_identities.csv"),
                "validation": validation,
                "source_hashes": source_hashes,
                "primary_config_hash": base.metadata["config_hash"],
                "primary_implementation_hash": base.metadata["implementation_hash"],
                "primary_sidecar_jar_sha256": base.metadata["implementation_hash_components"]["sidecar_jar_sha256"],
            }
            dataset_summaries.append(summary)

        summary_payload = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "namespace": PROTOCOL_NAMESPACE,
            "method": METHOD,
            "method_version": "3.18.1",
            "threshold": SOURCEAFIS_THRESHOLD,
            "datasets_filtered_independently": True,
            "cross_dataset_identity_equality_required": False,
            "plain_roll_score_used_for_selection": False,
            "base_rows_preserved_byte_exactly": True,
            "base_row_order_preserved": True,
            "deterministic_output": True,
            "sourceafis_sidecar_jar_path": str(jar_path),
            "sourceafis_sidecar_jar_sha256": file_sha256(jar_path),
            "threshold_audit": threshold_audit,
            "source_artifacts_sha256": dict(sorted(all_source_hashes.items())),
            "datasets": dataset_summaries,
        }
        write_json_atomic(summary_payload, candidate / "protocol_summary.json")
        write_csv_atomic(
            [_stringify_row({key: summary[key] for key in PROTOCOL_SUMMARY_COLUMNS}) for summary in dataset_summaries],
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
    """Validate a published protocol, its frozen source hashes, and both selections."""

    project_root = project_root.resolve()
    protocol_root = protocol_root.resolve()
    summary_path = protocol_root / "protocol_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivedProtocolError(f"Cannot read protocol summary {summary_path}: {exc}") from exc
    if summary.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise DerivedProtocolError("Derived protocol summary schema mismatch.")
    if summary.get("namespace") != PROTOCOL_NAMESPACE or float(summary.get("threshold")) != SOURCEAFIS_THRESHOLD:
        raise DerivedProtocolError("Derived protocol namespace/threshold mismatch.")
    if summary.get("plain_roll_score_used_for_selection") is not False:
        raise DerivedProtocolError("Derived protocol illegally uses plain-roll outcome for selection.")
    for relative, expected in summary["threshold_audit"].items():
        if relative == "path" and file_sha256(Path(expected)) != summary["threshold_audit"]["sha256"]:
            raise DerivedProtocolError("Threshold audit changed after protocol publication.")
    if file_sha256(Path(summary["sourceafis_sidecar_jar_path"])) != summary["sourceafis_sidecar_jar_sha256"]:
        raise DerivedProtocolError("SourceAFIS sidecar JAR changed after protocol publication.")

    dataset_reports = []
    for dataset_summary in summary["datasets"]:
        dataset = dataset_summary["dataset"]
        if dataset not in DATASETS:
            raise DerivedProtocolError(f"Unexpected dataset in protocol summary: {dataset}")
        base_manifest = (project_root / "protocols" / dataset / "plain_roll.csv").resolve()
        manifest = protocol_root / dataset / "plain_roll.csv"
        included_path = protocol_root / dataset / "included_identities.csv"
        included = _read_csv(included_path, INCLUDED_COLUMNS)
        expected_pair_ids = [row["base_plain_roll_pair_id"] for row in included]
        if file_sha256(manifest) != dataset_summary["derived_manifest_sha256"]:
            raise DerivedProtocolError(f"Derived manifest hash mismatch for {dataset}.")
        if file_sha256(included_path) != dataset_summary["included_identities_sha256"]:
            raise DerivedProtocolError(f"Included identities hash mismatch for {dataset}.")
        excluded_path = protocol_root / dataset / "excluded_identities.csv"
        if file_sha256(excluded_path) != dataset_summary["excluded_identities_sha256"]:
            raise DerivedProtocolError(f"Excluded identities hash mismatch for {dataset}.")
        if validate_sources:
            primary = {
                protocol: load_primary_bundle(
                    project_root,
                    data_root,
                    dataset,
                    protocol,
                    validate_source_manifest=True,
                )
                for protocol in PROTOCOLS
            }
            selection = select_dataset_identities(
                dataset,
                primary["plain_self"],
                primary["roll_self"],
                primary["plain_roll"],
            )
            if [row["base_plain_roll_pair_id"] for row in selection.included_rows] != expected_pair_ids:
                raise DerivedProtocolError(f"Published included identities are incomplete for {dataset}.")
            current_source_hashes = {
                "plain_self_manifest_sha256": file_sha256(primary["plain_self"].manifest_path),
                "plain_self_pairs_sha256": file_sha256(primary["plain_self"].bundle_path / RESULT_FILENAME),
                "plain_self_metadata_sha256": file_sha256(primary["plain_self"].bundle_path / METADATA_FILENAME),
                "roll_self_manifest_sha256": file_sha256(primary["roll_self"].manifest_path),
                "roll_self_pairs_sha256": file_sha256(primary["roll_self"].bundle_path / RESULT_FILENAME),
                "roll_self_metadata_sha256": file_sha256(primary["roll_self"].bundle_path / METADATA_FILENAME),
                "base_plain_roll_manifest_sha256": file_sha256(primary["plain_roll"].manifest_path),
                "primary_plain_roll_pairs_sha256": file_sha256(primary["plain_roll"].bundle_path / RESULT_FILENAME),
                "primary_plain_roll_metadata_sha256": file_sha256(primary["plain_roll"].bundle_path / METADATA_FILENAME),
            }
            if current_source_hashes != dataset_summary["source_hashes"]:
                raise DerivedProtocolError(f"Frozen source artifacts changed for {dataset}.")
        report = validate_exact_manifest_subset(
            manifest,
            base_manifest,
            expected_dataset=dataset,
            expected_pair_ids=expected_pair_ids,
        )
        dataset_reports.append(report)
    return {
        "protocol_root": str(protocol_root),
        "protocol_summary_path": str(summary_path),
        "protocol_summary_sha256": file_sha256(summary_path),
        "datasets": dataset_reports,
        "summary": summary,
    }


def _read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise DerivedProtocolError(
                    f"CSV schema mismatch in {path}: expected {expected_columns}, got {reader.fieldnames}"
                )
            rows = list(reader)
    except OSError as exc:
        raise DerivedProtocolError(f"Cannot read CSV {path}: {exc}") from exc
    if any(None in row for row in rows):
        raise DerivedProtocolError(f"CSV contains extra unnamed fields: {path}")
    return rows


def _stringify_row(row: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in row.items():
        if value is None:
            result[key] = ""
        elif isinstance(value, bool):
            result[key] = "true" if value else "false"
        else:
            result[key] = str(value)
    return result


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_stringify_row(row) for row in rows)
    return buffer.getvalue().encode("utf-8")


def _publish_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DerivedProtocolError(f"Immutable artifact already exists with different bytes: {path}")
        return
    _write_bytes(path, payload)


def _startup_dict(startup: SidecarStartup | None) -> dict[str, Any]:
    if startup is None:
        return {}
    return {
        "managed_by_runner": startup.managed_by_runner,
        "service_url": startup.service_url,
        "validation_result": startup.validation_result,
        "command": startup.command,
        "jar_path": startup.jar_path,
        "jar_sha256": startup.jar_sha256,
        "java_executable": startup.java_executable,
    }


def derived_implementation_compatibility(
    *,
    primary_hash: str,
    current_hash: str,
    primary_components: Mapping[str, Any],
    current_components: Mapping[str, Any],
) -> dict[str, Any]:
    """Allow only audited JAR and metadata-only provenance-helper variation.

    ``provenance.py`` computes hashes and repository state before execution. It
    is not imported by the SourceAFIS adapter, Java client, sidecar operations,
    or pair execution loop. Every pair-execution source remains hash-equal.
    """

    primary = json.loads(json.dumps(primary_components, sort_keys=True))
    current = json.loads(json.dumps(current_components, sort_keys=True))
    primary_jar = primary.pop("sidecar_jar_sha256", None)
    current_jar = current.pop("sidecar_jar_sha256", None)
    primary_support = dict(primary.get("benchmark_support_source_sha256", {}))
    current_support = dict(current.get("benchmark_support_source_sha256", {}))
    primary_provenance = primary_support.pop("provenance.py", None)
    current_provenance = current_support.pop("provenance.py", None)
    primary["benchmark_support_source_sha256"] = primary_support
    current["benchmark_support_source_sha256"] = current_support
    execution_components_equal = primary == current
    provenance_helper_equal = primary_provenance == current_provenance
    exact_hash_equal = primary_hash == current_hash and primary_components == current_components
    accepted = execution_components_equal and primary_jar is not None and current_jar is not None
    return {
        "primary_hash": primary_hash,
        "current_hash": current_hash,
        "exact_hash_equal": exact_hash_equal,
        "pair_execution_components_equal": execution_components_equal,
        "primary_sidecar_jar_sha256": primary_jar,
        "current_sidecar_jar_sha256": current_jar,
        "sidecar_jar_sha256_equal": primary_jar == current_jar,
        "primary_provenance_source_sha256": primary_provenance,
        "current_provenance_source_sha256": current_provenance,
        "provenance_source_sha256_equal": provenance_helper_equal,
        "allowed_variations": [
            "sidecar_jar_sha256",
            "benchmark_support_source_sha256.provenance.py",
        ],
        "policy": "require_all_pair_execution_sources_equal_allow_audited_jar_and_metadata_provenance_helper",
        "accepted": accepted,
    }


def _diagnostics_equal(expected: str, actual: Mapping[str, Any]) -> bool:
    try:
        parsed = json.loads(expected)
    except json.JSONDecodeError as exc:
        raise DerivedProtocolError(f"Primary diagnostics are invalid JSON: {expected!r}") from exc
    return parsed == dict(actual)


def runtime_preflight(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
    sidecar_jar: Path | None = None,
    service_url: str = DEFAULT_SERVICE_URL,
    timeout_seconds: float = 120.0,
    sample_count: int = 5,
) -> dict[str, Any]:
    """Prove exact Java-runtime scores on a small derived sample before long runs."""

    if sample_count <= 0:
        raise DerivedProtocolError("Runtime preflight sample_count must be positive.")
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    jar_path = (sidecar_jar or project_root / DEFAULT_SIDECAR_JAR).resolve()
    protocol = load_and_validate_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=protocol_root,
        validate_sources=True,
    )
    if file_sha256(jar_path) != protocol["summary"]["sourceafis_sidecar_jar_sha256"]:
        raise DerivedProtocolError("Runtime preflight JAR does not match the frozen protocol JAR.")

    conditions: list[dict[str, Any]] = []
    for dataset in DATASETS:
        primary = load_primary_bundle(
            project_root, data_root, dataset, "plain_roll", validate_source_manifest=False
        )
        primary_by_pair = {row["pair_id"]: row for row in primary.rows}
        pairs = read_pair_manifest(protocol_root / dataset / "plain_roll.csv")[:sample_count]
        if not pairs:
            raise DerivedProtocolError(f"Derived manifest is empty for {dataset}.")
        with ManagedSourceAfisSidecar(
            jar_path,
            service_url,
            timeout_seconds=timeout_seconds,
        ) as sidecar:
            client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
            try:
                health = client.health()
                validate_health(health)
                adapter = SourceAfisAdapter(client, health=health)
                sample_rows = []
                for pair in pairs:
                    expected = primary_by_pair[pair.pair_id]
                    if expected["status"] != "ok":
                        raise DerivedProtocolError(
                            f"Preflight pair is non-ok in primary result: {pair.pair_id}"
                        )
                    prepared_a = adapter.prepare(pair.path_a, pair.image_metadata_a())
                    prepared_b = adapter.prepare(pair.path_b, pair.image_metadata_b())
                    comparison = adapter.compare(
                        prepared_a.representation,
                        prepared_b.representation,
                    )
                    score_equal = comparison.raw_score == float(expected["raw_score"])
                    diagnostics_equal = (
                        _diagnostics_equal(expected["prepare_a_diagnostics"], prepared_a.diagnostics)
                        and _diagnostics_equal(expected["prepare_b_diagnostics"], prepared_b.diagnostics)
                        and _diagnostics_equal(expected["compare_diagnostics"], comparison.diagnostics)
                    )
                    if not score_equal or not diagnostics_equal:
                        raise DerivedProtocolError(
                            f"Java runtime preflight mismatch for {pair.pair_id}: "
                            f"score_equal={score_equal}, diagnostics_equal={diagnostics_equal}"
                        )
                    sample_rows.append(
                        {
                            "pair_id": pair.pair_id,
                            "raw_score": repr(comparison.raw_score),
                            "exact_score_equal": True,
                            "diagnostics_equal": True,
                        }
                    )
                startup = _startup_dict(sidecar.startup)
                conditions.append(
                    {
                        "dataset": dataset,
                        "sample_count": len(sample_rows),
                        "samples": sample_rows,
                        "java_executable": startup.get("java_executable"),
                        "sidecar_jar_sha256": startup.get("jar_sha256"),
                        "java_runtime_vendor": health.raw.get("java_runtime_vendor"),
                        "java_runtime_version": health.raw.get("java_runtime_version"),
                        "passed": True,
                    }
                )
            finally:
                client.close()

    report = {
        "schema_version": "derived-sourceafis-runtime-preflight-v1",
        "namespace": PROTOCOL_NAMESPACE,
        "protocol_summary_sha256": protocol["protocol_summary_sha256"],
        "sample_count_per_dataset": sample_count,
        "conditions": conditions,
        "exact_score_equality_required": True,
        "diagnostics_equality_required": True,
        "passed": all(condition["passed"] for condition in conditions),
    }
    output = run_root / "runtime_preflight.json"
    _publish_immutable_bytes(output, _json_bytes(report))
    return {**report, "path": str(output), "sha256": file_sha256(output)}


def run_protocol(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
    execution_results_root: Path | None = None,
    sidecar_jar: Path | None = None,
    service_url: str = DEFAULT_SERVICE_URL,
    timeout_seconds: float = 120.0,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run the two derived manifests with isolated JVMs and frozen implementation checks."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    protocol_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    execution_root = (execution_results_root or run_root).resolve()
    if execution_root != run_root:
        if not execution_root.exists() or not os.path.samefile(execution_root, run_root):
            raise DerivedProtocolError(
                "--execution-results-root must be a path alias to the same physical directory as --run-root."
            )
    jar_path = (sidecar_jar or project_root / DEFAULT_SIDECAR_JAR).resolve()
    preflight_path = run_root / "runtime_preflight.json"
    if not preflight_path.is_file():
        raise DerivedProtocolError("Runtime preflight must pass before the full derived runs.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise DerivedProtocolError("Runtime preflight did not pass.")
    protocol = load_and_validate_protocol(
        project_root=project_root,
        data_root=data_root,
        protocol_root=protocol_root,
        validate_sources=True,
    )
    if preflight.get("protocol_summary_sha256") != protocol["protocol_summary_sha256"]:
        raise DerivedProtocolError("Runtime preflight was run against a different protocol summary.")
    if file_sha256(jar_path) != protocol["summary"]["sourceafis_sidecar_jar_sha256"]:
        raise DerivedProtocolError("Run JAR does not match the frozen protocol JAR.")

    dataset_summaries = {item["dataset"]: item for item in protocol["summary"]["datasets"]}
    runs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_summary = dataset_summaries[dataset]
        manifest_path = protocol_root / dataset / "plain_roll.csv"
        included = _read_csv(protocol_root / dataset / "included_identities.csv", INCLUDED_COLUMNS)
        expected_pair_ids = [row["base_plain_roll_pair_id"] for row in included]
        base_manifest = project_root / "protocols" / dataset / "plain_roll.csv"
        dedicated_validator = make_derived_manifest_validator(
            base_manifest=base_manifest,
            expected_dataset=dataset,
            expected_manifest_sha256=dataset_summary["derived_manifest_sha256"],
            expected_pair_ids=expected_pair_ids,
        )
        primary = load_primary_bundle(
            project_root, data_root, dataset, "plain_roll", validate_source_manifest=False
        )
        with ManagedSourceAfisSidecar(
            jar_path,
            service_url,
            timeout_seconds=timeout_seconds,
        ) as sidecar:
            startup = _startup_dict(sidecar.startup)
            client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
            try:
                health = client.health()
                validate_health(health)
                startup_validation = {
                    **startup,
                    "health": health.raw,
                    "health_requests_before_pair_execution": client.health_request_count,
                    "derived_protocol_namespace": PROTOCOL_NAMESPACE,
                    "derived_protocol_summary_path": protocol["protocol_summary_path"],
                    "derived_protocol_summary_sha256": protocol["protocol_summary_sha256"],
                }
                adapter = SourceAfisAdapter(client, health=health)
                context = prepare_run_context(
                    manifest_path=manifest_path,
                    expected_dataset=dataset,
                    expected_protocol="plain_roll",
                    adapter=adapter,
                    results_root=execution_root,
                    startup_validation=startup_validation,
                )
                if context.spec.config_hash != primary.metadata["config_hash"]:
                    raise DerivedProtocolError(
                        f"SourceAFIS config hash differs from primary for {dataset}."
                    )
                implementation = derived_implementation_compatibility(
                    primary_hash=primary.metadata["implementation_hash"],
                    current_hash=context.spec.implementation_hash,
                    primary_components=primary.metadata["implementation_hash_components"],
                    current_components=context.implementation_hash_components,
                )
                if not implementation["accepted"]:
                    raise DerivedProtocolError(
                        f"Current SourceAFIS implementation is incompatible with primary {dataset}: {implementation}"
                    )
                metadata = run_benchmark_manifest(
                    manifest_path=manifest_path,
                    adapter=adapter,
                    expected_dataset=dataset,
                    expected_protocol="plain_roll",
                    results_root=execution_root,
                    startup_validation=startup_validation,
                    data_root=data_root,
                    dedicated_validator=dedicated_validator,
                    skip_existing=skip_existing,
                    progress_callback=lambda completed, total, d=dataset: print(
                        f"[{d}/derived plain_roll] {completed}/{total} measured pairs",
                        file=sys.stderr,
                        flush=True,
                    ),
                )
            finally:
                client.close()
        bundle_path = (
            run_root
            / dataset
            / "plain_roll"
            / METHOD
            / BENCHMARK_CONTRACT_VERSION
            / metadata["config_hash"]
        ).resolve()
        runs.append(
            {
                "dataset": dataset,
                "pair_count": metadata["result"]["row_count"],
                "manifest_sha256": dataset_summary["derived_manifest_sha256"],
                "bundle_path": str(bundle_path),
                "pairs_sha256": metadata["result"]["sha256"],
                "score_payload_sha256": metadata["result"]["score_payload_sha256"],
                "config_hash": metadata["config_hash"],
                "implementation_hash": metadata["implementation_hash"],
                "implementation_compatibility": implementation,
            }
        )
    report = {
        "schema_version": "derived-sourceafis-run-summary-v1",
        "namespace": PROTOCOL_NAMESPACE,
        "protocol_summary_sha256": protocol["protocol_summary_sha256"],
        "sidecar_jar_sha256": file_sha256(jar_path),
        "fresh_jvm_per_dataset": True,
        "runs": runs,
        "primary_artifacts_overwritten": False,
    }
    output = run_root / "run_summary.json"
    _publish_immutable_bytes(output, _json_bytes(report))
    return {**report, "path": str(output), "sha256": file_sha256(output)}


def _load_derived_bundle(
    *,
    run_root: Path,
    protocol_root: Path,
    dataset: str,
) -> PrimaryBundle:
    contract_root = run_root / dataset / "plain_roll" / METHOD / BENCHMARK_CONTRACT_VERSION
    bundles = sorted(
        metadata.parent
        for metadata in contract_root.glob(f"*/{METADATA_FILENAME}")
        if (metadata.parent / RESULT_FILENAME).is_file()
    )
    if len(bundles) != 1:
        raise DerivedProtocolError(f"Expected one derived bundle for {dataset}, found {len(bundles)}.")
    bundle = bundles[0].resolve()
    metadata = json.loads((bundle / METADATA_FILENAME).read_text(encoding="utf-8"))
    raw_spec = dict(metadata["run_spec"])
    raw_spec["manifest_path"] = Path(raw_spec["manifest_path"])
    spec = BenchmarkRunSpec(**raw_spec)
    expected_manifest = (protocol_root / dataset / "plain_roll.csv").resolve()
    if spec.manifest_path.resolve() != expected_manifest:
        raise DerivedProtocolError(f"Derived bundle points to wrong manifest for {dataset}.")
    pairs = read_pair_manifest(expected_manifest)
    validate_result_bundle(
        bundle,
        manifest_records=pairs,
        run_spec=spec,
        score_direction=metadata["score_direction"],
        score_semantics=metadata["score_semantics"],
    )
    rows = read_result_rows(bundle / RESULT_FILENAME)
    return PrimaryBundle(dataset, "plain_roll", expected_manifest, bundle, pairs, rows, metadata)


def compare_derived_to_primary(
    primary_rows: Sequence[Mapping[str, str]],
    derived_rows: Sequence[Mapping[str, str]],
    *,
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare exact score/status/error/diagnostic payloads by pair_id."""

    primary_by_pair = {row["pair_id"]: row for row in primary_rows}
    if len(primary_by_pair) != len(primary_rows):
        raise DerivedProtocolError(f"Duplicate pair_id in primary {dataset} rows.")
    details: list[dict[str, Any]] = []
    for derived in derived_rows:
        pair_id = derived["pair_id"]
        primary = primary_by_pair.get(pair_id)
        if primary is None:
            raise DerivedProtocolError(f"Derived pair is absent from primary result: {pair_id}")
        status_equal = primary["status"] == derived["status"]
        error_equal = primary["error_code"] == derived["error_code"]
        score_text_equal = primary["raw_score"] == derived["raw_score"]
        if primary["raw_score"] == "" or derived["raw_score"] == "":
            score_delta: float | None = None
            score_equal = primary["raw_score"] == derived["raw_score"]
        else:
            score_delta = abs(float(primary["raw_score"]) - float(derived["raw_score"]))
            score_equal = score_delta == 0.0
        prepare_a_equal = primary["prepare_a_diagnostics"] == derived["prepare_a_diagnostics"]
        prepare_b_equal = primary["prepare_b_diagnostics"] == derived["prepare_b_diagnostics"]
        compare_equal = primary["compare_diagnostics"] == derived["compare_diagnostics"]
        reproducible = (
            status_equal
            and error_equal
            and score_equal
            and prepare_a_equal
            and prepare_b_equal
            and compare_equal
        )
        details.append(
            {
                "dataset": dataset,
                "pair_id": pair_id,
                "subject_id": derived["subject_id"],
                "canonical_finger_position": derived["canonical_finger_position"],
                "primary_status": primary["status"],
                "derived_status": derived["status"],
                "status_equal": status_equal,
                "primary_error_code": primary["error_code"],
                "derived_error_code": derived["error_code"],
                "error_code_equal": error_equal,
                "primary_raw_score": primary["raw_score"],
                "derived_raw_score": derived["raw_score"],
                "raw_score_text_equal": score_text_equal,
                "raw_score_abs_delta": score_delta,
                "prepare_a_diagnostics_equal": prepare_a_equal,
                "prepare_b_diagnostics_equal": prepare_b_equal,
                "compare_diagnostics_equal": compare_equal,
                "reproducible": reproducible,
            }
        )
    deltas = [row["raw_score_abs_delta"] for row in details if row["raw_score_abs_delta"] is not None]
    summary = {
        "dataset": dataset,
        "pair_count": len(details),
        "reproducible_pair_count": sum(row["reproducible"] for row in details),
        "mismatch_pair_count": sum(not row["reproducible"] for row in details),
        "raw_score_text_equal_count": sum(row["raw_score_text_equal"] for row in details),
        "status_mismatch_count": sum(not row["status_equal"] for row in details),
        "error_code_mismatch_count": sum(not row["error_code_equal"] for row in details),
        "diagnostics_mismatch_count": sum(
            not (
                row["prepare_a_diagnostics_equal"]
                and row["prepare_b_diagnostics_equal"]
                and row["compare_diagnostics_equal"]
            )
            for row in details
        ),
        "max_absolute_score_delta": max(deltas) if deltas else None,
        "exact_score_reproducibility": all(delta == 0.0 for delta in deltas),
        "passed": all(row["reproducible"] for row in details),
    }
    return details, summary


def _nearest_rank_p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def decision_summary(rows: Sequence[Mapping[str, str]], *, dataset: str) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    scores = [float(row["raw_score"]) for row in ok]
    accepts = sum(score >= SOURCEAFIS_THRESHOLD for score in scores)
    rejects = sum(score < SOURCEAFIS_THRESHOLD for score in scores)
    zero = sum(score == 0.0 for score in scores)
    positive_below = sum(0.0 < score < SOURCEAFIS_THRESHOLD for score in scores)
    method_compare = [float(row["method_compare_ms"]) for row in ok if row["method_compare_ms"] != ""]
    total = [float(row["total_ms"]) for row in rows if row["total_ms"] != ""]
    pair_count = len(rows)
    return {
        "dataset": dataset,
        "threshold": SOURCEAFIS_THRESHOLD,
        "total_pairs": pair_count,
        "status_ok": len(ok),
        "failures": pair_count - len(ok),
        "genuine_accepts": accepts,
        "false_non_matches": rejects,
        "score_zero": zero,
        "positive_below_threshold": positive_below,
        "accept_percentage": (100.0 * accepts / pair_count) if pair_count else None,
        "reject_percentage": (100.0 * rejects / pair_count) if pair_count else None,
        "mean_score": _mean(scores),
        "median_score": _median(scores),
        "mean_method_compare_ms": _mean(method_compare),
        "median_method_compare_ms": _median(method_compare),
        "p95_method_compare_ms": _nearest_rank_p95(method_compare),
        "mean_total_ms": _mean(total),
        "median_total_ms": _median(total),
        "p95_total_ms": _nearest_rank_p95(total),
    }


def _finger_group_count(rows: Sequence[Mapping[str, str]], positions: set[int]) -> int:
    return sum(int(row["canonical_finger_position"]) in positions for row in rows)


def _supervisor_markdown(
    decisions: Sequence[Mapping[str, Any]],
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, str]]],
) -> str:
    lines = ["# Derived SourceAFIS supervisor tables", ""]
    decision_by_dataset = {row["dataset"]: row for row in decisions}
    fields = [
        ("Subjects", lambda rows, decision: len({row["subject_id"] for row in rows})),
        ("Anatomical identities", lambda rows, decision: len(rows)),
        ("Thumb", lambda rows, decision: _finger_group_count(rows, {1, 6})),
        ("Index", lambda rows, decision: _finger_group_count(rows, {2, 7})),
        ("Middle", lambda rows, decision: _finger_group_count(rows, {3, 8})),
        ("Ring", lambda rows, decision: _finger_group_count(rows, {4, 9})),
        ("Little", lambda rows, decision: _finger_group_count(rows, {5, 10})),
        ("All self accepted count", lambda rows, decision: len(rows)),
        ("Plain-roll accepted", lambda rows, decision: decision["genuine_accepts"]),
        ("Plain-roll rejected", lambda rows, decision: decision["false_non_matches"]),
        ("Score = 0", lambda rows, decision: decision["score_zero"]),
        ("0 < score < 40", lambda rows, decision: decision["positive_below_threshold"]),
        ("Accept percentage", lambda rows, decision: decision["accept_percentage"]),
        ("Mean method compare time (ms)", lambda rows, decision: decision["mean_method_compare_ms"]),
        ("Median method compare time (ms)", lambda rows, decision: decision["median_method_compare_ms"]),
        ("P95 method compare time (ms)", lambda rows, decision: decision["p95_method_compare_ms"]),
    ]
    for dataset in DATASETS:
        label = "SD300b" if dataset == "sd300b" else "SD300c"
        rows = rows_by_dataset[dataset]
        decision = decision_by_dataset[dataset]
        lines.extend(
            [
                f"## {label} derived plain_roll",
                "",
                "| Measure | Value |",
                "|---|---:|",
            ]
        )
        for name, getter in fields:
            value = getter(rows, decision)
            if isinstance(value, float):
                rendered = f"{value:.6f}"
            else:
                rendered = str(value)
            lines.append(f"| {name} | {rendered} |")
        lines.append("")
    lines.extend(
        [
            "Every PLAIN image in these tables passed its dataset-specific `plain_self` test at threshold 40.",
            "Every ROLL image passed its dataset-specific `roll_self` test at threshold 40.",
            "PLAIN-to-ROLL comparison was performed only after those two self decisions and availability intersection.",
            "A rejection in `plain_roll` remains an experimental false non-match and never removes the pair.",
            "",
        ]
    )
    return "\n".join(lines)


def _sift_alignment(project_root: Path) -> tuple[dict[str, Any], str]:
    development = project_root / "results" / "sift_geometric" / "development"
    config_path = development / "sift_geometric_config.json"
    decision_path = development / "decision_rule.json"
    if not config_path.is_file() or not decision_path.is_file():
        raise DerivedProtocolError("Frozen SIFT config or decision rule is missing.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    bundles: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            roots = sorted(
                (project_root / "results" / dataset / protocol / "sift_geometric" / BENCHMARK_CONTRACT_VERSION).glob("*")
            )
            roots = [root for root in roots if (root / RESULT_FILENAME).is_file() and (root / METADATA_FILENAME).is_file()]
            if len(roots) != 1:
                raise DerivedProtocolError(
                    f"Expected one reusable SIFT bundle for {dataset}/{protocol}, found {len(roots)}."
                )
            root = roots[0]
            bundles.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "bundle_path": str(root.resolve()),
                    "pairs_sha256": file_sha256(root / RESULT_FILENAME),
                    "metadata_sha256": file_sha256(root / METADATA_FILENAME),
                }
            )
    report = {
        "method": "sift_geometric",
        "method_version": "sift-geometric-v1",
        "reusable_full_self_runs": [
            bundle for bundle in bundles if bundle["protocol"] in SELF_PROTOCOLS
        ],
        "reusable_full_plain_roll_runs": [
            bundle for bundle in bundles if bundle["protocol"] == "plain_roll"
        ],
        "frozen_config_path": str(config_path.resolve()),
        "frozen_config_sha256": file_sha256(config_path),
        "frozen_decision_rule_path": str(decision_path.resolve()),
        "frozen_decision_rule_sha256": file_sha256(decision_path),
        "thresholds_by_dataset": decision.get("thresholds_by_dataset"),
        "existing_joint_cross_dataset_cohort_reusable": False,
        "existing_joint_cross_dataset_cohort_reason": (
            "The new protocol filters each dataset independently; the existing SIFT cohort requires acceptance across both datasets."
        ),
        "new_artifacts_required": [
            "per-dataset SIFT self-accepted identity tables",
            "sd300b SIFT-derived plain_roll manifest",
            "sd300c SIFT-derived plain_roll manifest",
            "sd300b SIFT-derived plain_roll rerun",
            "sd300c SIFT-derived plain_roll rerun",
        ],
        "sift_rerun_performed": False,
    }
    lines = [
        "# SIFT alignment plan",
        "",
        "No SIFT code or existing SIFT artifact was changed, and no SIFT rerun was performed.",
        "",
        "## Reusable artifacts",
        "",
        f"- Frozen configuration: `{config_path}` (SHA-256 `{file_sha256(config_path)}`).",
        f"- Frozen decision rule: `{decision_path}` (SHA-256 `{file_sha256(decision_path)}`).",
        "- All four full `plain_self`/`roll_self` result bundles.",
        "- Both full `plain_roll` result bundles for later exact-score comparison.",
        "",
        "## Required alignment work (not executed)",
        "",
        "1. Validate the six existing SIFT benchmark bundles and frozen decision rule.",
        "2. For SD300b only, select identities accepted by SD300b `plain_self` and SD300b `roll_self`, then intersect with SD300b `plain_roll` availability.",
        "3. Independently repeat the same operation for SD300c using its frozen threshold.",
        "4. Publish two method- and dataset-specific byte-exact derived manifests with inclusion/exclusion provenance.",
        "5. Rerun only those two derived `plain_roll` manifests under the frozen SIFT configuration.",
        "6. Compare rerun scores exactly to the corresponding full primary SIFT rows and publish decision/timing tables.",
        "",
        "The existing joint cross-dataset SIFT cohort must not be reused as the new evaluation population, because it imposes cross-dataset membership equality.",
        "",
    ]
    return report, "\n".join(lines)


def build_reports(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Validate both reruns, compare exact scores, and atomically publish reports."""

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
    reports_root = run_root / "reports"
    candidate = create_candidate_directory(reports_root)
    try:
        decisions: list[dict[str, Any]] = []
        reproducibility_summaries: list[dict[str, Any]] = []
        rows_by_dataset: dict[str, list[dict[str, str]]] = {}
        output_hashes: dict[str, str] = {}
        for dataset in DATASETS:
            primary = load_primary_bundle(
                project_root, data_root, dataset, "plain_roll", validate_source_manifest=False
            )
            derived = _load_derived_bundle(
                run_root=run_root,
                protocol_root=protocol_root,
                dataset=dataset,
            )
            details, comparison = compare_derived_to_primary(
                primary.rows,
                derived.rows,
                dataset=dataset,
            )
            detail_path = candidate / dataset / "score_reproducibility.csv"
            write_csv_atomic(
                [_stringify_row(row) for row in details],
                detail_path,
                REPRODUCIBILITY_COLUMNS,
            )
            output_hashes[f"{dataset}/score_reproducibility.csv"] = file_sha256(detail_path)
            reproducibility_summaries.append(comparison)
            decisions.append(decision_summary(derived.rows, dataset=dataset))
            rows_by_dataset[dataset] = derived.rows

        write_csv_atomic(
            [_stringify_row(row) for row in decisions],
            candidate / "decision_summary.csv",
            DECISION_SUMMARY_COLUMNS,
        )
        write_json_atomic(
            {
                "schema_version": "derived-sourceafis-decision-summary-v1",
                "threshold": SOURCEAFIS_THRESHOLD,
                "datasets": decisions,
            },
            candidate / "decision_summary.json",
        )
        repro_report = {
            "schema_version": "derived-sourceafis-exact-reproducibility-v1",
            "score_abs_tolerance": 0.0,
            "timing_equality_required": False,
            "datasets": reproducibility_summaries,
            "overall": {
                "pair_count": sum(row["pair_count"] for row in reproducibility_summaries),
                "mismatch_pair_count": sum(row["mismatch_pair_count"] for row in reproducibility_summaries),
                "max_absolute_score_delta": max(
                    row["max_absolute_score_delta"] or 0.0 for row in reproducibility_summaries
                ),
                "passed": all(row["passed"] for row in reproducibility_summaries),
            },
        }
        write_json_atomic(repro_report, candidate / "reproducibility_summary.json")
        write_csv_atomic(
            [_stringify_row(row) for row in reproducibility_summaries],
            candidate / "reproducibility_summary.csv",
            list(reproducibility_summaries[0].keys()),
        )
        (candidate / "supervisor_tables.md").write_text(
            _supervisor_markdown(decisions, rows_by_dataset), encoding="utf-8", newline="\n"
        )
        sift_report, sift_markdown = _sift_alignment(project_root)
        (candidate / "sift_alignment_plan.md").write_text(
            sift_markdown, encoding="utf-8", newline="\n"
        )
        write_json_atomic(sift_report, candidate / "sift_alignment_plan.json")
        report_summary = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "namespace": PROTOCOL_NAMESPACE,
            "protocol_summary_sha256": protocol["protocol_summary_sha256"],
            "decision_summaries": decisions,
            "reproducibility": repro_report,
            "sift_alignment": sift_report,
            "sourceafis_rerun_count": 2,
            "sift_rerun_count": 0,
            "primary_artifacts_overwritten": False,
        }
        write_json_atomic(report_summary, candidate / "report_summary.json")
        report_hashes = _directory_file_hashes(candidate)
        write_json_atomic(
            {
                "schema_version": "derived-report-artifact-hashes-v1",
                "files": report_hashes,
            },
            candidate / "report_artifact_hashes.json",
        )
        _publish_immutable_candidate(candidate, reports_root)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)
    final_summary = json.loads((reports_root / "report_summary.json").read_text(encoding="utf-8"))
    if final_summary["reproducibility"]["overall"]["passed"] is not True:
        raise DerivedProtocolError("Exact SourceAFIS score reproducibility failed.")
    return {
        **final_summary,
        "reports_root": str(reports_root),
        "report_summary_sha256": file_sha256(reports_root / "report_summary.json"),
    }


def _snapshot_footer(path: Path) -> dict[str, Any]:
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            last = line
    try:
        footer = json.loads(last)["footer"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise DerivedProtocolError(f"Protected snapshot lacks a valid footer: {path}") from exc
    return footer


def finalize_integrity(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    protocol_root: Path | None = None,
    run_root: Path | None = None,
    before_snapshot: Path,
    after_snapshot: Path,
) -> dict[str, Any]:
    """Publish before/after hash snapshots and a final output artifact manifest."""

    project_root = project_root.resolve()
    protocol_root = (protocol_root or project_root / DEFAULT_PROTOCOL_ROOT).resolve()
    run_root = (run_root or project_root / DEFAULT_RUN_ROOT).resolve()
    before_snapshot = before_snapshot.resolve()
    after_snapshot = after_snapshot.resolve()
    before = _snapshot_footer(before_snapshot)
    after = _snapshot_footer(after_snapshot)
    equal = before == after
    if not equal:
        raise DerivedProtocolError(
            f"Protected artifacts changed: before={before}, after={after}"
        )
    integrity_root = run_root / "integrity"
    candidate = create_candidate_directory(integrity_root)
    try:
        shutil.copyfile(before_snapshot, candidate / "protected_before.jsonl")
        shutil.copyfile(after_snapshot, candidate / "protected_after.jsonl")
        report = {
            "schema_version": "protected-artifact-integrity-report-v1",
            "algorithm": "sha256",
            "before_snapshot_sha256": file_sha256(before_snapshot),
            "after_snapshot_sha256": file_sha256(after_snapshot),
            "before": before,
            "after": after,
            "protected_artifacts_unchanged": True,
        }
        write_json_atomic(report, candidate / "protected_artifact_integrity.json")
        _publish_immutable_candidate(candidate, integrity_root)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)

    artifact_manifest_path = run_root / "artifact_manifest.json"
    files: dict[str, dict[str, Any]] = {}
    for root_name, root in (("derived_protocols", protocol_root), ("derived_runs", run_root)):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.resolve() == artifact_manifest_path.resolve():
                continue
            relative = path.relative_to(root).as_posix()
            files[f"{root_name}/{relative}"] = {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    artifact_manifest = {
        "schema_version": "derived-protocol-artifact-manifest-v1",
        "namespace": PROTOCOL_NAMESPACE,
        "files": files,
        "protected_artifacts_unchanged": True,
        "protected_tree_sha256": before["tree_sha256"],
    }
    _publish_immutable_bytes(artifact_manifest_path, _json_bytes(artifact_manifest))
    return {
        "protected_artifacts_unchanged": True,
        "protected_tree_sha256": before["tree_sha256"],
        "protected_file_count": before["file_count"],
        "artifact_manifest_path": str(artifact_manifest_path),
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "artifact_count": len(files),
    }


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--run-root", type=Path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the per-dataset SourceAFIS self-accept protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_common_paths(prepare)
    prepare.add_argument("--sidecar-jar", type=Path)
    prepare.add_argument("--threshold-audit", type=Path)

    preflight = subparsers.add_parser("preflight")
    _add_common_paths(preflight)
    preflight.add_argument("--sidecar-jar", type=Path)
    preflight.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    preflight.add_argument("--timeout-seconds", type=float, default=120.0)
    preflight.add_argument("--sample-count", type=int, default=5)

    run = subparsers.add_parser("run")
    _add_common_paths(run)
    run.add_argument("--sidecar-jar", type=Path)
    run.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    run.add_argument("--timeout-seconds", type=float, default=120.0)
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument(
        "--execution-results-root",
        type=Path,
        help="Optional short Windows path alias pointing at --run-root; avoids MAX_PATH in candidate bundles.",
    )

    report = subparsers.add_parser("report")
    _add_common_paths(report)

    integrity = subparsers.add_parser("integrity")
    _add_common_paths(integrity)
    integrity.add_argument("--before-snapshot", type=Path, required=True)
    integrity.add_argument("--after-snapshot", type=Path, required=True)
    return parser.parse_args(argv)


def _resolved_protocol_root(args: argparse.Namespace) -> Path | None:
    return args.protocol_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_protocol(
                project_root=args.project_root,
                data_root=args.data_root,
                protocol_root=_resolved_protocol_root(args),
                sidecar_jar=args.sidecar_jar,
                threshold_audit_path=args.threshold_audit,
            )
        elif args.command == "preflight":
            result = runtime_preflight(
                project_root=args.project_root,
                data_root=args.data_root,
                protocol_root=_resolved_protocol_root(args),
                run_root=args.run_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
                sample_count=args.sample_count,
            )
        elif args.command == "run":
            result = run_protocol(
                project_root=args.project_root,
                data_root=args.data_root,
                protocol_root=_resolved_protocol_root(args),
                run_root=args.run_root,
                execution_results_root=args.execution_results_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
                skip_existing=args.skip_existing,
            )
        elif args.command == "report":
            result = build_reports(
                project_root=args.project_root,
                data_root=args.data_root,
                protocol_root=_resolved_protocol_root(args),
                run_root=args.run_root,
            )
        elif args.command == "integrity":
            result = finalize_integrity(
                project_root=args.project_root,
                protocol_root=_resolved_protocol_root(args),
                run_root=args.run_root,
                before_snapshot=args.before_snapshot,
                after_snapshot=args.after_snapshot,
            )
        else:
            raise DerivedProtocolError(f"Unsupported command: {args.command}")
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
