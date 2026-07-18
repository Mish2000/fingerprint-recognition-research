"""Shared biometric-accuracy protocol for SourceAFIS and SIFT.

This module deliberately lives beside, rather than inside, the cold-pair
benchmark runner.  It reuses the frozen method adapters and their raw-score
contract, but caches prepared representations within exactly one
method/dataset/split run.  Thresholds are never consulted while scores are
produced.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from fingerprint_benchmark.contract import (
    COMPARISON_FAILURE,
    HIGHER_IS_MORE_SIMILAR,
    OK,
    PREPARE_A_FAILURE,
    PREPARE_B_FAILURE,
    MethodAdapter,
    MethodExecutionError,
    PrepareOutcome,
    PreparedRepresentation,
)
from fingerprint_benchmark.bundle import (
    create_candidate_directory,
    discard_candidate_directory,
    publish_candidate_directory,
)
from fingerprint_benchmark.derived_protocol import (
    DEFAULT_SERVICE_URL,
    DEFAULT_SIDECAR_JAR,
    derived_implementation_compatibility,
    load_primary_bundle as load_sourceafis_primary_bundle,
)
from fingerprint_benchmark.hashing import canonical_json_bytes, file_sha256, stable_hash
from fingerprint_benchmark.manifest import PairRecord, read_pair_manifest
from fingerprint_benchmark.provenance import implementation_provenance
from fingerprint_benchmark.runner import _execute_pair, prepare_run_context
from fingerprint_benchmark.sift.adapter import SiftGeometricAdapter
from fingerprint_benchmark.sift.development import build_subject_split
from fingerprint_benchmark.sift_derived_protocol import (
    compare_result_rows as compare_sift_result_rows,
    full_deterministic_diagnostics,
    load_primary_bundle as load_sift_primary_bundle,
    runtime_environment_provenance,
    validate_frozen_artifacts,
)
from fingerprint_benchmark.sourceafis_adapter import SourceAfisAdapter
from fingerprint_benchmark.sourceafis_client import SourceAfisSidecarClient, validate_health
from fingerprint_benchmark.sourceafis_sidecar import ManagedSourceAfisSidecar
from fingerprint_benchmark.shared_accuracy_metrics import (
    calibrate_common_threshold,
    compute_operating_metrics,
    discrete_eer,
    roc_det_points,
    trapezoidal_auc,
)
from fingerprint_benchmark.shared_accuracy_integrity import (
    capture_protected_before,
    verify_protected_after,
    write_artifact_manifest,
)


PROTOCOL_VERSION = "sourceafis-sift-shared-accuracy-v1"
SHARED_SPLIT_VERSION = "shared_biometric_accuracy_split_v1"
IMPOSTOR_POLICY_VERSION = "shared-impostor-plain-roll-v1"
IMPOSTOR_SEED = "sourceafis-sift-shared-impostors-2026-07-v1"
PREFLIGHT_SEED = "sourceafis-sift-genuine-preflight-v1"
SCORE_SCHEMA_VERSION = "shared-accuracy-score-v1"
DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
DEFAULT_DATA_ROOT = Path(r"C:\fingerprint-datasets")
DEFAULT_OUTPUT_ROOT = Path("results/shared_accuracy/sourceafis_sift_v1")
DATASETS = ("sd300b", "sd300c")
SPLITS = ("development", "evaluation")
METHODS = ("sourceafis", "sift_geometric")
TARGET_FARS = (0.01, 0.001)
IMPOSTORS_PER_IDENTITY = 10
PREFLIGHT_PAIRS_PER_CONDITION = 100
LEGACY_THRESHOLDS = {"sourceafis": 40.0, "sift_geometric": 4.0}
EXPECTED_SIFT_CONFIG_FILE_SHA256 = (
    "f9f0623ae89752d09c5933d49dc80acc5803863cc8dc7109efb98b96d282f01f"
)
EXPECTED_SIFT_DECISION_FILE_SHA256 = (
    "13e9e29d918f95783d68eecb70f6aa857009ac902417d3ac8d59dcf59b7a98fa"
)


class SharedAccuracyError(ValueError):
    """Raised whenever a safety or reproducibility gate fails."""


Identity = tuple[str, int]


@dataclass(frozen=True)
class AccuracyPair:
    accuracy_pair_id: str
    pair_label: str
    dataset: str
    split: str
    canonical_finger_position: int
    subject_id_a: str
    subject_id_b: str
    ppi: int
    raw_frgp_a: int
    raw_frgp_b: int
    path_a: Path
    path_b: Path
    source_pair_id_a: str
    source_pair_id_b: str

    def image_metadata(self, side: str) -> dict[str, Any]:
        if side not in ("a", "b"):
            raise ValueError(f"Invalid image side: {side}")
        return {
            "accuracy_pair_id": self.accuracy_pair_id,
            "side": side,
            "dataset": self.dataset,
            "protocol": "shared_accuracy_plain_roll",
            "subject_id": self.subject_id_a if side == "a" else self.subject_id_b,
            "canonical_finger_position": self.canonical_finger_position,
            "ppi": self.ppi,
            "raw_frgp": self.raw_frgp_a if side == "a" else self.raw_frgp_b,
            "path": str(self.path_a if side == "a" else self.path_b),
        }


@dataclass(frozen=True)
class _CacheEntry:
    outcome: PrepareOutcome | None
    error_code: str
    error_message: str
    diagnostics: dict[str, Any]


LOGICAL_COLUMNS = [
    "accuracy_pair_id",
    "split",
    "plain_subject_id",
    "roll_subject_id",
    "canonical_finger_position",
    "selection_sha256",
    "selection_rank",
]

MATERIALIZED_COLUMNS = [
    "accuracy_pair_id",
    "pair_label",
    "dataset",
    "split",
    "canonical_finger_position",
    "subject_id_a",
    "subject_id_b",
    "ppi",
    "raw_frgp_a",
    "raw_frgp_b",
    "path_a",
    "path_b",
    "source_pair_id_a",
    "source_pair_id_b",
]

SCORE_COLUMNS = [
    *MATERIALIZED_COLUMNS,
    "score_schema_version",
    "method",
    "method_version",
    "frozen_config_hash",
    "implementation_hash",
    "score_producing_implementation_hash",
    "accuracy_runner_sha256",
    "score_direction",
    "raw_score",
    "status",
    "error_code",
    "error_message",
    "prepare_a_diagnostics_json",
    "prepare_b_diagnostics_json",
    "compare_diagnostics_json",
    "representation_cache_scope_id",
    "prepared_image_key_a",
    "prepared_image_key_b",
    "representation_format_a",
    "representation_format_b",
    "representation_version_a",
    "representation_version_b",
    "score_origin",
    "source_primary_pair_id",
    "source_primary_bundle",
    "source_primary_config_hash",
    "source_primary_implementation_hash",
    "source_primary_sidecar_jar_sha256",
]


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _publish_immutable_bytes(path: Path, payload: bytes) -> None:
    """Publish once; an exact repeat is accepted, any changed repeat is refused."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise SharedAccuracyError(f"Immutable artifact already exists with different content: {path}")
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


def _publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    _publish_immutable_bytes(path, _json_bytes(dict(payload)))


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for raw in rows:
        writer.writerow({column: raw.get(column, "") for column in columns})
    return handle.getvalue().encode("utf-8")


def _publish_immutable_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    _publish_immutable_bytes(path, _csv_bytes(rows, columns))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SharedAccuracyError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SharedAccuracyError(f"JSON artifact must contain an object: {path}")
    return value


def _validate_definition_runner(output_root: Path) -> dict[str, Any]:
    definition_path = output_root / "protocol_definition.json"
    definition = _read_json(definition_path)
    expected = definition.get("accuracy_execution_identity", {}).get(
        "runner_source_sha256"
    )
    actual = file_sha256(Path(__file__).resolve())
    if expected != actual:
        raise SharedAccuracyError(
            f"Shared-accuracy runner changed after protocol definition: expected {expected}, got {actual}."
        )
    return definition


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(expected_columns):
                raise SharedAccuracyError(
                    f"CSV schema mismatch for {path}: {reader.fieldnames} != {list(expected_columns)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise SharedAccuracyError(f"Cannot read CSV artifact {path}: {exc}") from exc
    return rows


def validate_shared_split(project_root: Path) -> dict[str, Any]:
    """Rebuild and exactly match the pre-existing score-independent SIFT split."""

    project_root = project_root.resolve()
    split_path = project_root / "results/sift_geometric/development/subject_split.json"
    existing = _read_json(split_path)
    rebuilt = build_subject_split(project_root)
    if existing != rebuilt:
        raise SharedAccuracyError("Existing SIFT subject split does not exactly match deterministic rebuild.")
    development = list(existing.get("development_subjects", []))
    evaluation = list(existing.get("evaluation_subjects", []))
    if not development or not evaluation or set(development) & set(evaluation):
        raise SharedAccuracyError("Subject split is empty or contains leakage.")
    if len(development) != existing.get("development_count") or len(evaluation) != existing.get(
        "evaluation_count"
    ):
        raise SharedAccuracyError("Subject split counts are inconsistent.")
    if existing.get("dataset_subject_alignment") is not True:
        raise SharedAccuracyError("Existing split does not assert B/C subject alignment.")
    return {
        "shared_split_version": SHARED_SPLIT_VERSION,
        "reused_split_version": existing["split_version"],
        "reused_split_path": str(split_path.resolve()),
        "reused_split_file_sha256": file_sha256(split_path),
        "subject_lists_sha256": existing["subject_lists_sha256"],
        "development_subjects": development,
        "evaluation_subjects": evaluation,
        "development_count": len(development),
        "evaluation_count": len(evaluation),
        "split_rule": existing["split_rule"],
        "validation": {
            "deterministic_rebuild_exact": True,
            "subject_based": True,
            "no_subject_leakage": True,
            "dataset_alignment": True,
            "protocol_alignment": True,
            "score_independent_rule": True,
            "chronology_evidence": (
                "The existing workflow and protected-input artifacts freeze this hash before primary reporting; "
                "the repository has no external signed pre-primary commit, so absolute negative historical proof "
                "of no earlier exploratory viewing is unavailable."
            ),
        },
    }


def _pair_index(pairs: Sequence[PairRecord], label: str) -> dict[Identity, PairRecord]:
    output: dict[Identity, PairRecord] = {}
    for pair in pairs:
        identity = (pair.subject_id, pair.canonical_finger_position)
        if identity in output:
            raise SharedAccuracyError(f"Duplicate plain-roll identity in {label}: {identity}")
        if pair.path_a.resolve() == pair.path_b.resolve():
            raise SharedAccuracyError(f"Genuine pair uses the same image twice in {label}: {pair.pair_id}")
        output[identity] = pair
    return output


def load_base_plain_roll(project_root: Path) -> dict[str, list[PairRecord]]:
    pairs = {
        dataset: read_pair_manifest(project_root / f"protocols/{dataset}/plain_roll.csv")
        for dataset in DATASETS
    }
    indexes = {dataset: _pair_index(rows, dataset) for dataset, rows in pairs.items()}
    if set(indexes["sd300b"]) != set(indexes["sd300c"]):
        raise SharedAccuracyError("SD300b and SD300c plain-roll identity sets differ.")
    for identity in sorted(indexes["sd300b"]):
        left = indexes["sd300b"][identity]
        right = indexes["sd300c"][identity]
        if (
            left.subject_id != right.subject_id
            or left.canonical_finger_position != right.canonical_finger_position
            or left.raw_frgp_a != right.raw_frgp_a
            or left.raw_frgp_b != right.raw_frgp_b
        ):
            raise SharedAccuracyError(f"B/C logical genuine alignment failed for {identity}.")
    return pairs


def split_genuine_pairs(
    pairs: Mapping[str, Sequence[PairRecord]], split_reference: Mapping[str, Any]
) -> dict[tuple[str, str], list[PairRecord]]:
    development = set(split_reference["development_subjects"])
    evaluation = set(split_reference["evaluation_subjects"])
    output: dict[tuple[str, str], list[PairRecord]] = {}
    for dataset in DATASETS:
        assigned: set[str] = set()
        for split, subjects in (("development", development), ("evaluation", evaluation)):
            selected = [pair for pair in pairs[dataset] if pair.subject_id in subjects]
            if any(pair.subject_id in assigned for pair in selected):
                raise SharedAccuracyError(f"Subject leakage while splitting {dataset}.")
            assigned.update(pair.subject_id for pair in selected)
            output[(dataset, split)] = selected
        if {pair.subject_id for pair in pairs[dataset]} != assigned:
            raise SharedAccuracyError(f"Unassigned subject in {dataset}.")
    for split in SPLITS:
        b = [(p.subject_id, p.canonical_finger_position) for p in output[("sd300b", split)]]
        c = [(p.subject_id, p.canonical_finger_position) for p in output[("sd300c", split)]]
        if b != c:
            raise SharedAccuracyError(f"B/C genuine alignment failed in {split}.")
    return output


def _impostor_hash(split: str, plain: Identity, roll: Identity) -> str:
    payload = "|".join(
        (
            SHARED_SPLIT_VERSION,
            split,
            plain[0],
            str(plain[1]),
            roll[0],
            str(roll[1]),
            IMPOSTOR_SEED,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_logical_impostors(
    identities: Sequence[Identity],
    split_subjects: Mapping[str, set[str]],
    *,
    per_identity: int = IMPOSTORS_PER_IDENTITY,
) -> dict[str, list[dict[str, Any]]]:
    """Generate directed plain->roll pairs with a deterministic no-reciprocal orientation."""

    if per_identity <= 0:
        raise SharedAccuracyError("Impostors per identity must be positive.")
    identity_set = set(identities)
    if len(identity_set) != len(identities):
        raise SharedAccuracyError("Logical genuine identities are not unique.")
    output: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        subjects = set(split_subjects[split])
        split_identities = sorted(identity for identity in identity_set if identity[0] in subjects)
        by_finger: dict[int, list[Identity]] = defaultdict(list)
        for identity in split_identities:
            by_finger[identity[1]].append(identity)
        rows: list[dict[str, Any]] = []
        unordered_seen: set[tuple[Identity, Identity]] = set()
        for plain in split_identities:
            candidates: list[tuple[str, Identity]] = []
            for roll in by_finger[plain[1]]:
                if roll[0] == plain[0]:
                    continue
                forward = _impostor_hash(split, plain, roll)
                reverse = _impostor_hash(split, roll, plain)
                # Exactly one direction is eligible.  Lexical identity is the
                # deterministic tie-breaker for the cryptographically remote
                # event of equal hashes.
                if (forward, plain) >= (reverse, roll):
                    continue
                candidates.append((forward, roll))
            candidates.sort(key=lambda item: (item[0], item[1]))
            selected = candidates[:per_identity]
            for rank, (selection_hash, roll) in enumerate(selected, start=1):
                unordered = tuple(sorted((plain, roll)))
                if unordered in unordered_seen:
                    raise SharedAccuracyError(f"Reciprocal/duplicate impostor identity pair: {unordered}")
                unordered_seen.add(unordered)
                rows.append(
                    {
                        "accuracy_pair_id": (
                            f"shared_imp_{split}_{plain[1]:02d}_{plain[0]}_{roll[0]}"
                        ),
                        "split": split,
                        "plain_subject_id": plain[0],
                        "roll_subject_id": roll[0],
                        "canonical_finger_position": plain[1],
                        "selection_sha256": selection_hash,
                        "selection_rank": rank,
                    }
                )
        validate_logical_impostors(rows, split, set(split_identities))
        output[split] = rows
    return output


def validate_logical_impostors(
    rows: Sequence[Mapping[str, Any]], split: str, allowed_identities: set[Identity]
) -> None:
    ids: set[str] = set()
    directed: set[tuple[Identity, Identity]] = set()
    unordered: set[tuple[Identity, Identity]] = set()
    source_counts: Counter[Identity] = Counter()
    for row in rows:
        pair_id = str(row["accuracy_pair_id"])
        plain = (str(row["plain_subject_id"]), int(row["canonical_finger_position"]))
        roll = (str(row["roll_subject_id"]), int(row["canonical_finger_position"]))
        if str(row["split"]) != split or plain not in allowed_identities or roll not in allowed_identities:
            raise SharedAccuracyError(f"Impostor split crossing or unknown identity: {pair_id}")
        if plain[0] == roll[0] or plain == roll:
            raise SharedAccuracyError(f"Genuine contamination in impostor manifest: {pair_id}")
        relation = (plain, roll)
        unordered_relation = tuple(sorted(relation))
        if pair_id in ids or relation in directed or unordered_relation in unordered:
            raise SharedAccuracyError(f"Duplicate impostor pair: {pair_id}")
        ids.add(pair_id)
        directed.add(relation)
        unordered.add(unordered_relation)
        source_counts[plain] += 1
    if any(count > IMPOSTORS_PER_IDENTITY for count in source_counts.values()):
        raise SharedAccuracyError("Impostor count exceeds frozen per-identity policy.")


def materialize_impostors(
    logical: Sequence[Mapping[str, Any]], pairs: Sequence[PairRecord], dataset: str
) -> list[AccuracyPair]:
    index = _pair_index(pairs, dataset)
    output: list[AccuracyPair] = []
    for row in logical:
        finger = int(row["canonical_finger_position"])
        plain = index[(str(row["plain_subject_id"]), finger)]
        roll = index[(str(row["roll_subject_id"]), finger)]
        output.append(
            AccuracyPair(
                accuracy_pair_id=str(row["accuracy_pair_id"]),
                pair_label="impostor",
                dataset=dataset,
                split=str(row["split"]),
                canonical_finger_position=finger,
                subject_id_a=plain.subject_id,
                subject_id_b=roll.subject_id,
                ppi=plain.ppi,
                raw_frgp_a=plain.raw_frgp_a,
                raw_frgp_b=roll.raw_frgp_b,
                path_a=plain.path_a,
                path_b=roll.path_b,
                source_pair_id_a=plain.pair_id,
                source_pair_id_b=roll.pair_id,
            )
        )
    return output


def materialize_genuine(pairs: Sequence[PairRecord], split: str) -> list[AccuracyPair]:
    return [
        AccuracyPair(
            accuracy_pair_id=f"shared_gen_{pair.subject_id}_{pair.canonical_finger_position:02d}",
            pair_label="genuine",
            dataset=pair.dataset,
            split=split,
            canonical_finger_position=pair.canonical_finger_position,
            subject_id_a=pair.subject_id,
            subject_id_b=pair.subject_id,
            ppi=pair.ppi,
            raw_frgp_a=pair.raw_frgp_a,
            raw_frgp_b=pair.raw_frgp_b,
            path_a=pair.path_a,
            path_b=pair.path_b,
            source_pair_id_a=pair.pair_id,
            source_pair_id_b=pair.pair_id,
        )
        for pair in pairs
    ]


def _accuracy_pair_row(pair: AccuracyPair) -> dict[str, str]:
    return {
        "accuracy_pair_id": pair.accuracy_pair_id,
        "pair_label": pair.pair_label,
        "dataset": pair.dataset,
        "split": pair.split,
        "canonical_finger_position": str(pair.canonical_finger_position),
        "subject_id_a": pair.subject_id_a,
        "subject_id_b": pair.subject_id_b,
        "ppi": str(pair.ppi),
        "raw_frgp_a": str(pair.raw_frgp_a),
        "raw_frgp_b": str(pair.raw_frgp_b),
        "path_a": str(pair.path_a),
        "path_b": str(pair.path_b),
        "source_pair_id_a": pair.source_pair_id_a,
        "source_pair_id_b": pair.source_pair_id_b,
    }


def _accuracy_pair_from_row(row: Mapping[str, str]) -> AccuracyPair:
    return AccuracyPair(
        accuracy_pair_id=row["accuracy_pair_id"],
        pair_label=row["pair_label"],
        dataset=row["dataset"],
        split=row["split"],
        canonical_finger_position=int(row["canonical_finger_position"]),
        subject_id_a=row["subject_id_a"],
        subject_id_b=row["subject_id_b"],
        ppi=int(row["ppi"]),
        raw_frgp_a=int(row["raw_frgp_a"]),
        raw_frgp_b=int(row["raw_frgp_b"]),
        path_a=Path(row["path_a"]),
        path_b=Path(row["path_b"]),
        source_pair_id_a=row["source_pair_id_a"],
        source_pair_id_b=row["source_pair_id_b"],
    )


def _logical_summary(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    per_source = Counter(
        (str(row["plain_subject_id"]), int(row["canonical_finger_position"])) for row in rows
    )
    return {
        "split": split,
        "impostor_pair_count": len(rows),
        "subject_count": len(
            {str(row["plain_subject_id"]) for row in rows}
            | {str(row["roll_subject_id"]) for row in rows}
        ),
        "canonical_finger_distribution": dict(
            sorted(Counter(str(row["canonical_finger_position"]) for row in rows).items())
        ),
        "plain_identity_count": len(per_source),
        "identities_with_fewer_than_requested": sum(
            count < IMPOSTORS_PER_IDENTITY for count in per_source.values()
        ),
        "minimum_impostors_per_plain_identity": min(per_source.values()) if per_source else 0,
        "maximum_impostors_per_plain_identity": max(per_source.values()) if per_source else 0,
        "duplicate_directed_count": 0,
        "duplicate_unordered_count": 0,
        "genuine_contamination_count": 0,
    }


def prepare_protocol(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the split/base manifests and publish deterministic pair definitions."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    output_root = (output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    split_reference = validate_shared_split(project_root)
    base = load_base_plain_roll(project_root)
    genuine = split_genuine_pairs(base, split_reference)
    all_identities = sorted(
        (pair.subject_id, pair.canonical_finger_position) for pair in base["sd300b"]
    )
    split_subjects = {
        split: set(split_reference[f"{split}_subjects"]) for split in SPLITS
    }
    logical = generate_logical_impostors(all_identities, split_subjects)

    # Complete every read-only safety/provenance validation before publishing
    # any protocol-definition artifact.  protocol_definition.json is written
    # last and is the completion marker for this stage.
    frozen_sift = validate_frozen_artifacts(project_root)
    if frozen_sift["config_file_sha256"] != EXPECTED_SIFT_CONFIG_FILE_SHA256:
        raise SharedAccuracyError("Frozen SIFT configuration hash changed.")
    if frozen_sift["decision_rule_file_sha256"] != EXPECTED_SIFT_DECISION_FILE_SHA256:
        raise SharedAccuracyError("Frozen SIFT legacy decision-rule hash changed.")
    primary = validate_primary_bundles(project_root, data_root)
    source_config_hashes = {
        primary[("sourceafis", dataset)].metadata["config_hash"] for dataset in DATASETS
    }
    sift_config_hashes = {
        primary[("sift_geometric", dataset)].metadata["config_hash"] for dataset in DATASETS
    }
    sift_implementation_hashes = {
        primary[("sift_geometric", dataset)].metadata["implementation_hash"]
        for dataset in DATASETS
    }
    if len(source_config_hashes) != 1 or len(sift_config_hashes) != 1 or len(
        sift_implementation_hashes
    ) != 1:
        raise SharedAccuracyError("Primary method config/implementation identities are not frozen.")

    summaries: dict[str, Any] = {}
    materialized: dict[tuple[str, str, str], list[AccuracyPair]] = {}
    for split in SPLITS:
        summaries[split] = _logical_summary(logical[split], split)
        for dataset in DATASETS:
            genuine_pairs = materialize_genuine(genuine[(dataset, split)], split)
            impostor_pairs = materialize_impostors(logical[split], base[dataset], dataset)
            if [pair.accuracy_pair_id for pair in impostor_pairs] != [
                str(row["accuracy_pair_id"]) for row in logical[split]
            ]:
                raise SharedAccuracyError(f"Logical/materialized alignment failed for {dataset}/{split}.")
            materialized[(dataset, split, "genuine")] = genuine_pairs
            materialized[(dataset, split, "impostor")] = impostor_pairs

    _publish_immutable_json(output_root / "shared_split_reference.json", split_reference)
    manifests = output_root / "manifests"
    for split in SPLITS:
        _publish_immutable_csv(
            manifests / f"logical_impostors_{split}.csv", logical[split], LOGICAL_COLUMNS
        )
        for dataset in DATASETS:
            for pair_label in ("genuine", "impostor"):
                _publish_immutable_csv(
                    manifests / f"{dataset}_{split}_{pair_label}.csv",
                    (
                        _accuracy_pair_row(pair)
                        for pair in materialized[(dataset, split, pair_label)]
                    ),
                    MATERIALIZED_COLUMNS,
                )

    base_hashes = {
        dataset: file_sha256(project_root / f"protocols/{dataset}/plain_roll.csv")
        for dataset in DATASETS
    }
    manifest_hashes = {
        path.relative_to(output_root).as_posix(): file_sha256(path)
        for path in sorted(manifests.glob("*.csv"))
    }
    report = {
        "schema_version": "shared-accuracy-protocol-definition-v1",
        "protocol_version": PROTOCOL_VERSION,
        "split_reference_sha256": file_sha256(output_root / "shared_split_reference.json"),
        "base_plain_roll_manifest_sha256": base_hashes,
        "manifest_sha256": manifest_hashes,
        "genuine_counts": {
            f"{dataset}/{split}": len(genuine[(dataset, split)])
            for dataset in DATASETS
            for split in SPLITS
        },
        "impostor_policy": {
            "version": IMPOSTOR_POLICY_VERSION,
            "seed": IMPOSTOR_SEED,
            "hash": "sha256",
            "impostors_per_genuine_plain_identity": IMPOSTORS_PER_IDENTITY,
            "same_canonical_finger_only": True,
            "different_subject_required": True,
            "unordered_duplicates_forbidden": True,
            "selection": (
                "For every plain identity, legal same-finger roll candidates in its split are oriented by "
                "the pair of directional SHA-256 values; only the lower direction is eligible, then the "
                "10 lowest forward hashes are selected."
            ),
        },
        "impostor_summaries": summaries,
        "frozen_methods": {
            "sourceafis": {
                "version": "3.18.1",
                "legacy_threshold": 40.0,
                "primary_benchmark_config_hash": next(iter(source_config_hashes)),
                "historical_primary_by_dataset": {
                    dataset: {
                        "implementation_hash": primary[("sourceafis", dataset)].metadata[
                            "implementation_hash"
                        ],
                        "sidecar_jar_sha256": _source_primary_jar(
                            primary[("sourceafis", dataset)]
                        ),
                        "bundle": str(primary[("sourceafis", dataset)].bundle_path),
                    }
                    for dataset in DATASETS
                },
                "sidecar_jar": str((project_root / DEFAULT_SIDECAR_JAR).resolve()),
                "sidecar_jar_sha256": file_sha256(project_root / DEFAULT_SIDECAR_JAR),
            },
            "sift_geometric": {
                "version": "sift-geometric-v1",
                "config_file_sha256": frozen_sift["config_file_sha256"],
                "primary_benchmark_config_hash": next(iter(sift_config_hashes)),
                "primary_implementation_hash": next(iter(sift_implementation_hashes)),
                "legacy_decision_rule_file_sha256": frozen_sift["decision_rule_file_sha256"],
                "legacy_thresholds": frozen_sift["thresholds"],
            },
        },
        "threshold_targets": list(TARGET_FARS),
        "score_direction": HIGHER_IS_MORE_SIMILAR,
        "threshold_selection_data": "development_only",
        "evaluation_access_before_threshold_freeze": False,
        "accuracy_mode": "prepared_representation_run_local_cache",
        "accuracy_execution_identity": {
            "runner_source": str(Path(__file__).resolve()),
            "runner_source_sha256": file_sha256(Path(__file__).resolve()),
            "policy_hash": stable_hash(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "mode": "prepared_representation_run_local_cache",
                    "cache_sharding": "canonical_finger_position",
                    "cache_cross_method": False,
                    "cache_cross_dataset": False,
                    "cache_cross_split": False,
                    "thresholding_during_scoring": False,
                }
            ),
        },
        "cold_pair_results_modified": False,
    }
    _publish_immutable_json(output_root / "protocol_definition.json", report)
    return {**report, "output_root": str(output_root)}


def _source_primary_jar(bundle: Any) -> str | None:
    return bundle.metadata.get("implementation_provenance", {}).get("sidecar_jar_sha256")


def validate_primary_bundles(
    project_root: Path, data_root: Path, *, validate_source_manifests: bool = True
) -> dict[tuple[str, str], Any]:
    """Fully validate the four primary plain-roll bundles used for genuine scores."""

    bundles: dict[tuple[str, str], Any] = {}
    for method in METHODS:
        loader = (
            load_sourceafis_primary_bundle if method == "sourceafis" else load_sift_primary_bundle
        )
        for dataset in DATASETS:
            bundle = loader(
                project_root,
                data_root,
                dataset,
                "plain_roll",
                validate_source_manifest=validate_source_manifests,
            )
            if len(bundle.pairs) != len(bundle.rows):
                raise SharedAccuracyError(f"Primary bundle length mismatch: {method}/{dataset}")
            bundles[(method, dataset)] = bundle
    for method in METHODS:
        left = bundles[(method, "sd300b")]
        right = bundles[(method, "sd300c")]
        left_ids = [(p.subject_id, p.canonical_finger_position) for p in left.pairs]
        right_ids = [(p.subject_id, p.canonical_finger_position) for p in right.pairs]
        if left_ids != right_ids:
            raise SharedAccuracyError(f"Primary B/C pair alignment failed for {method}.")
    return bundles


def _startup_dict(sidecar: ManagedSourceAfisSidecar) -> dict[str, Any]:
    if sidecar.startup is None:
        raise SharedAccuracyError("Managed SourceAFIS sidecar lacks startup provenance.")
    return {
        "managed_by_runner": sidecar.startup.managed_by_runner,
        "service_url": sidecar.startup.service_url,
        "validation_result": sidecar.startup.validation_result,
        "jar_path": sidecar.startup.jar_path,
        "jar_sha256": sidecar.startup.jar_sha256,
        "java_executable": sidecar.startup.java_executable,
    }


def _sample_pairs(pairs: Sequence[PairRecord], method: str, dataset: str) -> list[PairRecord]:
    if len(pairs) < PREFLIGHT_PAIRS_PER_CONDITION:
        raise SharedAccuracyError(f"Not enough genuine pairs for preflight: {method}/{dataset}")
    return sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{PREFLIGHT_SEED}|{method}|{dataset}|{pair.pair_id}".encode("utf-8")
        ).hexdigest(),
    )[:PREFLIGHT_PAIRS_PER_CONDITION]


def _source_preflight_comparison(primary: Mapping[str, str], rerun: Mapping[str, str]) -> dict[str, Any]:
    score_equal = primary.get("raw_score", "") == rerun.get("raw_score", "")
    if not score_equal and primary.get("raw_score", "") and rerun.get("raw_score", ""):
        score_equal = float(primary["raw_score"]) == float(rerun["raw_score"])
    fields = ("status", "error_code", "prepare_a_diagnostics", "prepare_b_diagnostics", "compare_diagnostics")
    field_equality = {field: primary.get(field, "") == rerun.get(field, "") for field in fields}
    return {
        "pair_id": primary["pair_id"],
        "primary_raw_score": primary.get("raw_score", ""),
        "rerun_raw_score": rerun.get("raw_score", ""),
        "exact_numeric_score_equal": score_equal,
        "field_equality": field_equality,
        "passed": score_equal and all(field_equality.values()),
    }


def _preflight_condition(
    *,
    adapter: MethodAdapter,
    startup: Mapping[str, Any],
    method: str,
    dataset: str,
    bundle: Any,
    output_root: Path,
) -> tuple[dict[str, Any], Any]:
    context = prepare_run_context(
        manifest_path=bundle.manifest_path,
        expected_dataset=dataset,
        expected_protocol="plain_roll",
        adapter=adapter,
        results_root=output_root,
        startup_validation=dict(startup),
    )
    primary_by_pair = {row["pair_id"]: row for row in bundle.rows}
    samples = []
    for pair in _sample_pairs(bundle.pairs, method, dataset):
        rerun = _execute_pair(
            pair,
            adapter,
            run_spec=context.spec,
            method_metadata=context.method_metadata,
        )
        primary = primary_by_pair[pair.pair_id]
        if method == "sift_geometric":
            comparison = compare_sift_result_rows(
                primary, rerun, threshold=LEGACY_THRESHOLDS[method], include_residuals=True
            )
        else:
            comparison = _source_preflight_comparison(primary, rerun)
        samples.append(comparison)
        if not comparison["passed"]:
            raise SharedAccuracyError(
                f"Genuine preflight mismatch for {method}/{dataset}/{pair.pair_id}."
            )
    return (
        {
            "method": method,
            "dataset": dataset,
            "sample_count": len(samples),
            "sample_selection_seed": PREFLIGHT_SEED,
            "sample_pair_ids": [sample["pair_id"] for sample in samples],
            "samples": samples,
            "mismatch_count": 0,
            "exact_equality_required": True,
            "passed": True,
            "primary_config_hash": bundle.metadata["config_hash"],
            "primary_implementation_hash": bundle.metadata["implementation_hash"],
            "current_config_hash": context.spec.config_hash,
            "current_implementation_hash": context.spec.implementation_hash,
        },
        context,
    )


def run_genuine_preflight(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path | None = None,
    sidecar_jar: Path | None = None,
    service_url: str = DEFAULT_SERVICE_URL,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Validate provenance and exactly reproduce 100 genuine pairs per method/dataset."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    output_root = (output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    definition = _validate_definition_runner(output_root)
    bundles = validate_primary_bundles(project_root, data_root)
    frozen = validate_frozen_artifacts(project_root)
    jar_path = (sidecar_jar or project_root / DEFAULT_SIDECAR_JAR).resolve()
    if file_sha256(jar_path) != definition["frozen_methods"]["sourceafis"]["sidecar_jar_sha256"]:
        raise SharedAccuracyError("Selected SourceAFIS JAR differs from frozen protocol definition.")

    conditions: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    # One adapter per dataset is intentional: no representation state/cache is
    # shared across dataset preflight runs.
    for dataset in DATASETS:
        adapter = SiftGeometricAdapter(frozen["config"])
        try:
            report, context = _preflight_condition(
                adapter=adapter,
                startup={"shared_accuracy_preflight": True},
                method="sift_geometric",
                dataset=dataset,
                bundle=bundles[("sift_geometric", dataset)],
                output_root=output_root,
            )
            primary = bundles[("sift_geometric", dataset)]
            report["config_hash_equal"] = context.spec.config_hash == primary.metadata["config_hash"]
            report["implementation_hash_equal"] = (
                context.spec.implementation_hash == primary.metadata["implementation_hash"]
            )
            current_environment = runtime_environment_provenance()
            primary_runtime = primary.metadata.get("external_runtime", {})
            primary_build = str(primary_runtime.get("opencv_build_information", ""))
            primary_environment = {
                "opencv_version": primary_runtime.get("opencv_version"),
                "opencv_build_information_sha256": hashlib.sha256(
                    primary_build.encode("utf-8")
                ).hexdigest(),
                "use_optimized": primary_runtime.get("opencv_optimized"),
                "num_threads": primary_runtime.get("opencv_thread_count"),
            }
            environment_checks = {
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
            report["environment"] = current_environment
            report["primary_environment"] = primary_environment
            report["environment_checks"] = environment_checks
            report["environment_passed"] = all(environment_checks.values())
            if (
                not report["config_hash_equal"]
                or not report["implementation_hash_equal"]
                or not report["environment_passed"]
            ):
                raise SharedAccuracyError(f"Frozen SIFT provenance mismatch for {dataset}.")
            conditions.append(report)
            runtime.setdefault(
                "sift_geometric",
                {
                    "method": context.spec.method,
                    "method_version": context.spec.method_version,
                    "frozen_config_hash": context.spec.config_hash,
                    "implementation_hash": context.spec.implementation_hash,
                    "implementation_hash_components": context.implementation_hash_components,
                    "implementation_provenance": context.implementation_provenance,
                    "environment": runtime_environment_provenance(),
                },
            )
        finally:
            adapter.close()

    source_contexts: dict[str, Any] = {}
    source_environments: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        with ManagedSourceAfisSidecar(jar_path, service_url, timeout_seconds=timeout_seconds) as sidecar:
            client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
            try:
                health = client.health()
                validate_health(health)
                adapter = SourceAfisAdapter(client, health=health)
                startup = _startup_dict(sidecar)
                report, context = _preflight_condition(
                    adapter=adapter,
                    startup=startup,
                    method="sourceafis",
                    dataset=dataset,
                    bundle=bundles[("sourceafis", dataset)],
                    output_root=output_root,
                )
                if context.spec.method_version != "3.18.1":
                    raise SharedAccuracyError("SourceAFIS runtime is not official version 3.18.1.")
                if context.spec.config_hash != bundles[("sourceafis", dataset)].metadata["config_hash"]:
                    raise SharedAccuracyError(f"Frozen SourceAFIS config mismatch for {dataset}.")
                report["current_sidecar_jar_sha256"] = startup["jar_sha256"]
                report["primary_sidecar_jar_sha256"] = _source_primary_jar(
                    bundles[("sourceafis", dataset)]
                )
                source_environment = {
                    "java_runtime_vendor": health.raw.get("java_runtime_vendor"),
                    "java_runtime_version": health.raw.get("java_runtime_version"),
                    "sourceafis_version": health.sourceafis_version,
                    "sidecar_contract_version": health.contract_version,
                    "sidecar_implementation_version": health.raw.get(
                        "sidecar_implementation_version"
                    ),
                }
                report["environment"] = source_environment
                source_environments[dataset] = source_environment
                report["same_jar_as_primary"] = (
                    report["current_sidecar_jar_sha256"] == report["primary_sidecar_jar_sha256"]
                )
                compatibility = derived_implementation_compatibility(
                    primary_hash=bundles[("sourceafis", dataset)].metadata["implementation_hash"],
                    current_hash=context.spec.implementation_hash,
                    primary_components=bundles[("sourceafis", dataset)].metadata[
                        "implementation_hash_components"
                    ],
                    current_components=context.implementation_hash_components,
                )
                report["implementation_compatibility"] = compatibility
                if compatibility["pair_execution_components_equal"] is not True:
                    raise SharedAccuracyError(f"SourceAFIS implementation mismatch for {dataset}.")
                report["genuine_reuse_allowed"] = report["same_jar_as_primary"]
                report["genuine_score_policy"] = (
                    "reuse_validated_primary"
                    if report["genuine_reuse_allowed"]
                    else "recompute_in_shared_accuracy_with_frozen_current_jar"
                )
                conditions.append(report)
                source_contexts[dataset] = context
            finally:
                client.close()

    if source_contexts["sd300b"].spec.implementation_hash != source_contexts[
        "sd300c"
    ].spec.implementation_hash:
        raise SharedAccuracyError("SourceAFIS current implementation hash differs between B and C.")
    if source_environments["sd300b"] != source_environments["sd300c"]:
        raise SharedAccuracyError("SourceAFIS runtime environment differs between B and C.")
    source_context = source_contexts["sd300b"]
    runtime["sourceafis"] = {
        "method": source_context.spec.method,
        "method_version": source_context.spec.method_version,
        "frozen_config_hash": source_context.spec.config_hash,
        "implementation_hash": source_context.spec.implementation_hash,
        "implementation_hash_components": source_context.implementation_hash_components,
        "implementation_provenance": source_context.implementation_provenance,
        "sidecar_jar_sha256": file_sha256(jar_path),
        "environment": source_environments["sd300b"],
    }
    runtime["accuracy_runner_sha256"] = file_sha256(Path(__file__).resolve())
    report = {
        "schema_version": "shared-accuracy-genuine-preflight-v1",
        "protocol_definition_sha256": file_sha256(output_root / "protocol_definition.json"),
        "sample_count_per_method_dataset": PREFLIGHT_PAIRS_PER_CONDITION,
        "conditions": conditions,
        "passed": all(condition["passed"] for condition in conditions),
        "evaluation_scoring_permitted": True,
        "runtime_provenance": runtime,
    }
    _publish_immutable_json(output_root / "provenance/genuine_preflight.json", report)
    _publish_immutable_json(output_root / "provenance/method_runtime.json", runtime)
    return report


def _load_materialized(output_root: Path, dataset: str, split: str, pair_label: str) -> list[AccuracyPair]:
    path = output_root / f"manifests/{dataset}_{split}_{pair_label}.csv"
    rows = _read_csv(path, MATERIALIZED_COLUMNS)
    pairs = [_accuracy_pair_from_row(row) for row in rows]
    for pair in pairs:
        if pair.dataset != dataset or pair.split != split or pair.pair_label != pair_label:
            raise SharedAccuracyError(f"Materialized manifest identity mismatch: {path}")
        if pair.canonical_finger_position < 1 or pair.canonical_finger_position > 10:
            raise SharedAccuracyError(f"Invalid canonical finger in {path}: {pair.accuracy_pair_id}")
        if pair_label == "genuine" and pair.subject_id_a != pair.subject_id_b:
            raise SharedAccuracyError(f"Genuine row has different subjects: {pair.accuracy_pair_id}")
        if pair_label == "impostor" and pair.subject_id_a == pair.subject_id_b:
            raise SharedAccuracyError(f"Impostor row contains a genuine identity: {pair.accuracy_pair_id}")
    if len({pair.accuracy_pair_id for pair in pairs}) != len(pairs):
        raise SharedAccuracyError(f"Duplicate accuracy_pair_id in {path}")
    return pairs


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in sorted(value.items())
            if not key.lower().endswith("_ms") and "timing" not in key.lower()
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _prepare_diagnostics(method: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return _without_timing(dict(value)) if method == "sift_geometric" else {}


def _compare_diagnostics(method: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if method == "sift_geometric":
        return full_deterministic_diagnostics(value)
    return {}


def _image_key(pair: AccuracyPair, side: str) -> str:
    metadata = pair.image_metadata(side)
    path = pair.path_a if side == "a" else pair.path_b
    return stable_hash(
        {
            "resolved_path": str(path.resolve()).lower(),
            "dataset": pair.dataset,
            "subject_id": metadata["subject_id"],
            "canonical_finger_position": metadata["canonical_finger_position"],
            "ppi": metadata["ppi"],
            "raw_frgp": metadata["raw_frgp"],
        }
    )


def _failure_entry(exc: MethodExecutionError) -> _CacheEntry:
    return _CacheEntry(
        outcome=None,
        error_code=str(exc.error_code).strip(),
        error_message=str(exc.message).strip(),
        diagnostics=dict(exc.diagnostics),
    )


def _prepare_once(
    adapter: MethodAdapter, pair: AccuracyPair, side: str, method: str
) -> _CacheEntry:
    try:
        outcome = adapter.prepare(
            pair.path_a if side == "a" else pair.path_b,
            pair.image_metadata(side),
        )
    except MethodExecutionError as exc:
        return _failure_entry(exc)
    if not isinstance(outcome, PrepareOutcome):
        raise SharedAccuracyError(
            f"Adapter {method} returned {type(outcome).__name__} instead of PrepareOutcome."
        )
    return _CacheEntry(
        outcome=outcome,
        error_code="",
        error_message="",
        diagnostics=_prepare_diagnostics(method, outcome.diagnostics),
    )


def _score_base(
    pair: AccuracyPair,
    *,
    method: str,
    method_version: str,
    frozen_config_hash: str,
    implementation_hash: str,
    accuracy_runner_sha256: str,
    cache_scope_id: str,
) -> dict[str, str]:
    return {
        **_accuracy_pair_row(pair),
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "method": method,
        "method_version": method_version,
        "frozen_config_hash": frozen_config_hash,
        "implementation_hash": implementation_hash,
        "score_producing_implementation_hash": implementation_hash,
        "accuracy_runner_sha256": accuracy_runner_sha256,
        "score_direction": HIGHER_IS_MORE_SIMILAR,
        "raw_score": "",
        "status": "",
        "error_code": "",
        "error_message": "",
        "prepare_a_diagnostics_json": "{}",
        "prepare_b_diagnostics_json": "{}",
        "compare_diagnostics_json": "{}",
        "representation_cache_scope_id": cache_scope_id,
        "prepared_image_key_a": _image_key(pair, "a"),
        "prepared_image_key_b": _image_key(pair, "b"),
        "representation_format_a": "",
        "representation_format_b": "",
        "representation_version_a": "",
        "representation_version_b": "",
        "score_origin": "prepared_representation_accuracy_mode",
        "source_primary_pair_id": "",
        "source_primary_bundle": "",
        "source_primary_config_hash": "",
        "source_primary_implementation_hash": "",
        "source_primary_sidecar_jar_sha256": "",
    }


def run_prepared_scores(
    *,
    adapter: MethodAdapter,
    pairs: Sequence[AccuracyPair],
    method: str,
    method_version: str,
    frozen_config_hash: str,
    implementation_hash: str,
    accuracy_runner_sha256: str,
    cache_scope_id: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Score pairs with one ephemeral preparation per unique image in this scope.

    The cache is sharded by canonical finger to bound SIFT memory.  Base data
    guarantees that an image belongs to exactly one canonical finger, which is
    checked here before each shard is released.
    """

    if not pairs:
        raise SharedAccuracyError("Prepared score run cannot be empty.")
    if any(pair.dataset != pairs[0].dataset or pair.split != pairs[0].split for pair in pairs):
        raise SharedAccuracyError("Prepared run mixed dataset or split cache scopes.")
    by_finger: dict[int, list[AccuracyPair]] = defaultdict(list)
    for pair in pairs:
        by_finger[pair.canonical_finger_position].append(pair)
    seen_image_keys: set[str] = set()
    rows: list[dict[str, str]] = []
    prepare_invocations = 0
    prepare_failures = 0
    comparison_failures = 0
    started = perf_counter()
    for finger in sorted(by_finger):
        shard = by_finger[finger]
        print(
            f"[shared-accuracy] {method}/{pairs[0].dataset}/{pairs[0].split}: "
            f"finger {finger}, {len(shard)} comparisons",
            file=sys.stderr,
            flush=True,
        )
        cache: dict[str, _CacheEntry] = {}
        requests: dict[str, tuple[AccuracyPair, str]] = {}
        for pair in shard:
            for side in ("a", "b"):
                key = _image_key(pair, side)
                path = (pair.path_a if side == "a" else pair.path_b).resolve()
                if key in requests:
                    old_pair, old_side = requests[key]
                    old_path = (old_pair.path_a if old_side == "a" else old_pair.path_b).resolve()
                    if old_path != path:
                        raise SharedAccuracyError(f"Cache-key collision: {key}")
                else:
                    requests[key] = (pair, side)
        if set(requests) & seen_image_keys:
            raise SharedAccuracyError("An image crossed canonical-finger cache shards.")
        for key in sorted(requests):
            pair, side = requests[key]
            cache[key] = _prepare_once(adapter, pair, side, method)
            prepare_invocations += 1
            if cache[key].outcome is None:
                prepare_failures += 1
        seen_image_keys.update(requests)

        for pair in shard:
            row = _score_base(
                pair,
                method=method,
                method_version=method_version,
                frozen_config_hash=frozen_config_hash,
                implementation_hash=implementation_hash,
                accuracy_runner_sha256=accuracy_runner_sha256,
                cache_scope_id=cache_scope_id,
            )
            left = cache[row["prepared_image_key_a"]]
            right = cache[row["prepared_image_key_b"]]
            row["prepare_a_diagnostics_json"] = _compact_json(left.diagnostics)
            row["prepare_b_diagnostics_json"] = _compact_json(right.diagnostics)
            if left.outcome is None:
                row.update(
                    status=PREPARE_A_FAILURE,
                    error_code=left.error_code,
                    error_message=left.error_message,
                )
                rows.append(row)
                continue
            if right.outcome is None:
                row.update(
                    status=PREPARE_B_FAILURE,
                    error_code=right.error_code,
                    error_message=right.error_message,
                )
                rows.append(row)
                continue
            representation_a = left.outcome.representation
            representation_b = right.outcome.representation
            if not isinstance(representation_a, PreparedRepresentation) or not isinstance(
                representation_b, PreparedRepresentation
            ):
                raise SharedAccuracyError("Prepared cache contains an invalid representation.")
            row.update(
                representation_format_a=representation_a.representation_format,
                representation_format_b=representation_b.representation_format,
                representation_version_a=representation_a.representation_version,
                representation_version_b=representation_b.representation_version,
            )
            try:
                comparison = adapter.compare(representation_a, representation_b)
            except MethodExecutionError as exc:
                row.update(
                    status=COMPARISON_FAILURE,
                    error_code=str(exc.error_code).strip(),
                    error_message=str(exc.message).strip(),
                    compare_diagnostics_json=_compact_json(
                        _compare_diagnostics(method, exc.diagnostics)
                    ),
                )
                comparison_failures += 1
                rows.append(row)
                continue
            score = float(comparison.raw_score)
            if not math.isfinite(score):
                row.update(
                    status=COMPARISON_FAILURE,
                    error_code="non_finite_raw_score",
                    error_message="Method returned a non-finite raw score.",
                )
                comparison_failures += 1
                rows.append(row)
                continue
            row.update(
                raw_score=repr(score),
                status=OK,
                compare_diagnostics_json=_compact_json(
                    _compare_diagnostics(method, comparison.diagnostics)
                ),
            )
            rows.append(row)
        # Representations are run-local and are intentionally not serialized.
        cache.clear()
        print(
            f"[shared-accuracy] {method}/{pairs[0].dataset}/{pairs[0].split}: "
            f"finger {finger} complete",
            file=sys.stderr,
            flush=True,
        )

    expected_unique = len(
        {
            _image_key(pair, side)
            for pair in pairs
            for side in ("a", "b")
        }
    )
    if prepare_invocations != expected_unique or len(seen_image_keys) != expected_unique:
        raise SharedAccuracyError(
            f"Prepared-cache count mismatch: invocations={prepare_invocations}, unique={expected_unique}."
        )
    metadata = {
        "mode": "prepared_representation_accuracy",
        "cache_scope_id": cache_scope_id,
        "cache_instance_policy": "new_empty_in_memory_cache_each_invocation",
        "cache_sharding": "canonical_finger_position; released after every finger",
        "cache_persisted": False,
        "cache_shared_across_methods": False,
        "cache_shared_across_datasets": False,
        "cache_shared_across_splits": False,
        "cache_shared_across_runs": False,
        "unique_image_count": expected_unique,
        "prepare_invocation_count": prepare_invocations,
        "prepare_failure_count": prepare_failures,
        "comparison_count": len(pairs),
        "comparison_failure_count": comparison_failures,
        "wall_runtime_seconds": perf_counter() - started,
        "wall_runtime_interpretation": "accuracy-pipeline runtime; not production or cold-pair latency",
    }
    return rows, metadata


def _primary_diagnostics(method: str, raw: str, *, compare: bool) -> dict[str, Any]:
    if raw in ("", None):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SharedAccuracyError("Primary result contains invalid diagnostics JSON.") from exc
    if not isinstance(parsed, dict):
        raise SharedAccuracyError("Primary diagnostics must be an object.")
    return _compare_diagnostics(method, parsed) if compare else _prepare_diagnostics(method, parsed)


def project_reused_genuine_scores(
    *,
    pairs: Sequence[AccuracyPair],
    bundle: Any,
    method: str,
    runtime: Mapping[str, Any],
) -> list[dict[str, str]]:
    primary = {row["pair_id"]: row for row in bundle.rows}
    output: list[dict[str, str]] = []
    for pair in pairs:
        row = _score_base(
            pair,
            method=method,
            method_version=str(runtime["method_version"]),
            frozen_config_hash=str(runtime["frozen_config_hash"]),
            implementation_hash=str(runtime["implementation_hash"]),
            accuracy_runner_sha256=str(runtime["accuracy_runner_sha256"]),
            cache_scope_id="not_applicable_reused_validated_primary",
        )
        source = primary.get(pair.source_pair_id_a)
        if source is None or source["pair_id"] != pair.source_pair_id_b:
            raise SharedAccuracyError(f"Missing primary genuine score: {pair.source_pair_id_a}")
        row.update(
            raw_score=source.get("raw_score", ""),
            status=source["status"],
            error_code=source.get("error_code", ""),
            error_message=source.get("error_message", ""),
            prepare_a_diagnostics_json=_compact_json(
                _primary_diagnostics(method, source.get("prepare_a_diagnostics", "{}"), compare=False)
            ),
            prepare_b_diagnostics_json=_compact_json(
                _primary_diagnostics(method, source.get("prepare_b_diagnostics", "{}"), compare=False)
            ),
            compare_diagnostics_json=_compact_json(
                _primary_diagnostics(method, source.get("compare_diagnostics", "{}"), compare=True)
            ),
            score_origin="reused_validated_primary_plain_roll_after_100_pair_preflight",
            source_primary_pair_id=source["pair_id"],
            source_primary_bundle=str(bundle.bundle_path),
            source_primary_config_hash=bundle.metadata["config_hash"],
            source_primary_implementation_hash=bundle.metadata["implementation_hash"],
            source_primary_sidecar_jar_sha256=(
                _source_primary_jar(bundle) or ""
            ),
            score_producing_implementation_hash=bundle.metadata["implementation_hash"],
        )
        output.append(row)
    return output


def _condition_from_preflight(
    preflight: Mapping[str, Any], method: str, dataset: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in preflight.get("conditions", [])
        if item.get("method") == method and item.get("dataset") == dataset
    ]
    if len(matches) != 1 or matches[0].get("passed") is not True:
        raise SharedAccuracyError(f"Missing passed preflight condition: {method}/{dataset}")
    return matches[0]


def _validate_score_rows(
    rows: Sequence[Mapping[str, str]],
    planned: Sequence[AccuracyPair],
    *,
    method: str,
    dataset: str,
    split: str,
    pair_label: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ids = [pair.accuracy_pair_id for pair in planned]
    actual_ids = [row["accuracy_pair_id"] for row in rows]
    if actual_ids != expected_ids:
        raise SharedAccuracyError(
            f"Score order/alignment mismatch for {method}/{dataset}/{split}/{pair_label}."
        )
    failures = Counter()
    for row in rows:
        if (
            row["method"] != method
            or row["dataset"] != dataset
            or row["split"] != split
            or row["pair_label"] != pair_label
            or row["score_direction"] != HIGHER_IS_MORE_SIMILAR
            or row["frozen_config_hash"] != runtime["frozen_config_hash"]
            or row["implementation_hash"] != runtime["implementation_hash"]
        ):
            raise SharedAccuracyError("Score row provenance/identity mismatch.")
        producing_hash = row["score_producing_implementation_hash"]
        if not producing_hash:
            raise SharedAccuracyError("Score row lacks producing implementation provenance.")
        if row["score_origin"] == "prepared_representation_accuracy_mode":
            if producing_hash != runtime["implementation_hash"]:
                raise SharedAccuracyError("Prepared score row has the wrong producing implementation hash.")
        elif row["score_origin"] == "reused_validated_primary_plain_roll_after_100_pair_preflight":
            if producing_hash != row["source_primary_implementation_hash"]:
                raise SharedAccuracyError("Reused score row misstates its primary implementation hash.")
        else:
            raise SharedAccuracyError(f"Unknown score origin: {row['score_origin']}")
        if row["status"] == OK:
            try:
                score = float(row["raw_score"])
            except ValueError as exc:
                raise SharedAccuracyError("Successful score row is non-numeric.") from exc
            if not math.isfinite(score) or row["error_code"]:
                raise SharedAccuracyError("Successful score row contains invalid score/error.")
        else:
            if row["status"] not in (PREPARE_A_FAILURE, PREPARE_B_FAILURE, COMPARISON_FAILURE):
                raise SharedAccuracyError(f"Unknown score status: {row['status']}")
            if row["raw_score"] or not row["error_code"]:
                raise SharedAccuracyError("Failure score row has score or lacks error code.")
            failures[row["status"]] += 1
    return {
        "planned_count": len(planned),
        "scoreable_count": len(rows) - sum(failures.values()),
        "failure_count": sum(failures.values()),
        "failure_counts": dict(sorted(failures.items())),
    }


def _candidate_score_path(candidate: Path, pair_label: str) -> Path:
    return candidate / f"{pair_label}.csv"


def _validate_existing_score_condition(
    directory: Path,
    *,
    method: str,
    dataset: str,
    split: str,
    planned: Mapping[str, Sequence[AccuracyPair]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _read_json(directory / "run_metadata.json")
    for pair_label in ("genuine", "impostor"):
        rows = _read_csv(directory / f"{pair_label}.csv", SCORE_COLUMNS)
        _validate_score_rows(
            rows,
            planned[pair_label],
            method=method,
            dataset=dataset,
            split=split,
            pair_label=pair_label,
            runtime=runtime,
        )
        expected_hash = metadata["score_files"][pair_label]["sha256"]
        if file_sha256(directory / f"{pair_label}.csv") != expected_hash:
            raise SharedAccuracyError(f"Published score file hash mismatch: {directory}/{pair_label}.csv")
    return metadata


def _score_condition(
    *,
    project_root: Path,
    output_root: Path,
    method: str,
    dataset: str,
    split: str,
    adapter: MethodAdapter,
    startup: Mapping[str, Any],
    bundles: Mapping[tuple[str, str], Any],
    preflight: Mapping[str, Any],
    runtime_all: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = {
        **runtime_all[method],
        "accuracy_runner_sha256": runtime_all["accuracy_runner_sha256"],
    }
    context = prepare_run_context(
        manifest_path=project_root / f"protocols/{dataset}/plain_roll.csv",
        expected_dataset=dataset,
        expected_protocol="plain_roll",
        adapter=adapter,
        results_root=output_root,
        startup_validation=dict(startup),
    )
    if (
        context.spec.config_hash != runtime["frozen_config_hash"]
        or context.spec.implementation_hash != runtime["implementation_hash"]
        or file_sha256(Path(__file__).resolve()) != runtime["accuracy_runner_sha256"]
    ):
        raise SharedAccuracyError(f"Runtime provenance changed after preflight for {method}/{dataset}.")
    condition = _condition_from_preflight(preflight, method, dataset)
    reuse_genuine = method == "sift_geometric" or condition.get("genuine_reuse_allowed") is True
    genuine = _load_materialized(output_root, dataset, split, "genuine")
    impostor = _load_materialized(output_root, dataset, split, "impostor")
    planned = {"genuine": genuine, "impostor": impostor}
    final = output_root / f"scores/{method}/{dataset}/{split}"
    if final.exists():
        return _validate_existing_score_condition(
            final,
            method=method,
            dataset=dataset,
            split=split,
            planned=planned,
            runtime=runtime,
        )

    candidate = create_candidate_directory(final)
    try:
        prepared_pairs = list(impostor) if reuse_genuine else [*genuine, *impostor]
        prepared_rows, cache_metadata = run_prepared_scores(
            adapter=adapter,
            pairs=prepared_pairs,
            method=method,
            method_version=str(runtime["method_version"]),
            frozen_config_hash=str(runtime["frozen_config_hash"]),
            implementation_hash=str(runtime["implementation_hash"]),
            accuracy_runner_sha256=str(runtime["accuracy_runner_sha256"]),
            cache_scope_id=stable_hash(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "method": method,
                    "dataset": dataset,
                    "split": split,
                    "config_hash": runtime["frozen_config_hash"],
                    "implementation_hash": runtime["implementation_hash"],
                    "accuracy_runner_sha256": runtime["accuracy_runner_sha256"],
                }
            ),
        )
        prepared_by_id = {row["accuracy_pair_id"]: row for row in prepared_rows}
        if len(prepared_by_id) != len(prepared_rows):
            raise SharedAccuracyError("Prepared runner returned duplicate accuracy_pair_id values.")
        by_label = {
            label: [
                prepared_by_id[pair.accuracy_pair_id]
                for pair in planned[label]
                if pair.accuracy_pair_id in prepared_by_id
            ]
            for label in ("genuine", "impostor")
        }
        if reuse_genuine:
            by_label["genuine"] = project_reused_genuine_scores(
                pairs=genuine,
                bundle=bundles[(method, dataset)],
                method=method,
                runtime=runtime,
            )
        validations: dict[str, Any] = {}
        for pair_label in ("genuine", "impostor"):
            validations[pair_label] = _validate_score_rows(
                by_label[pair_label],
                planned[pair_label],
                method=method,
                dataset=dataset,
                split=split,
                pair_label=pair_label,
                runtime=runtime,
            )
            _publish_immutable_csv(
                _candidate_score_path(candidate, pair_label),
                by_label[pair_label],
                SCORE_COLUMNS,
            )
        metadata = {
            "schema_version": "shared-accuracy-score-run-v1",
            "protocol_version": PROTOCOL_VERSION,
            "method": method,
            "dataset": dataset,
            "split": split,
            "score_direction": HIGHER_IS_MORE_SIMILAR,
            "thresholding_during_scoring": False,
            "frozen_config_hash": runtime["frozen_config_hash"],
            "implementation_hash": runtime["implementation_hash"],
            "accuracy_runner_sha256": runtime["accuracy_runner_sha256"],
            "genuine_score_origin": (
                "reused_validated_primary" if reuse_genuine else "recomputed_current_frozen_jar"
            ),
            "genuine_reuse_reason": (
                "primary implementation/config and 100-pair exact preflight passed"
                if reuse_genuine
                else "historical primary JAR differs from the one frozen for both B and C"
            ),
            "cache": cache_metadata,
            "score_files": {},
            "validation": validations,
        }
        for pair_label in ("genuine", "impostor"):
            path = candidate / f"{pair_label}.csv"
            metadata["score_files"][pair_label] = {
                "filename": path.name,
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            }
        _publish_immutable_json(candidate / "run_metadata.json", metadata)
        publish_candidate_directory(candidate, final)
        candidate = Path()
    finally:
        if candidate != Path():
            discard_candidate_directory(candidate)
    return _validate_existing_score_condition(
        final,
        method=method,
        dataset=dataset,
        split=split,
        planned=planned,
        runtime=runtime,
    )


def score_split(
    split: str,
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path | None = None,
    sidecar_jar: Path | None = None,
    service_url: str = DEFAULT_SERVICE_URL,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Produce threshold-free score files for one split after all safety gates."""

    if split not in SPLITS:
        raise SharedAccuracyError(f"Unsupported split: {split}")
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    output_root = (output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    preflight_path = output_root / "provenance/genuine_preflight.json"
    runtime_path = output_root / "provenance/method_runtime.json"
    _validate_definition_runner(output_root)
    preflight = _read_json(preflight_path)
    runtime = _read_json(runtime_path)
    if preflight.get("passed") is not True:
        raise SharedAccuracyError("Mandatory genuine preflight did not pass.")
    if preflight.get("protocol_definition_sha256") != file_sha256(
        output_root / "protocol_definition.json"
    ):
        raise SharedAccuracyError("Genuine preflight belongs to a different protocol definition.")
    if runtime.get("accuracy_runner_sha256") != file_sha256(Path(__file__).resolve()):
        raise SharedAccuracyError("Accuracy runner changed after preflight.")
    if preflight.get("runtime_provenance") != runtime:
        raise SharedAccuracyError("Method runtime provenance does not match the passed preflight.")
    if split == "evaluation":
        thresholds = output_root / "calibration/frozen_thresholds.json"
        gate = output_root / "calibration/evaluation_safety_gate.json"
        gate_payload = _read_json(gate) if gate.is_file() else {}
        if (
            not thresholds.is_file()
            or gate_payload.get("passed") is not True
            or gate_payload.get("thresholds_sha256") != file_sha256(thresholds)
        ):
            raise SharedAccuracyError("Evaluation is forbidden before calibrated thresholds and gate are frozen.")
    bundles = validate_primary_bundles(
        project_root, data_root, validate_source_manifests=False
    )
    frozen = validate_frozen_artifacts(project_root)
    results: list[dict[str, Any]] = []

    for dataset in DATASETS:
        adapter = SiftGeometricAdapter(frozen["config"])
        try:
            current_sift_environment = runtime_environment_provenance()
            if current_sift_environment != runtime["sift_geometric"]["environment"]:
                raise SharedAccuracyError(
                    f"SIFT OpenCV/runtime environment changed after preflight for {dataset}."
                )
            results.append(
                _score_condition(
                    project_root=project_root,
                    output_root=output_root,
                    method="sift_geometric",
                    dataset=dataset,
                    split=split,
                    adapter=adapter,
                    startup={"shared_accuracy_score_run": True},
                    bundles=bundles,
                    preflight=preflight,
                    runtime_all=runtime,
                )
            )
        finally:
            adapter.close()

    jar_path = (sidecar_jar or project_root / DEFAULT_SIDECAR_JAR).resolve()
    if file_sha256(jar_path) != runtime["sourceafis"]["sidecar_jar_sha256"]:
        raise SharedAccuracyError("SourceAFIS JAR changed after preflight.")
    for dataset in DATASETS:
        with ManagedSourceAfisSidecar(jar_path, service_url, timeout_seconds=timeout_seconds) as sidecar:
            client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
            try:
                health = client.health()
                validate_health(health)
                current_source_environment = {
                    "java_runtime_vendor": health.raw.get("java_runtime_vendor"),
                    "java_runtime_version": health.raw.get("java_runtime_version"),
                    "sourceafis_version": health.sourceafis_version,
                    "sidecar_contract_version": health.contract_version,
                    "sidecar_implementation_version": health.raw.get(
                        "sidecar_implementation_version"
                    ),
                }
                if current_source_environment != runtime["sourceafis"]["environment"]:
                    raise SharedAccuracyError(
                        f"SourceAFIS Java/runtime environment changed after preflight for {dataset}."
                    )
                adapter = SourceAfisAdapter(client, health=health)
                results.append(
                    _score_condition(
                        project_root=project_root,
                        output_root=output_root,
                        method="sourceafis",
                        dataset=dataset,
                        split=split,
                        adapter=adapter,
                        startup=_startup_dict(sidecar),
                        bundles=bundles,
                        preflight=preflight,
                        runtime_all=runtime,
                    )
                )
            finally:
                client.close()
    summary = {
        "schema_version": "shared-accuracy-split-score-summary-v1",
        "split": split,
        "conditions": results,
        "thresholding_during_scoring": False,
        "completed": True,
    }
    _publish_immutable_json(output_root / f"scores/{split}_summary.json", summary)
    return summary


def _score_file(output_root: Path, method: str, dataset: str, split: str, pair_label: str) -> Path:
    return output_root / f"scores/{method}/{dataset}/{split}/{pair_label}.csv"


def _validate_split_score_summary(output_root: Path, split: str) -> dict[str, Any]:
    summary = _read_json(output_root / f"scores/{split}_summary.json")
    if summary.get("split") != split or summary.get("completed") is not True:
        raise SharedAccuracyError(f"{split} score summary is incomplete.")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for condition in summary.get("conditions", []):
        key = (str(condition.get("method")), str(condition.get("dataset")))
        if key in indexed:
            raise SharedAccuracyError(f"Duplicate score condition in {split} summary: {key}")
        indexed[key] = condition
    expected = {(method, dataset) for method in METHODS for dataset in DATASETS}
    if set(indexed) != expected:
        raise SharedAccuracyError(f"Missing score conditions in {split} summary.")
    for method, dataset in sorted(expected):
        current = _read_json(output_root / f"scores/{method}/{dataset}/{split}/run_metadata.json")
        if current != indexed[(method, dataset)]:
            raise SharedAccuracyError(
                f"Score metadata changed after {split} summary: {method}/{dataset}."
            )
    return summary


def _load_score_rows(
    output_root: Path, method: str, dataset: str, split: str, pair_label: str
) -> list[dict[str, str]]:
    path = _score_file(output_root, method, dataset, split, pair_label)
    metadata_path = path.parent / "run_metadata.json"
    metadata = _read_json(metadata_path)
    if (
        metadata.get("method") != method
        or metadata.get("dataset") != dataset
        or metadata.get("split") != split
        or metadata.get("thresholding_during_scoring") is not False
    ):
        raise SharedAccuracyError(f"Score-run metadata identity mismatch: {metadata_path}")
    expected_hash = metadata.get("score_files", {}).get(pair_label, {}).get("sha256")
    if expected_hash != file_sha256(path):
        raise SharedAccuracyError(f"Score file does not match run metadata: {path}")
    rows = _read_csv(path, SCORE_COLUMNS)
    if not rows:
        raise SharedAccuracyError(f"Score file is empty: {path}")
    if any(
        row["method"] != method
        or row["dataset"] != dataset
        or row["split"] != split
        or row["pair_label"] != pair_label
        for row in rows
    ):
        raise SharedAccuracyError(f"Score file contains mixed identity fields: {path}")
    return rows


def _observations(rows: Sequence[Mapping[str, str]]) -> list[float | int | None]:
    output: list[float | int | None] = []
    integer_method = bool(rows) and rows[0]["method"] == "sift_geometric"
    for row in rows:
        if row["status"] != OK:
            output.append(None)
            continue
        try:
            value = float(row["raw_score"])
        except ValueError as exc:
            raise SharedAccuracyError(f"Invalid raw score for {row['accuracy_pair_id']}.") from exc
        if not math.isfinite(value):
            raise SharedAccuracyError(f"Non-finite raw score for {row['accuracy_pair_id']}.")
        if integer_method:
            if not value.is_integer():
                raise SharedAccuracyError("SIFT raw score is not integral under the frozen definition.")
            output.append(int(value))
        else:
            output.append(value)
    return output


def _target_key(target: float) -> str:
    if target == 0.01:
        return "far_1_percent"
    if target == 0.001:
        return "far_0_1_percent"
    return f"far_{target:.12g}"


def _confidence_dict(value: Any) -> dict[str, Any] | None:
    return asdict(value) if value is not None else None


def _operating_dict(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    for name in ("tar_wilson_95", "fnmr_wilson_95", "far_wilson_95"):
        payload[name] = _confidence_dict(getattr(value, name))
    return payload


def calibrate_thresholds(
    *, project_root: Path = DEFAULT_PROJECT_ROOT, output_root: Path | None = None
) -> dict[str, Any]:
    """Freeze one development-only threshold per method and requested FAR."""

    project_root = project_root.resolve()
    output_root = (output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    _validate_definition_runner(output_root)
    development_summary = _validate_split_score_summary(output_root, "development")
    preflight = _read_json(output_root / "provenance/genuine_preflight.json")
    if preflight.get("passed") is not True:
        raise SharedAccuracyError("Genuine preflight is not passed.")
    if preflight.get("protocol_definition_sha256") != file_sha256(
        output_root / "protocol_definition.json"
    ):
        raise SharedAccuracyError("Genuine preflight belongs to a different protocol definition.")

    inputs: dict[str, dict[str, str]] = {}
    calibrations: dict[str, Any] = {}
    for method in METHODS:
        integer_scores = method == "sift_geometric"
        impostor_rows = {
            dataset: _load_score_rows(
                output_root, method, dataset, "development", "impostor"
            )
            for dataset in DATASETS
        }
        genuine_rows = {
            dataset: _load_score_rows(
                output_root, method, dataset, "development", "genuine"
            )
            for dataset in DATASETS
        }
        observations = {dataset: _observations(rows) for dataset, rows in impostor_rows.items()}
        inputs[method] = {
            f"{dataset}_impostor_sha256": file_sha256(
                _score_file(output_root, method, dataset, "development", "impostor")
            )
            for dataset in DATASETS
        }
        inputs[method].update(
            {
                f"{dataset}_genuine_sha256": file_sha256(
                    _score_file(output_root, method, dataset, "development", "genuine")
                )
                for dataset in DATASETS
            }
        )
        method_calibrations: dict[str, Any] = {}
        for target in TARGET_FARS:
            for dataset in DATASETS:
                scored = sum(value is not None for value in observations[dataset])
                # At least ten expected target-FAR events is a transparent
                # minimum-resolution gate.  Both requested targets pass with
                # the frozen 10-per-identity policy.
                if scored * target < 10:
                    raise SharedAccuracyError(
                        f"Insufficient development impostors for {method}/{dataset} at FAR={target}: "
                        f"n={scored}, n*target={scored * target:.3f}."
                    )
            calibrated = calibrate_common_threshold(
                observations,
                target,
                integer_scores=integer_scores,
            )
            per_dataset: dict[str, Any] = {}
            for dataset in DATASETS:
                development_metrics = compute_operating_metrics(
                    _observations(genuine_rows[dataset]),
                    observations[dataset],
                    calibrated.threshold,
                    integer_scores=integer_scores,
                )
                calibration_metrics = asdict(calibrated.per_dataset[dataset])
                if calibration_metrics["far"] > target:
                    raise SharedAccuracyError(
                        f"Calibrated threshold misses target in {method}/{dataset}/{target}."
                    )
                per_dataset[dataset] = {
                    "calibration_impostors": calibration_metrics,
                    "development_operating_metrics": _operating_dict(development_metrics),
                }
            method_calibrations[_target_key(target)] = {
                "target_far": target,
                "threshold": calibrated.threshold,
                "integer_score_threshold": integer_scores,
                "score_direction": HIGHER_IS_MORE_SIMILAR,
                "acceptance_operator": "raw_score >= threshold",
                "tie_policy": "all scores exactly at threshold are accepted",
                "candidate_count": calibrated.candidate_count,
                "per_dataset": per_dataset,
            }
        calibrations[method] = method_calibrations

    thresholds = {
        "schema_version": "shared-accuracy-frozen-thresholds-v1",
        "protocol_version": PROTOCOL_VERSION,
        "selected_on_split": "development",
        "evaluation_scores_read_during_selection": False,
        "target_fars": list(TARGET_FARS),
        "score_direction": HIGHER_IS_MORE_SIMILAR,
        "input_development_score_sha256": inputs,
        "methods": calibrations,
        "frozen": True,
    }
    threshold_path = output_root / "calibration/frozen_thresholds.json"
    _publish_immutable_json(threshold_path, thresholds)
    gate = {
        "schema_version": "shared-accuracy-evaluation-safety-gate-v1",
        "passed": True,
        "thresholds_path": str(threshold_path),
        "thresholds_sha256": file_sha256(threshold_path),
        "checks": {
            "shared_split_valid": True,
            "no_subject_leakage": True,
            "logical_impostors_valid": True,
            "no_genuine_contamination": True,
            "method_provenance_frozen": True,
            "genuine_preflight_exact": True,
            "development_only_calibration": True,
            "target_far_satisfied_in_both_development_datasets": True,
            "development_impostor_sample_size_sufficient": True,
            "evaluation_scores_absent_or_unread": True,
        },
    }
    _publish_immutable_json(output_root / "calibration/evaluation_safety_gate.json", gate)
    return thresholds


def _score_distribution(observations: Sequence[float | int | None]) -> dict[str, Any]:
    scores = sorted(float(value) for value in observations if value is not None)
    failures = len(observations) - len(scores)
    if not scores:
        return {"scored_count": 0, "failure_count": failures}

    def quantile(probability: float) -> float:
        if len(scores) == 1:
            return scores[0]
        position = probability * (len(scores) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return scores[lower]
        fraction = position - lower
        return scores[lower] * (1 - fraction) + scores[upper] * fraction

    return {
        "scored_count": len(scores),
        "failure_count": failures,
        "minimum": scores[0],
        "p05": quantile(0.05),
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "maximum": scores[-1],
        "mean": sum(scores) / len(scores),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator_left = sum((a - mean_left) ** 2 for a in left)
    denominator_right = sum((b - mean_right) ** 2 for b in right)
    denominator = math.sqrt(denominator_left * denominator_right)
    return numerator / denominator if denominator else None


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[ordered[offset]] = average
        start = end
    return ranks


def _paired_resolution(
    left_rows: Sequence[Mapping[str, str]],
    right_rows: Sequence[Mapping[str, str]],
    *,
    threshold: float | int,
) -> dict[str, Any]:
    left = {row["accuracy_pair_id"]: row for row in left_rows}
    right = {row["accuracy_pair_id"]: row for row in right_rows}
    if list(left) != list(right):
        raise SharedAccuracyError("B/C paired-resolution score rows are not exactly aligned.")
    accepted_both = rejected_both = accepted_only_left = accepted_only_right = 0
    scores_left: list[float] = []
    scores_right: list[float] = []
    deltas: list[float] = []
    failures = 0
    by_finger: dict[int, list[tuple[bool, bool]]] = defaultdict(list)
    for pair_id in left:
        row_left, row_right = left[pair_id], right[pair_id]
        if row_left["status"] != OK or row_right["status"] != OK:
            failures += 1
            continue
        score_left = float(row_left["raw_score"])
        score_right = float(row_right["raw_score"])
        decision_left = score_left >= threshold
        decision_right = score_right >= threshold
        if decision_left and decision_right:
            accepted_both += 1
        elif not decision_left and not decision_right:
            rejected_both += 1
        elif decision_left:
            accepted_only_left += 1
        else:
            accepted_only_right += 1
        scores_left.append(score_left)
        scores_right.append(score_right)
        deltas.append(score_left - score_right)
        by_finger[int(row_left["canonical_finger_position"])].append(
            (decision_left, decision_right)
        )
    scored = len(scores_left)
    return {
        "planned_pair_count": len(left),
        "paired_scored_count": scored,
        "paired_failure_count": failures,
        "accepted_in_both": accepted_both,
        "rejected_in_both": rejected_both,
        "accepted_only_in_sd300b": accepted_only_left,
        "accepted_only_in_sd300c": accepted_only_right,
        "decision_agreement": (accepted_both + rejected_both) / scored if scored else None,
        "mcnemar_discordant_b_only": accepted_only_left,
        "mcnemar_discordant_c_only": accepted_only_right,
        "paired_score_delta_b_minus_c": _score_distribution(deltas),
        "pearson_score_correlation": _pearson(scores_left, scores_right),
        "spearman_score_correlation": _pearson(
            _average_ranks(scores_left), _average_ranks(scores_right)
        ),
        "per_finger": {
            str(finger): {
                "count": len(decisions),
                "sd300b_acceptance_rate": sum(left for left, _ in decisions) / len(decisions),
                "sd300c_acceptance_rate": sum(right for _, right in decisions) / len(decisions),
            }
            for finger, decisions in sorted(by_finger.items())
        },
    }


def _fusion_feasibility(output_root: Path, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Development-only OR/AND feasibility; never used for final method ranking."""

    output: dict[str, Any] = {
        "schema_version": "shared-accuracy-fusion-feasibility-v1",
        "split": "development",
        "exploratory_only": True,
        "evaluation_tuning_performed": False,
        "rules": {},
    }
    for target in TARGET_FARS:
        target_key = _target_key(target)
        source_threshold = thresholds["methods"]["sourceafis"][target_key]["threshold"]
        sift_threshold = thresholds["methods"]["sift_geometric"][target_key]["threshold"]
        target_report: dict[str, Any] = {}
        for dataset in DATASETS:
            dataset_report: dict[str, Any] = {}
            for pair_label in ("genuine", "impostor"):
                source_rows = _load_score_rows(
                    output_root, "sourceafis", dataset, "development", pair_label
                )
                sift_rows = _load_score_rows(
                    output_root, "sift_geometric", dataset, "development", pair_label
                )
                if [row["accuracy_pair_id"] for row in source_rows] != [
                    row["accuracy_pair_id"] for row in sift_rows
                ]:
                    raise SharedAccuracyError("Method score rows are not aligned for fusion feasibility.")
                decisions: dict[str, list[bool]] = {"or": [], "and": []}
                failure_count = 0
                for source, sift in zip(source_rows, sift_rows, strict=True):
                    if source["status"] != OK or sift["status"] != OK:
                        failure_count += 1
                        continue
                    source_accept = float(source["raw_score"]) >= source_threshold
                    sift_accept = float(sift["raw_score"]) >= sift_threshold
                    decisions["or"].append(source_accept or sift_accept)
                    decisions["and"].append(source_accept and sift_accept)
                dataset_report[pair_label] = {
                    rule: {
                        "scored_count": len(values),
                        "failure_count": failure_count,
                        "accepted_count": sum(values),
                        "acceptance_rate": sum(values) / len(values) if values else None,
                    }
                    for rule, values in decisions.items()
                }
            target_report[dataset] = dataset_report
        output["rules"][target_key] = target_report
    return output


def evaluate_and_report(
    *, project_root: Path = DEFAULT_PROJECT_ROOT, output_root: Path | None = None
) -> dict[str, Any]:
    """Evaluate frozen thresholds, curves, legacy points, and paired B/C effects."""

    project_root = project_root.resolve()
    output_root = (output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    _validate_definition_runner(output_root)
    evaluation_summary = _validate_split_score_summary(output_root, "evaluation")
    thresholds_path = output_root / "calibration/frozen_thresholds.json"
    thresholds = _read_json(thresholds_path)
    gate = _read_json(output_root / "calibration/evaluation_safety_gate.json")
    if gate.get("thresholds_sha256") != file_sha256(thresholds_path) or gate.get("passed") is not True:
        raise SharedAccuracyError("Frozen threshold safety gate is invalid.")

    calibrated: dict[str, Any] = {}
    legacy: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    resolution: dict[str, Any] = {}
    central_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    for method in METHODS:
        integer_scores = method == "sift_geometric"
        calibrated[method] = {}
        legacy[method] = {}
        curves[method] = {}
        resolution[method] = {}
        score_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
        score_values: dict[tuple[str, str], list[float | int | None]] = {}
        for dataset in DATASETS:
            for pair_label in ("genuine", "impostor"):
                rows = _load_score_rows(output_root, method, dataset, "evaluation", pair_label)
                score_rows[(dataset, pair_label)] = rows
                score_values[(dataset, pair_label)] = _observations(rows)

            points = roc_det_points(
                score_values[(dataset, "genuine")],
                score_values[(dataset, "impostor")],
                integer_scores=integer_scores,
            )
            eer = discrete_eer(points)
            auc = trapezoidal_auc(points)
            curve_rows = [asdict(point) for point in points]
            roc_path = output_root / f"curves/{method}_{dataset}_roc.csv"
            det_path = output_root / f"curves/{method}_{dataset}_det.csv"
            curve_columns = list(curve_rows[0])
            _publish_immutable_csv(roc_path, curve_rows, curve_columns)
            _publish_immutable_csv(det_path, curve_rows, curve_columns)
            curves[method][dataset] = {
                "roc_csv": str(roc_path),
                "roc_sha256": file_sha256(roc_path),
                "det_csv": str(det_path),
                "det_sha256": file_sha256(det_path),
                "point_count": len(points),
                "auc": auc,
                "eer": asdict(eer),
                "smoothing": False,
            }

            legacy_metrics = compute_operating_metrics(
                score_values[(dataset, "genuine")],
                score_values[(dataset, "impostor")],
                LEGACY_THRESHOLDS[method],
                integer_scores=integer_scores,
            )
            legacy[method][dataset] = {
                "threshold": LEGACY_THRESHOLDS[method],
                "metrics": _operating_dict(legacy_metrics),
                "genuine_distribution": _score_distribution(score_values[(dataset, "genuine")]),
                "impostor_distribution": _score_distribution(score_values[(dataset, "impostor")]),
            }
            legacy_rows.append(
                {
                    "method": method,
                    "operating_point": "legacy",
                    "dataset": dataset,
                    "target_far": "",
                    "threshold": LEGACY_THRESHOLDS[method],
                    "tar": legacy_metrics.tar,
                    "fnmr": legacy_metrics.fnmr,
                    "far": legacy_metrics.far,
                    "genuine_failures": legacy_metrics.genuine_failure_count,
                    "impostor_failures": legacy_metrics.impostor_failure_count,
                }
            )

        for target in TARGET_FARS:
            target_key = _target_key(target)
            threshold = thresholds["methods"][method][target_key]["threshold"]
            calibrated[method][target_key] = {}
            for dataset in DATASETS:
                metrics = compute_operating_metrics(
                    score_values[(dataset, "genuine")],
                    score_values[(dataset, "impostor")],
                    threshold,
                    integer_scores=integer_scores,
                )
                calibrated[method][target_key][dataset] = {
                    "target_far": target,
                    "threshold": threshold,
                    "metrics": _operating_dict(metrics),
                    "genuine_distribution": _score_distribution(
                        score_values[(dataset, "genuine")]
                    ),
                    "impostor_distribution": _score_distribution(
                        score_values[(dataset, "impostor")]
                    ),
                }
                legacy_rows.append(
                    {
                        "method": method,
                        "operating_point": target_key,
                        "dataset": dataset,
                        "target_far": target,
                        "threshold": threshold,
                        "tar": metrics.tar,
                        "fnmr": metrics.fnmr,
                        "far": metrics.far,
                        "genuine_failures": metrics.genuine_failure_count,
                        "impostor_failures": metrics.impostor_failure_count,
                    }
                )
            resolution[method][target_key] = {
                "target_far": target,
                "threshold": threshold,
                "genuine": _paired_resolution(
                    score_rows[("sd300b", "genuine")],
                    score_rows[("sd300c", "genuine")],
                    threshold=threshold,
                ),
                "impostor": _paired_resolution(
                    score_rows[("sd300b", "impostor")],
                    score_rows[("sd300c", "impostor")],
                    threshold=threshold,
                ),
                "paired_conditions_not_independent": True,
            }

            b = calibrated[method][target_key]["sd300b"]["metrics"]
            c = calibrated[method][target_key]["sd300c"]["metrics"]
            central_rows.append(
                {
                    "method": method,
                    "target_far": target,
                    "threshold": threshold,
                    "sd300b_tar": b["tar"],
                    "sd300b_far": b["far"],
                    "sd300c_tar": c["tar"],
                    "sd300c_far": c["far"],
                    "sd300b_fnmr": b["fnmr"],
                    "sd300c_fnmr": c["fnmr"],
                    "sd300b_genuine_failures": b["genuine_failure_count"],
                    "sd300b_impostor_failures": b["impostor_failure_count"],
                    "sd300c_genuine_failures": c["genuine_failure_count"],
                    "sd300c_impostor_failures": c["impostor_failure_count"],
                }
            )

    central_path = output_root / "reports/primary_scientific_comparison.csv"
    _publish_immutable_csv(central_path, central_rows, list(central_rows[0]))
    legacy_path = output_root / "reports/legacy_vs_calibrated.csv"
    _publish_immutable_csv(legacy_path, legacy_rows, list(legacy_rows[0]))
    fusion = _fusion_feasibility(output_root, thresholds)
    _publish_immutable_json(output_root / "reports/fusion_feasibility_development_only.json", fusion)

    rankings: dict[str, Any] = {}
    for target in TARGET_FARS:
        key = _target_key(target)
        source = next(row for row in central_rows if row["method"] == "sourceafis" and row["target_far"] == target)
        sift = next(row for row in central_rows if row["method"] == "sift_geometric" and row["target_far"] == target)
        comparisons = {
            dataset: source[f"{dataset}_tar"] - sift[f"{dataset}_tar"]
            for dataset in DATASETS
        }
        if all(delta >= 0 for delta in comparisons.values()) and any(
            delta > 0 for delta in comparisons.values()
        ):
            winner = "sourceafis"
        elif all(delta <= 0 for delta in comparisons.values()) and any(
            delta < 0 for delta in comparisons.values()
        ):
            winner = "sift_geometric"
        else:
            winner = "mixed_or_tied"
        rankings[key] = {
            "winner_by_tar_in_both_paired_resolution_conditions": winner,
            "tar_difference_sourceafis_minus_sift": comparisons,
            "valid_same_far_comparison": True,
            "interpretation": "descriptive paired-condition comparison; B and C are not independent samples",
        }

    report = {
        "schema_version": "shared-accuracy-evaluation-report-v1",
        "protocol_version": PROTOCOL_VERSION,
        "thresholds_sha256": file_sha256(thresholds_path),
        "primary_outcome": "TAR at a method-specific threshold calibrated to the same target FAR",
        "calibrated_evaluation": calibrated,
        "legacy_evaluation": legacy,
        "curves": curves,
        "paired_resolution_analysis": resolution,
        "ranking": rankings,
        "far_0_01_percent_claimed": False,
        "plots_created": False,
        "fusion": {
            "performed": True,
            "scope": "development-only exploratory feasibility",
            "used_for_ranking": False,
        },
        "primary_table": {
            "path": str(central_path),
            "sha256": file_sha256(central_path),
        },
        "legacy_comparison_table": {
            "path": str(legacy_path),
            "sha256": file_sha256(legacy_path),
        },
    }
    _publish_immutable_json(output_root / "reports/evaluation_metrics.json", report)
    _publish_immutable_json(output_root / "reports/resolution_analysis.json", resolution)
    _publish_immutable_json(output_root / "reports/legacy_comparison.json", legacy)
    return report


def protect_before(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    output_root = (output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    return capture_protected_before(
        project_root,
        data_root,
        output_root / "integrity/protected_before.jsonl",
        shared_results_root=project_root / "results/shared_accuracy",
    )


def record_pytest_result(
    *,
    output_root: Path,
    command: str,
    passed: bool,
    passed_count: int,
    failed_count: int,
    skipped_count: int = 0,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    if passed != (failed_count == 0):
        raise SharedAccuracyError("Pytest pass flag/counts are inconsistent.")
    report = {
        "schema_version": "shared-accuracy-pytest-result-v1",
        "command": command,
        "passed": passed,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "duration_seconds": duration_seconds,
    }
    _publish_immutable_json(output_root.resolve() / "provenance/pytest.json", report)
    return report


def _read_table(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise SharedAccuracyError(f"Cannot read report table {path}: {exc}") from exc


def _supervisor_markdown(
    *,
    definition: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    pytest_report: Mapping[str, Any],
    integrity: Mapping[str, Any],
    central_rows: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# Shared SourceAFIS/SIFT biometric accuracy — supervisor summary",
        "",
        "This benchmark is separate from the existing cold-pair latency benchmark. Its primary outcome is "
        "TAR at a common target FAR using the same subjects, genuine pairs, impostor identity relations, and "
        "development/evaluation split.",
        "",
        "## Cohort and pair counts",
        "",
        "| Split | Subjects | Genuine per dataset | Impostors per dataset |",
        "|---|---:|---:|---:|",
    ]
    split_ref = _read_json(Path(definition["output_root"]) / "shared_split_reference.json") if "output_root" in definition else None
    subject_counts = (
        {
            "development": split_ref["development_count"],
            "evaluation": split_ref["evaluation_count"],
        }
        if split_ref
        else {"development": 192, "evaluation": 696}
    )
    for split in SPLITS:
        lines.append(
            f"| {split} | {subject_counts[split]} | "
            f"{definition['genuine_counts'][f'sd300b/{split}']} | "
            f"{definition['impostor_summaries'][split]['impostor_pair_count']} |"
        )
    lines.extend(
        [
            "",
            "## Primary scientific comparison",
            "",
            "| Method | Target FAR | Threshold | B TAR | B FAR | C TAR | C FAR | B FNMR | C FNMR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in central_rows:
        lines.append(
            "| {method} | {target_far:.3%} | {threshold} | {b_tar:.3%} | {b_far:.3%} | "
            "{c_tar:.3%} | {c_far:.3%} | {b_fnmr:.3%} | {c_fnmr:.3%} |".format(
                method=row["method"],
                target_far=float(row["target_far"]),
                threshold=row["threshold"],
                b_tar=float(row["sd300b_tar"]),
                b_far=float(row["sd300b_far"]),
                c_tar=float(row["sd300c_tar"]),
                c_far=float(row["sd300c_far"]),
                b_fnmr=float(row["sd300b_fnmr"]),
                c_fnmr=float(row["sd300c_fnmr"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation and safeguards",
            "",
            f"- Frozen split: `{SHARED_SPLIT_VERSION}` referencing the existing deterministic SIFT split.",
            f"- Impostor policy: {IMPOSTORS_PER_IDENTITY} same-finger, different-subject roll identities per plain identity.",
            "- Thresholds were selected on development scores only and then frozen before evaluation scoring.",
            "- B and C are paired resolution conditions, not independent samples.",
            "- Legacy operating points are reported separately from shared-FAR calibrated points.",
            "- FAR 0.01% is not claimed; the prespecified primary/secondary targets are 1% and 0.1%.",
            "- Prepared-mode wall runtime is not a cold-pair or production latency measurement.",
            f"- Pytest: {pytest_report['passed_count']} passed, {pytest_report['failed_count']} failed.",
            f"- Protected inputs unchanged: {integrity['protected_artifacts_unchanged']}.",
            "",
            "## Ranking validity",
            "",
            "The protocol now supports a valid like-for-like comparison at the same target FAR. Any method "
            "ranking is conditional on the two paired NIST resolution conditions and the reported confidence "
            "intervals; per-method derived-cohort acceptance and unmatched legacy thresholds are not used for ranking.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_study(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    output_root = (output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    pytest_report = _read_json(output_root / "provenance/pytest.json")
    if pytest_report.get("passed") is not True:
        raise SharedAccuracyError("Full pytest result is missing or failed.")
    evaluation = _read_json(output_root / "reports/evaluation_metrics.json")
    definition = _validate_definition_runner(output_root)
    thresholds = _read_json(output_root / "calibration/frozen_thresholds.json")
    integrity = verify_protected_after(
        project_root,
        data_root,
        output_root / "integrity/protected_before.jsonl",
        shared_results_root=project_root / "results/shared_accuracy",
        after_snapshot_path=output_root / "integrity/protected_after.jsonl",
        report_path=output_root / "integrity/protected_artifact_integrity.json",
        raise_on_mismatch=True,
    )
    central_path = output_root / "reports/primary_scientific_comparison.csv"
    central_rows = _read_table(central_path)
    development_score = _read_json(output_root / "scores/development_summary.json")
    evaluation_score = _read_json(output_root / "scores/evaluation_summary.json")
    runtime_rows: list[dict[str, Any]] = []
    for split, summary in (("development", development_score), ("evaluation", evaluation_score)):
        for condition in summary["conditions"]:
            runtime_rows.append(
                {
                    "method": condition["method"],
                    "dataset": condition["dataset"],
                    "split": split,
                    "wall_runtime_seconds": condition["cache"]["wall_runtime_seconds"],
                    "unique_prepare_count": condition["cache"]["unique_image_count"],
                    "comparison_count": condition["cache"]["comparison_count"],
                    "prepare_failure_count": condition["cache"]["prepare_failure_count"],
                    "comparison_failure_count": condition["cache"]["comparison_failure_count"],
                }
            )
    _publish_immutable_csv(
        output_root / "reports/runtime_summary.csv", runtime_rows, list(runtime_rows[0])
    )
    summary = {
        "schema_version": "shared-accuracy-supervisor-summary-v1",
        "protocol_version": PROTOCOL_VERSION,
        "subject_split": {
            "version": SHARED_SPLIT_VERSION,
            "development_subjects": 192,
            "evaluation_subjects": 696,
        },
        "pair_counts": {
            "genuine": definition["genuine_counts"],
            "impostor": {
                split: definition["impostor_summaries"][split]["impostor_pair_count"]
                for split in SPLITS
            },
        },
        "impostor_generation_rule": definition["impostor_policy"],
        "representation_cache_policy": definition["accuracy_mode"],
        "genuine_preflight": _read_json(output_root / "provenance/genuine_preflight.json"),
        "calibrated_thresholds": thresholds["methods"],
        "development_metrics": {
            method: {
                key: value["per_dataset"]
                for key, value in thresholds["methods"][method].items()
            }
            for method in METHODS
        },
        "evaluation_metrics": evaluation["calibrated_evaluation"],
        "roc_det_eer_auc": evaluation["curves"],
        "legacy_vs_calibrated": str(output_root / "reports/legacy_vs_calibrated.csv"),
        "paired_resolution_analysis": evaluation["paired_resolution_analysis"],
        "fusion_feasibility": str(
            output_root / "reports/fusion_feasibility_development_only.json"
        ),
        "pytest": pytest_report,
        "runtime_summary": {
            "path": str(output_root / "reports/runtime_summary.csv"),
            "rows": runtime_rows,
            "not_cold_pair_latency": True,
        },
        "protected_artifacts": integrity,
        "valid_same_far_method_comparison": True,
        "ranking": evaluation["ranking"],
        "existing_artifacts_overwritten": False,
    }
    _publish_immutable_json(output_root / "supervisor_summary.json", summary)
    markdown = _supervisor_markdown(
        definition={**definition, "output_root": str(output_root)},
        thresholds=thresholds,
        evaluation=evaluation,
        pytest_report=pytest_report,
        integrity=integrity,
        central_rows=central_rows,
    )
    _publish_immutable_bytes(output_root / "supervisor_summary.md", markdown.encode("utf-8"))
    artifact_manifest = write_artifact_manifest(output_root, namespace="sourceafis_sift_v1")
    result = {
        **summary,
        "artifact_manifest": {
            "path": str(output_root / "artifact_manifest.json"),
            "sha256": file_sha256(output_root / "artifact_manifest.json"),
            "file_count": artifact_manifest["file_count"],
            "tree_sha256": artifact_manifest["tree_sha256"],
        },
        "supervisor_summary_path": str(output_root / "supervisor_summary.json"),
        "supervisor_summary_sha256": file_sha256(output_root / "supervisor_summary.json"),
    }
    return result


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared SourceAFIS/SIFT biometric accuracy protocol")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "protect-before",
        "prepare",
        "preflight",
        "score-development",
        "calibrate",
        "score-evaluation",
        "report",
        "record-tests",
        "finalize",
    ):
        child = commands.add_parser(name)
        _common_paths(child)
        if name in ("preflight", "score-development", "score-evaluation"):
            child.add_argument("--sidecar-jar", type=Path)
            child.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
            child.add_argument("--timeout-seconds", type=float, default=120.0)
        if name == "record-tests":
            child.add_argument("--test-command", required=True)
            child.add_argument("--passed-count", type=int, required=True)
            child.add_argument("--failed-count", type=int, required=True)
            child.add_argument("--skipped-count", type=int, default=0)
            child.add_argument("--duration-seconds", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    output_root = (args.output_root or project_root / DEFAULT_OUTPUT_ROOT).resolve()
    try:
        if args.command == "protect-before":
            result = protect_before(
                project_root=project_root, data_root=data_root, output_root=output_root
            )
        elif args.command == "prepare":
            result = prepare_protocol(
                project_root=project_root, data_root=data_root, output_root=output_root
            )
        elif args.command == "preflight":
            result = run_genuine_preflight(
                project_root=project_root,
                data_root=data_root,
                output_root=output_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "score-development":
            result = score_split(
                "development",
                project_root=project_root,
                data_root=data_root,
                output_root=output_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "calibrate":
            result = calibrate_thresholds(project_root=project_root, output_root=output_root)
        elif args.command == "score-evaluation":
            result = score_split(
                "evaluation",
                project_root=project_root,
                data_root=data_root,
                output_root=output_root,
                sidecar_jar=args.sidecar_jar,
                service_url=args.service_url,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "report":
            result = evaluate_and_report(project_root=project_root, output_root=output_root)
        elif args.command == "record-tests":
            result = record_pytest_result(
                output_root=output_root,
                command=args.test_command,
                passed=args.failed_count == 0,
                passed_count=args.passed_count,
                failed_count=args.failed_count,
                skipped_count=args.skipped_count,
                duration_seconds=args.duration_seconds,
            )
        elif args.command == "finalize":
            result = finalize_study(
                project_root=project_root, data_root=data_root, output_root=output_root
            )
        else:
            raise AssertionError(args.command)
    except (OSError, SharedAccuracyError, ValueError) as exc:
        print(f"shared-accuracy error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
