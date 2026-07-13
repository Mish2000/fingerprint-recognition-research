"""Context-aware implementation of the plain_self protocol."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .canonical_fingers import CanonicalFingerMappingError, canonical_finger_position
from .nist_sd300 import (
    ImageRecord,
    ScanError,
    ScanResult,
    SchemaValidationError,
    scan_dataset,
    validate_image_path,
)
from .protocol_dataset import ProtocolDatasetContext
from .self_manifest_common import (
    ensure_unique_pair_ids,
    ensure_unique_row,
    ensure_unique_value,
    parse_int,
    read_manifest_rows,
    validate_completeness,
    validate_required_values,
    write_manifest_atomic as write_self_manifest_atomic,
)


PROTOCOL = "plain_self"


@dataclass(frozen=True)
class PlainSelfPair:
    pair_id: str
    dataset: str
    protocol: str
    subject_id: str
    canonical_finger_position: int
    ppi: int
    raw_frgp_a: int
    raw_frgp_b: int
    path_a: Path
    path_b: Path

    def as_csv_row(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "dataset": self.dataset,
            "protocol": self.protocol,
            "subject_id": self.subject_id,
            "canonical_finger_position": str(self.canonical_finger_position),
            "ppi": str(self.ppi),
            "raw_frgp_a": str(self.raw_frgp_a),
            "raw_frgp_b": str(self.raw_frgp_b),
            "path_a": str(self.path_a),
            "path_b": str(self.path_b),
        }


@dataclass(frozen=True)
class ManifestValidationReport:
    row_count: int
    expected_identity_count: int
    actual_identity_count: int
    canonical_finger_counts: dict[int, int]


@dataclass(frozen=True)
class _ManifestRowRecord:
    row_number: int
    ppi: int
    raw_frgp: int
    path: Path
    source_record: ImageRecord


def make_pair_id(
    context: ProtocolDatasetContext,
    subject_id: str,
    canonical_finger_position: int,
) -> str:
    return context.pair_id(PROTOCOL, subject_id, canonical_finger_position)


def build_plain_self_pairs(
    context: ProtocolDatasetContext,
    scan_result: ScanResult,
    error_cls: type[Exception],
) -> list[PlainSelfPair]:
    """Build deterministic plain_self pairs from a completed dataset scan."""

    if scan_result.errors:
        raise error_cls(_format_scan_errors(context, scan_result.errors))

    wrong_dataset_records = sorted(
        {record.dataset for record in scan_result.records if record.dataset != context.name}
    )
    if wrong_dataset_records:
        raise error_cls(
            f"Expected only {context.name!r} records, found datasets: "
            f"{', '.join(wrong_dataset_records)}."
        )

    records_by_identity: dict[tuple[str, str, int], list[ImageRecord]] = {}
    for record in scan_result.records:
        if record.impression_type != "plain":
            continue
        if record.ppi != context.expected_ppi:
            raise error_cls(
                f"Expected {context.name!r} PPI {context.expected_ppi}, "
                f"found {record.ppi} for {record.absolute_path}."
            )

        canonical_position = _canonical_position_for_record(record, error_cls)
        if canonical_position is None:
            continue

        key = (record.dataset, record.subject_id, canonical_position)
        records_by_identity.setdefault(key, []).append(record)

    duplicates = {
        key: records
        for key, records in records_by_identity.items()
        if len(records) > 1
    }
    if duplicates:
        raise error_cls(_format_duplicate_identities(duplicates))

    pairs = [
        _pair_from_record(context, records[0], canonical_position=key[2])
        for key, records in records_by_identity.items()
    ]
    pairs.sort(key=lambda pair: (pair.subject_id, pair.canonical_finger_position))
    ensure_unique_pair_ids(pairs, error_cls)
    return pairs


def validate_manifest(
    context: ProtocolDatasetContext,
    manifest_path: Path,
    data_root: Path,
    generation_error_cls: type[Exception],
    validation_error_cls: type[Exception],
) -> ManifestValidationReport:
    rows = read_manifest_rows(manifest_path, validation_error_cls)
    expected_plain_dir = context.spec.impression_dir(data_root, "plain").resolve()
    expected_pairs = _expected_pairs_by_identity(
        context,
        data_root,
        generation_error_cls,
        validation_error_cls,
    )

    seen_pair_ids: set[str] = set()
    seen_rows: set[tuple[str, ...]] = set()
    actual_rows: dict[tuple[str, str, int], _ManifestRowRecord] = {}
    canonical_counts: Counter[int] = Counter()

    for row_number, row in enumerate(rows, start=2):
        validate_required_values(row, row_number, validation_error_cls)
        ensure_unique_row(row, seen_rows, row_number, validation_error_cls)

        pair_id = row["pair_id"]
        ensure_unique_value(pair_id, seen_pair_ids, "pair_id", row_number, validation_error_cls)

        if row["dataset"] != context.name:
            raise validation_error_cls(
                f"Invalid dataset {row['dataset']!r} at CSV line {row_number}; "
                f"expected {context.name!r}."
            )
        if row["protocol"] != PROTOCOL:
            raise validation_error_cls(
                f"Invalid protocol {row['protocol']!r} at CSV line {row_number}; expected {PROTOCOL!r}."
            )
        if row["path_a"] != row["path_b"]:
            raise validation_error_cls(f"path_a and path_b differ at CSV line {row_number}.")
        if row["raw_frgp_a"] != row["raw_frgp_b"]:
            raise validation_error_cls(f"raw_frgp_a and raw_frgp_b differ at CSV line {row_number}.")

        canonical_position = parse_int(row, "canonical_finger_position", row_number, validation_error_cls)
        if canonical_position not in range(1, 11):
            raise validation_error_cls(
                "canonical_finger_position must be in 1-10 at "
                f"CSV line {row_number}; got {canonical_position}."
            )

        ppi = parse_int(row, "ppi", row_number, validation_error_cls)
        if ppi != context.expected_ppi:
            raise validation_error_cls(
                f"PPI mismatch at CSV line {row_number}: manifest has {ppi}, "
                f"{context.name} expects {context.expected_ppi}."
            )
        raw_frgp = parse_int(row, "raw_frgp_a", row_number, validation_error_cls)
        path = Path(row["path_a"])
        if not path.is_file():
            raise validation_error_cls(f"Manifest path does not exist at CSV line {row_number}: {path}")

        _ensure_path_under_plain_dataset(context, path, expected_plain_dir, row_number, validation_error_cls)
        source_record = _validate_source_path(context, path, row_number, validation_error_cls)

        mapped_position = _canonical_position_for_row(raw_frgp, row_number, validation_error_cls)
        if mapped_position != canonical_position:
            raise validation_error_cls(
                "Canonical mapping mismatch at CSV line "
                f"{row_number}: raw FRGP {raw_frgp} maps to {mapped_position}, "
                f"manifest has {canonical_position}."
            )

        expected_pair_id = make_pair_id(context, row["subject_id"], canonical_position)
        if pair_id != expected_pair_id:
            raise validation_error_cls(
                f"Invalid pair_id {pair_id!r} at CSV line {row_number}; "
                f"expected {expected_pair_id!r}."
            )

        identity = (row["dataset"], row["subject_id"], canonical_position)
        if identity in actual_rows:
            raise validation_error_cls(
                "Duplicate anatomical identity in manifest at CSV line "
                f"{row_number}: {identity}."
            )
        actual_rows[identity] = _ManifestRowRecord(
            row_number=row_number,
            ppi=ppi,
            raw_frgp=raw_frgp,
            path=path,
            source_record=source_record,
        )
        canonical_counts[canonical_position] += 1

    validate_completeness(set(expected_pairs), set(actual_rows), validation_error_cls)
    _validate_expected_source_records(context, expected_pairs, actual_rows, validation_error_cls)

    return ManifestValidationReport(
        row_count=len(rows),
        expected_identity_count=len(expected_pairs),
        actual_identity_count=len(actual_rows),
        canonical_finger_counts={position: canonical_counts[position] for position in range(1, 11)},
    )


def write_manifest_atomic(
    pairs: Iterable[PlainSelfPair],
    output_path: Path,
    generation_error_cls: type[Exception],
) -> None:
    """Write the CSV manifest by replacing the target only after a full write succeeds."""

    write_self_manifest_atomic(pairs, output_path, generation_error_cls)


def _pair_from_record(
    context: ProtocolDatasetContext,
    record: ImageRecord,
    canonical_position: int,
) -> PlainSelfPair:
    return PlainSelfPair(
        pair_id=make_pair_id(context, record.subject_id, canonical_position),
        dataset=context.name,
        protocol=PROTOCOL,
        subject_id=record.subject_id,
        canonical_finger_position=canonical_position,
        ppi=record.ppi,
        raw_frgp_a=record.frgp,
        raw_frgp_b=record.frgp,
        path_a=record.absolute_path,
        path_b=record.absolute_path,
    )


def _identity(pair: PlainSelfPair) -> tuple[str, str, int]:
    return (pair.dataset, pair.subject_id, pair.canonical_finger_position)


def _expected_pairs_by_identity(
    context: ProtocolDatasetContext,
    data_root: Path,
    generation_error_cls: type[Exception],
    validation_error_cls: type[Exception],
) -> dict[tuple[str, str, int], PlainSelfPair]:
    scan_result = scan_dataset(data_root, context.spec)
    try:
        pairs = build_plain_self_pairs(context, scan_result, generation_error_cls)
    except generation_error_cls as exc:
        raise validation_error_cls(
            f"Cannot validate {context.name}/{PROTOCOL}: expected {_dataset_label(context)} "
            f"identities are invalid: {exc}"
        ) from exc
    return {_identity(pair): pair for pair in pairs}


def _canonical_position_for_record(record: ImageRecord, error_cls: type[Exception]) -> int | None:
    try:
        return canonical_finger_position(record.impression_type, record.frgp)
    except CanonicalFingerMappingError as exc:
        raise error_cls(
            f"Cannot map {record.impression_type!r} FRGP {record.frgp} for {record.absolute_path}: {exc}"
        ) from exc


def _canonical_position_for_row(
    raw_frgp: int,
    row_number: int,
    validation_error_cls: type[Exception],
) -> int | None:
    try:
        return canonical_finger_position("plain", raw_frgp)
    except CanonicalFingerMappingError as exc:
        raise validation_error_cls(
            f"raw FRGP is not valid for plain at CSV line {row_number}: {exc}"
        ) from exc


def _format_scan_errors(context: ProtocolDatasetContext, errors: list[ScanError]) -> str:
    details = "; ".join(f"{error.path}: {error.message}" for error in errors[:5])
    suffix = "" if len(errors) <= 5 else f"; ... and {len(errors) - 5} more"
    return (
        f"Cannot generate {context.name}/{PROTOCOL}: {_dataset_label(context)} scan has "
        f"{len(errors)} error(s): {details}{suffix}"
    )


def _format_duplicate_identities(
    duplicates: dict[tuple[str, str, int], list[ImageRecord]]
) -> str:
    details: list[str] = []
    for (dataset, subject_id, canonical_position), records in sorted(duplicates.items()):
        paths = ", ".join(str(record.absolute_path) for record in records[:3])
        suffix = "" if len(records) <= 3 else f", ... and {len(records) - 3} more"
        details.append(
            f"{dataset} subject {subject_id} canonical {canonical_position:02d} "
            f"has {len(records)} records: {paths}{suffix}"
        )
    return "Duplicate plain single-finger anatomical identities found: " + "; ".join(details)


def _ensure_path_under_plain_dataset(
    context: ProtocolDatasetContext,
    path: Path,
    expected_plain_dir: Path,
    row_number: int,
    validation_error_cls: type[Exception],
) -> None:
    try:
        path.resolve().relative_to(expected_plain_dir)
    except ValueError as exc:
        raise validation_error_cls(
            f"Path at CSV line {row_number} is not under the {_dataset_label(context)} plain directory: {path}"
        ) from exc


def _validate_source_path(
    context: ProtocolDatasetContext,
    path: Path,
    row_number: int,
    validation_error_cls: type[Exception],
) -> ImageRecord:
    try:
        return validate_image_path(path, context.spec, "plain")
    except SchemaValidationError as exc:
        raise validation_error_cls(
            f"Path does not validate as an {_dataset_label(context)} plain record at CSV line "
            f"{row_number}: {exc}"
        ) from exc


def _validate_expected_source_records(
    context: ProtocolDatasetContext,
    expected_pairs: dict[tuple[str, str, int], PlainSelfPair],
    actual_rows: dict[tuple[str, str, int], _ManifestRowRecord],
    validation_error_cls: type[Exception],
) -> None:
    for identity, expected_pair in sorted(expected_pairs.items()):
        actual = actual_rows[identity]
        source_record = actual.source_record
        expected_path = expected_pair.path_a.resolve()
        actual_path = actual.path.resolve()
        if actual_path != expected_path:
            raise validation_error_cls(
                "Manifest identity points to the wrong source path at CSV line "
                f"{actual.row_number}: {identity} expected {expected_path}, got {actual.path}."
            )
        if source_record.subject_id != expected_pair.subject_id:
            raise validation_error_cls(
                "Subject mismatch at CSV line "
                f"{actual.row_number}: manifest has {expected_pair.subject_id!r}, "
                f"path has {source_record.subject_id!r}."
            )
        if actual.raw_frgp != expected_pair.raw_frgp_a:
            raise validation_error_cls(
                "Manifest identity points to the wrong raw FRGP at CSV line "
                f"{actual.row_number}: {identity} expected {expected_pair.raw_frgp_a}, "
                f"got {actual.raw_frgp}."
            )
        if source_record.frgp != actual.raw_frgp:
            raise validation_error_cls(
                f"FRGP mismatch at CSV line {actual.row_number}: "
                f"manifest has {actual.raw_frgp}, path has {source_record.frgp}."
            )
        if actual.ppi != expected_pair.ppi:
            raise validation_error_cls(
                "Manifest identity points to the wrong PPI at CSV line "
                f"{actual.row_number}: {identity} expected {expected_pair.ppi}, got {actual.ppi}."
            )
        if source_record.ppi != actual.ppi:
            raise validation_error_cls(
                f"PPI mismatch at CSV line {actual.row_number}: "
                f"manifest has {actual.ppi}, path has {source_record.ppi}."
            )
        if actual.ppi != context.expected_ppi:
            raise validation_error_cls(
                f"PPI mismatch at CSV line {actual.row_number}: "
                f"manifest has {actual.ppi}, {context.name} expects {context.expected_ppi}."
            )


def _dataset_label(context: ProtocolDatasetContext) -> str:
    if context.name.startswith("sd"):
        return "SD" + context.name[2:]
    return context.name
