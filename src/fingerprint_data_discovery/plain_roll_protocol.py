"""Context-aware implementation of the plain_roll genuine-pair protocol."""

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


PROTOCOL = "plain_roll"


@dataclass(frozen=True)
class PlainRollPair:
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
    eligible_plain_identity_count: int
    eligible_roll_identity_count: int
    expected_intersection_count: int
    actual_identity_count: int
    plain_only_identity_count: int
    roll_only_identity_count: int
    canonical_finger_counts: dict[int, int]


@dataclass(frozen=True)
class _ManifestRowRecord:
    row_number: int
    ppi: int
    plain_raw_frgp: int
    roll_raw_frgp: int
    plain_path: Path
    roll_path: Path
    plain_record: ImageRecord
    roll_record: ImageRecord


def make_pair_id(
    context: ProtocolDatasetContext,
    subject_id: str,
    canonical_finger_position: int,
) -> str:
    return context.pair_id(PROTOCOL, subject_id, canonical_finger_position)


def build_plain_roll_pairs(
    context: ProtocolDatasetContext,
    scan_result: ScanResult,
    error_cls: type[Exception],
) -> list[PlainRollPair]:
    """Build deterministic genuine plain-roll pairs from a completed dataset scan."""

    plain_by_identity, roll_by_identity = _eligible_records_by_identity(
        context,
        scan_result,
        error_cls,
    )
    pairs = [
        _pair_from_records(
            context,
            plain_by_identity[identity],
            roll_by_identity[identity],
            identity[2],
            error_cls,
        )
        for identity in sorted(set(plain_by_identity) & set(roll_by_identity))
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
    expected_roll_dir = context.spec.impression_dir(data_root, "roll").resolve()
    plain_by_identity, roll_by_identity = _expected_records_by_identity(
        context,
        data_root,
        generation_error_cls,
        validation_error_cls,
    )
    expected_intersection = set(plain_by_identity) & set(roll_by_identity)
    plain_only = set(plain_by_identity) - set(roll_by_identity)
    roll_only = set(roll_by_identity) - set(plain_by_identity)

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
        if row["path_a"] == row["path_b"]:
            raise validation_error_cls(f"path_a and path_b must differ at CSV line {row_number}.")

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
        plain_raw_frgp = parse_int(row, "raw_frgp_a", row_number, validation_error_cls)
        roll_raw_frgp = parse_int(row, "raw_frgp_b", row_number, validation_error_cls)
        plain_path = Path(row["path_a"])
        roll_path = Path(row["path_b"])
        if not plain_path.is_file():
            raise validation_error_cls(f"Manifest path_a does not exist at CSV line {row_number}: {plain_path}")
        if not roll_path.is_file():
            raise validation_error_cls(f"Manifest path_b does not exist at CSV line {row_number}: {roll_path}")

        _ensure_path_under_plain_dataset(context, plain_path, expected_plain_dir, row_number, validation_error_cls)
        _ensure_path_under_roll_dataset(context, roll_path, expected_roll_dir, row_number, validation_error_cls)
        plain_record = _validate_source_path(context, plain_path, "plain", row_number, validation_error_cls)
        roll_record = _validate_source_path(context, roll_path, "roll", row_number, validation_error_cls)

        plain_mapped_position = _canonical_position_for_row("plain", plain_raw_frgp, row_number, validation_error_cls)
        roll_mapped_position = _canonical_position_for_row("roll", roll_raw_frgp, row_number, validation_error_cls)
        if plain_mapped_position != canonical_position:
            raise validation_error_cls(
                "Canonical mapping mismatch for path_a at CSV line "
                f"{row_number}: raw FRGP {plain_raw_frgp} maps to {plain_mapped_position}, "
                f"manifest has {canonical_position}."
            )
        if roll_mapped_position != canonical_position:
            raise validation_error_cls(
                "Canonical mapping mismatch for path_b at CSV line "
                f"{row_number}: raw FRGP {roll_raw_frgp} maps to {roll_mapped_position}, "
                f"manifest has {canonical_position}."
            )

        _validate_row_matches_records(
            context,
            row,
            plain_record,
            roll_record,
            ppi,
            plain_raw_frgp,
            roll_raw_frgp,
            row_number,
            validation_error_cls,
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
            plain_raw_frgp=plain_raw_frgp,
            roll_raw_frgp=roll_raw_frgp,
            plain_path=plain_path,
            roll_path=roll_path,
            plain_record=plain_record,
            roll_record=roll_record,
        )
        canonical_counts[canonical_position] += 1

    validate_completeness(expected_intersection, set(actual_rows), validation_error_cls)
    _validate_expected_source_records(context, plain_by_identity, roll_by_identity, actual_rows, validation_error_cls)

    return ManifestValidationReport(
        row_count=len(rows),
        eligible_plain_identity_count=len(plain_by_identity),
        eligible_roll_identity_count=len(roll_by_identity),
        expected_intersection_count=len(expected_intersection),
        actual_identity_count=len(actual_rows),
        plain_only_identity_count=len(plain_only),
        roll_only_identity_count=len(roll_only),
        canonical_finger_counts={position: canonical_counts[position] for position in range(1, 11)},
    )


def write_manifest_atomic(
    pairs: Iterable[PlainRollPair],
    output_path: Path,
    generation_error_cls: type[Exception],
) -> None:
    """Write the CSV manifest by replacing the target only after a full write succeeds."""

    write_self_manifest_atomic(pairs, output_path, generation_error_cls)


def _pair_from_records(
    context: ProtocolDatasetContext,
    plain_record: ImageRecord,
    roll_record: ImageRecord,
    canonical_position: int,
    error_cls: type[Exception],
) -> PlainRollPair:
    if plain_record.ppi != roll_record.ppi:
        raise error_cls(
            "Cannot pair records with different PPI values: "
            f"{plain_record.absolute_path} has {plain_record.ppi}, "
            f"{roll_record.absolute_path} has {roll_record.ppi}."
        )
    if plain_record.ppi != context.expected_ppi:
        raise error_cls(
            f"Cannot pair records with unexpected PPI {plain_record.ppi}; "
            f"{context.name} expects {context.expected_ppi}."
        )

    return PlainRollPair(
        pair_id=make_pair_id(context, plain_record.subject_id, canonical_position),
        dataset=context.name,
        protocol=PROTOCOL,
        subject_id=plain_record.subject_id,
        canonical_finger_position=canonical_position,
        ppi=plain_record.ppi,
        raw_frgp_a=plain_record.frgp,
        raw_frgp_b=roll_record.frgp,
        path_a=plain_record.absolute_path,
        path_b=roll_record.absolute_path,
    )


def _eligible_records_by_identity(
    context: ProtocolDatasetContext,
    scan_result: ScanResult,
    error_cls: type[Exception],
) -> tuple[dict[tuple[str, str, int], ImageRecord], dict[tuple[str, str, int], ImageRecord]]:
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

    plain_records = _records_by_identity(context, scan_result.records, "plain", error_cls)
    roll_records = _records_by_identity(context, scan_result.records, "roll", error_cls)
    return plain_records, roll_records


def _records_by_identity(
    context: ProtocolDatasetContext,
    records: list[ImageRecord],
    impression_type: str,
    error_cls: type[Exception],
) -> dict[tuple[str, str, int], ImageRecord]:
    records_by_identity: dict[tuple[str, str, int], list[ImageRecord]] = {}
    for record in records:
        if record.impression_type != impression_type:
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
        key: duplicate_records
        for key, duplicate_records in records_by_identity.items()
        if len(duplicate_records) > 1
    }
    if duplicates:
        raise error_cls(_format_duplicate_identities(impression_type, duplicates))

    return {
        identity: identity_records[0]
        for identity, identity_records in records_by_identity.items()
    }


def _canonical_position_for_record(record: ImageRecord, error_cls: type[Exception]) -> int | None:
    try:
        return canonical_finger_position(record.impression_type, record.frgp)
    except CanonicalFingerMappingError as exc:
        raise error_cls(
            f"Cannot map {record.impression_type!r} FRGP {record.frgp} for {record.absolute_path}: {exc}"
        ) from exc


def _expected_records_by_identity(
    context: ProtocolDatasetContext,
    data_root: Path,
    generation_error_cls: type[Exception],
    validation_error_cls: type[Exception],
) -> tuple[dict[tuple[str, str, int], ImageRecord], dict[tuple[str, str, int], ImageRecord]]:
    scan_result = scan_dataset(data_root, context.spec)
    try:
        return _eligible_records_by_identity(context, scan_result, generation_error_cls)
    except generation_error_cls as exc:
        raise validation_error_cls(
            f"Cannot validate {context.name}/{PROTOCOL}: expected {_dataset_label(context)} "
            f"identities are invalid: {exc}"
        ) from exc


def _format_scan_errors(context: ProtocolDatasetContext, errors: list[ScanError]) -> str:
    details = "; ".join(f"{error.path}: {error.message}" for error in errors[:5])
    suffix = "" if len(errors) <= 5 else f"; ... and {len(errors) - 5} more"
    return (
        f"Cannot generate {context.name}/{PROTOCOL}: {_dataset_label(context)} scan has "
        f"{len(errors)} error(s): {details}{suffix}"
    )


def _format_duplicate_identities(
    impression_type: str,
    duplicates: dict[tuple[str, str, int], list[ImageRecord]],
) -> str:
    details: list[str] = []
    for (dataset, subject_id, canonical_position), records in sorted(duplicates.items()):
        paths = ", ".join(str(record.absolute_path) for record in records[:3])
        suffix = "" if len(records) <= 3 else f", ... and {len(records) - 3} more"
        details.append(
            f"{dataset} subject {subject_id} canonical {canonical_position:02d} "
            f"has {len(records)} {impression_type} records: {paths}{suffix}"
        )
    return f"Duplicate {impression_type} anatomical identities found: " + "; ".join(details)


def _validate_source_path(
    context: ProtocolDatasetContext,
    path: Path,
    impression_type: str,
    row_number: int,
    validation_error_cls: type[Exception],
) -> ImageRecord:
    try:
        return validate_image_path(path, context.spec, impression_type)
    except SchemaValidationError as exc:
        raise validation_error_cls(
            f"Path does not validate as an {_dataset_label(context)} {impression_type} record at "
            f"CSV line {row_number}: {exc}"
        ) from exc


def _canonical_position_for_row(
    impression_type: str,
    raw_frgp: int,
    row_number: int,
    validation_error_cls: type[Exception],
) -> int | None:
    try:
        return canonical_finger_position(impression_type, raw_frgp)
    except CanonicalFingerMappingError as exc:
        raise validation_error_cls(
            f"raw FRGP is not valid for {impression_type} at CSV line {row_number}: {exc}"
        ) from exc


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
            f"path_a at CSV line {row_number} is not under the {_dataset_label(context)} plain directory: {path}"
        ) from exc


def _ensure_path_under_roll_dataset(
    context: ProtocolDatasetContext,
    path: Path,
    expected_roll_dir: Path,
    row_number: int,
    validation_error_cls: type[Exception],
) -> None:
    try:
        path.resolve().relative_to(expected_roll_dir)
    except ValueError as exc:
        raise validation_error_cls(
            f"path_b at CSV line {row_number} is not under the {_dataset_label(context)} roll directory: {path}"
        ) from exc


def _validate_row_matches_records(
    context: ProtocolDatasetContext,
    row: dict[str, str],
    plain_record: ImageRecord,
    roll_record: ImageRecord,
    ppi: int,
    plain_raw_frgp: int,
    roll_raw_frgp: int,
    row_number: int,
    validation_error_cls: type[Exception],
) -> None:
    if plain_record.subject_id != row["subject_id"]:
        raise validation_error_cls(
            f"Subject mismatch for path_a at CSV line {row_number}: manifest has {row['subject_id']!r}, "
            f"path has {plain_record.subject_id!r}."
        )
    if roll_record.subject_id != row["subject_id"]:
        raise validation_error_cls(
            f"Subject mismatch for path_b at CSV line {row_number}: manifest has {row['subject_id']!r}, "
            f"path has {roll_record.subject_id!r}."
        )
    if plain_record.subject_id != roll_record.subject_id:
        raise validation_error_cls(
            f"path_a and path_b belong to different subjects at CSV line {row_number}: "
            f"{plain_record.subject_id!r} != {roll_record.subject_id!r}."
        )
    if plain_record.ppi != roll_record.ppi:
        raise validation_error_cls(
            f"PPI mismatch between path_a and path_b at CSV line {row_number}: "
            f"{plain_record.ppi} != {roll_record.ppi}."
        )
    if ppi != plain_record.ppi:
        raise validation_error_cls(
            f"PPI mismatch at CSV line {row_number}: manifest has {ppi}, path_a has {plain_record.ppi}."
        )
    if ppi != roll_record.ppi:
        raise validation_error_cls(
            f"PPI mismatch at CSV line {row_number}: manifest has {ppi}, path_b has {roll_record.ppi}."
        )
    if ppi != context.expected_ppi:
        raise validation_error_cls(
            f"PPI mismatch at CSV line {row_number}: manifest has {ppi}, "
            f"{context.name} expects {context.expected_ppi}."
        )
    if plain_raw_frgp != plain_record.frgp:
        raise validation_error_cls(
            f"FRGP mismatch for path_a at CSV line {row_number}: "
            f"manifest has {plain_raw_frgp}, path has {plain_record.frgp}."
        )
    if roll_raw_frgp != roll_record.frgp:
        raise validation_error_cls(
            f"FRGP mismatch for path_b at CSV line {row_number}: "
            f"manifest has {roll_raw_frgp}, path has {roll_record.frgp}."
        )


def _validate_expected_source_records(
    context: ProtocolDatasetContext,
    plain_by_identity: dict[tuple[str, str, int], ImageRecord],
    roll_by_identity: dict[tuple[str, str, int], ImageRecord],
    actual_rows: dict[tuple[str, str, int], _ManifestRowRecord],
    validation_error_cls: type[Exception],
) -> None:
    for identity in sorted(set(plain_by_identity) & set(roll_by_identity)):
        expected_plain = plain_by_identity[identity]
        expected_roll = roll_by_identity[identity]
        actual = actual_rows[identity]
        if actual.plain_path.resolve() != expected_plain.absolute_path.resolve():
            raise validation_error_cls(
                "Manifest identity points to the wrong plain source path at CSV line "
                f"{actual.row_number}: {identity} expected {expected_plain.absolute_path}, got {actual.plain_path}."
            )
        if actual.roll_path.resolve() != expected_roll.absolute_path.resolve():
            raise validation_error_cls(
                "Manifest identity points to the wrong roll source path at CSV line "
                f"{actual.row_number}: {identity} expected {expected_roll.absolute_path}, got {actual.roll_path}."
            )
        if actual.plain_raw_frgp != expected_plain.frgp:
            raise validation_error_cls(
                "Manifest identity points to the wrong plain raw FRGP at CSV line "
                f"{actual.row_number}: {identity} expected {expected_plain.frgp}, got {actual.plain_raw_frgp}."
            )
        if actual.roll_raw_frgp != expected_roll.frgp:
            raise validation_error_cls(
                "Manifest identity points to the wrong roll raw FRGP at CSV line "
                f"{actual.row_number}: {identity} expected {expected_roll.frgp}, got {actual.roll_raw_frgp}."
            )
        if actual.ppi != expected_plain.ppi or actual.ppi != expected_roll.ppi:
            raise validation_error_cls(
                "Manifest identity points to the wrong PPI at CSV line "
                f"{actual.row_number}: {identity} expected {expected_plain.ppi}, got {actual.ppi}."
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
