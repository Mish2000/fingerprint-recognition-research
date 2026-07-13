"""Shared mechanics for SD300b self-manifest CSV files.

This module deliberately contains only protocol-neutral plumbing. It does not
decide which records are eligible, how FRGP values map anatomically, or which
source directory a protocol should use.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Protocol, TypeVar


MANIFEST_COLUMNS = [
    "pair_id",
    "dataset",
    "protocol",
    "subject_id",
    "canonical_finger_position",
    "ppi",
    "raw_frgp_a",
    "raw_frgp_b",
    "path_a",
    "path_b",
]

Identity = tuple[str, str, int]
ReportT = TypeVar("ReportT")


class SelfManifestPair(Protocol):
    pair_id: str

    def as_csv_row(self) -> dict[str, str]:
        ...


def write_validated_manifest(
    pairs: Iterable[SelfManifestPair],
    output_path: Path,
    validate_candidate: Callable[[Path], ReportT],
    error_cls: type[Exception],
) -> ReportT:
    """Write a candidate file, validate it, then atomically replace the target."""

    temp_path: Path | None = None
    try:
        temp_path = write_manifest_candidate(pairs, output_path, error_cls)
        report = validate_candidate(temp_path)
        os.replace(temp_path, output_path)
        temp_path = None
        return report
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_manifest_atomic(
    pairs: Iterable[SelfManifestPair],
    output_path: Path,
    error_cls: type[Exception],
) -> None:
    """Write a manifest with an atomic replace, without protocol-level validation."""

    temp_path: Path | None = None
    try:
        temp_path = write_manifest_candidate(pairs, output_path, error_cls)
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_manifest_candidate(
    pairs: Iterable[SelfManifestPair],
    output_path: Path,
    error_cls: type[Exception],
) -> Path:
    pairs = list(pairs)
    ensure_unique_pair_ids(pairs, error_cls)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for pair in pairs:
                writer.writerow(pair.as_csv_row())
            handle.flush()
            os.fsync(handle.fileno())

        return temp_path
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def ensure_unique_pair_ids(
    pairs: Iterable[SelfManifestPair],
    error_cls: type[Exception],
) -> None:
    seen: set[str] = set()
    collisions: list[str] = []
    for pair in pairs:
        if pair.pair_id in seen:
            collisions.append(pair.pair_id)
        seen.add(pair.pair_id)

    if collisions:
        joined = ", ".join(sorted(set(collisions)))
        raise error_cls(f"Pair ID collision detected before writing: {joined}.")


def read_manifest_rows(
    manifest_path: Path,
    error_cls: type[Exception],
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise error_cls(f"Manifest file does not exist: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise error_cls(
                f"Manifest schema mismatch. Expected columns {MANIFEST_COLUMNS}, "
                f"got {reader.fieldnames}."
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise error_cls(f"Extra unnamed CSV values at line {row_number}.")
            rows.append(row)
    return rows


def validate_required_values(
    row: dict[str, str],
    row_number: int,
    error_cls: type[Exception],
) -> None:
    for column in MANIFEST_COLUMNS:
        if row.get(column) in (None, ""):
            raise error_cls(f"Missing value for {column!r} at CSV line {row_number}.")


def parse_int(
    row: dict[str, str],
    column: str,
    row_number: int,
    error_cls: type[Exception],
) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise error_cls(
            f"Column {column!r} must be an integer at CSV line {row_number}; got {row[column]!r}."
        ) from exc


def ensure_unique_row(
    row: dict[str, str],
    seen_rows: set[tuple[str, ...]],
    row_number: int,
    error_cls: type[Exception],
) -> None:
    row_tuple = tuple(row[column] for column in MANIFEST_COLUMNS)
    if row_tuple in seen_rows:
        raise error_cls(f"Duplicate manifest row at CSV line {row_number}.")
    seen_rows.add(row_tuple)


def ensure_unique_value(
    value: str,
    seen_values: set[str],
    label: str,
    row_number: int,
    error_cls: type[Exception],
) -> None:
    if value in seen_values:
        raise error_cls(f"Duplicate {label} {value!r} at CSV line {row_number}.")
    seen_values.add(value)


def validate_completeness(
    expected_identities: set[Identity],
    actual_identities: set[Identity],
    error_cls: type[Exception],
) -> None:
    missing = expected_identities - actual_identities
    extra = actual_identities - expected_identities

    if missing:
        raise error_cls(
            "Manifest is incomplete; missing expected anatomical identities: "
            f"{format_identities(missing)}."
        )
    if extra:
        raise error_cls(
            "Manifest contains unexpected anatomical identities: "
            f"{format_identities(extra)}."
        )


def format_identities(identities: set[Identity]) -> str:
    formatted = [
        f"({dataset}, {subject_id}, {canonical_position:02d})"
        for dataset, subject_id, canonical_position in sorted(identities)
    ]
    if len(formatted) <= 10:
        return ", ".join(formatted)
    shown = ", ".join(formatted[:10])
    return f"{shown}, ... and {len(formatted) - 10} more"
