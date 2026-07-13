"""Cross-protocol audit for the existing SD300c manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

from .nist_sd300 import DEFAULT_DATA_ROOT
from .protocol_dataset import SD300C_CONTEXT
from . import sd300c_plain_roll
from . import sd300c_plain_self
from . import sd300c_roll_self
from .self_manifest_common import (
    Identity,
    format_identities,
    parse_int,
    read_manifest_rows,
    validate_required_values,
)


DATASET_CONTEXT = SD300C_CONTEXT
DATASET = DATASET_CONTEXT.name
DEFAULT_REPORT_PATH = DATASET_CONTEXT.protocol_output_dir / "protocol_audit.json"


class ProtocolAuditError(ValueError):
    """Raised when the SD300c protocol cross-audit fails."""


@dataclass(frozen=True)
class _SelfRecord:
    row_number: int
    pair_id: str
    identity: Identity
    ppi: int
    raw_frgp: int
    path: Path


@dataclass(frozen=True)
class _PlainRollRecord:
    row_number: int
    pair_id: str
    identity: Identity
    ppi: int
    raw_frgp_a: int
    raw_frgp_b: int
    path_a: Path
    path_b: Path


@dataclass(frozen=True)
class ProtocolAuditReport:
    dataset: str
    plain_self_count: int
    roll_self_count: int
    plain_roll_count: int
    expected_intersection_count: int
    plain_only_count: int
    roll_only_count: int
    intersection_consistent: bool
    source_consistency_checked_pairs: int
    canonical_pair_counts: dict[str, dict[str, int]]
    manifest_sha256: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_pair_counts": self.canonical_pair_counts,
            "dataset": self.dataset,
            "expected_intersection_count": self.expected_intersection_count,
            "intersection_consistent": self.intersection_consistent,
            "manifest_sha256": self.manifest_sha256,
            "plain_only_count": self.plain_only_count,
            "plain_roll_count": self.plain_roll_count,
            "plain_self_count": self.plain_self_count,
            "roll_only_count": self.roll_only_count,
            "roll_self_count": self.roll_self_count,
            "source_consistency_checked_pairs": self.source_consistency_checked_pairs,
        }


def build_protocol_audit_report(
    data_root: Path = DEFAULT_DATA_ROOT,
    plain_self_manifest: Path = sd300c_plain_self.DEFAULT_MANIFEST_PATH,
    roll_self_manifest: Path = sd300c_roll_self.DEFAULT_MANIFEST_PATH,
    plain_roll_manifest: Path = sd300c_plain_roll.DEFAULT_MANIFEST_PATH,
) -> ProtocolAuditReport:
    """Validate the three SD300c manifests, then check their cross-protocol relation."""

    _run_dedicated_validators(data_root, plain_self_manifest, roll_self_manifest, plain_roll_manifest)

    plain_self_records = _read_self_records(
        plain_self_manifest,
        expected_protocol=sd300c_plain_self.PROTOCOL,
        raw_frgp_column="raw_frgp_a",
        path_column="path_a",
    )
    roll_self_records = _read_self_records(
        roll_self_manifest,
        expected_protocol=sd300c_roll_self.PROTOCOL,
        raw_frgp_column="raw_frgp_a",
        path_column="path_a",
    )
    plain_roll_records = _read_plain_roll_records(plain_roll_manifest)

    plain_self_identities = set(plain_self_records)
    roll_self_identities = set(roll_self_records)
    plain_roll_identities = set(plain_roll_records)
    expected_intersection = plain_self_identities & roll_self_identities

    _validate_plain_roll_is_intersection(plain_roll_identities, expected_intersection)
    _validate_source_consistency(plain_self_records, roll_self_records, plain_roll_records)

    return ProtocolAuditReport(
        dataset=DATASET,
        plain_self_count=len(plain_self_identities),
        roll_self_count=len(roll_self_identities),
        plain_roll_count=len(plain_roll_identities),
        expected_intersection_count=len(expected_intersection),
        plain_only_count=len(plain_self_identities - roll_self_identities),
        roll_only_count=len(roll_self_identities - plain_self_identities),
        intersection_consistent=True,
        source_consistency_checked_pairs=len(plain_roll_identities),
        canonical_pair_counts={
            "plain_roll": _canonical_counts(plain_roll_identities),
            "plain_self": _canonical_counts(plain_self_identities),
            "roll_self": _canonical_counts(roll_self_identities),
        },
        manifest_sha256={
            "plain_roll": _sha256(plain_roll_manifest),
            "plain_self": _sha256(plain_self_manifest),
            "roll_self": _sha256(roll_self_manifest),
        },
    )


def save_report_json(report: ProtocolAuditReport, output_path: Path = DEFAULT_REPORT_PATH) -> None:
    """Write a deterministic JSON audit report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_report_json(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit consistency across the existing SD300c protocol manifests."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root directory. Defaults to {DEFAULT_DATA_ROOT}.",
    )
    parser.add_argument(
        "--plain-self-manifest",
        type=Path,
        default=sd300c_plain_self.DEFAULT_MANIFEST_PATH,
        help=f"plain_self manifest path. Defaults to {sd300c_plain_self.DEFAULT_MANIFEST_PATH}.",
    )
    parser.add_argument(
        "--roll-self-manifest",
        type=Path,
        default=sd300c_roll_self.DEFAULT_MANIFEST_PATH,
        help=f"roll_self manifest path. Defaults to {sd300c_roll_self.DEFAULT_MANIFEST_PATH}.",
    )
    parser.add_argument(
        "--plain-roll-manifest",
        type=Path,
        default=sd300c_plain_roll.DEFAULT_MANIFEST_PATH,
        help=f"plain_roll manifest path. Defaults to {sd300c_plain_roll.DEFAULT_MANIFEST_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Optional deterministic JSON report output path, for example {DEFAULT_REPORT_PATH}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_protocol_audit_report(
            data_root=args.data_root,
            plain_self_manifest=args.plain_self_manifest,
            roll_self_manifest=args.roll_self_manifest,
            plain_roll_manifest=args.plain_roll_manifest,
        )
        if args.output is not None:
            save_report_json(report, args.output)
        print(_report_json(report), end="")
    except ProtocolAuditError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_dedicated_validators(
    data_root: Path,
    plain_self_manifest: Path,
    roll_self_manifest: Path,
    plain_roll_manifest: Path,
) -> None:
    validators: list[tuple[str, Callable[[Path, Path], object], Path]] = [
        ("plain_self", sd300c_plain_self.validate_manifest, plain_self_manifest),
        ("roll_self", sd300c_roll_self.validate_manifest, roll_self_manifest),
        ("plain_roll", sd300c_plain_roll.validate_manifest, plain_roll_manifest),
    ]
    for protocol, validator, manifest_path in validators:
        try:
            validator(manifest_path, data_root)
        except Exception as exc:
            raise ProtocolAuditError(
                f"Dedicated validator failed for {protocol}: {exc}"
            ) from exc


def _read_self_records(
    manifest_path: Path,
    expected_protocol: str,
    raw_frgp_column: str,
    path_column: str,
) -> dict[Identity, _SelfRecord]:
    records: dict[Identity, _SelfRecord] = {}
    rows = read_manifest_rows(manifest_path, ProtocolAuditError)
    for row_number, row in enumerate(rows, start=2):
        validate_required_values(row, row_number, ProtocolAuditError)
        _validate_constant(row, "dataset", DATASET, row_number)
        _validate_constant(row, "protocol", expected_protocol, row_number)
        identity = _identity_from_row(row, row_number)
        if identity in records:
            raise ProtocolAuditError(
                f"Duplicate identity in {expected_protocol} manifest at CSV line {row_number}: {identity}."
            )
        records[identity] = _SelfRecord(
            row_number=row_number,
            pair_id=row["pair_id"],
            identity=identity,
            ppi=parse_int(row, "ppi", row_number, ProtocolAuditError),
            raw_frgp=parse_int(row, raw_frgp_column, row_number, ProtocolAuditError),
            path=Path(row[path_column]),
        )
    return records


def _read_plain_roll_records(manifest_path: Path) -> dict[Identity, _PlainRollRecord]:
    records: dict[Identity, _PlainRollRecord] = {}
    rows = read_manifest_rows(manifest_path, ProtocolAuditError)
    for row_number, row in enumerate(rows, start=2):
        validate_required_values(row, row_number, ProtocolAuditError)
        _validate_constant(row, "dataset", DATASET, row_number)
        _validate_constant(row, "protocol", sd300c_plain_roll.PROTOCOL, row_number)
        identity = _identity_from_row(row, row_number)
        if identity in records:
            raise ProtocolAuditError(
                f"Duplicate identity in plain_roll manifest at CSV line {row_number}: {identity}."
            )
        records[identity] = _PlainRollRecord(
            row_number=row_number,
            pair_id=row["pair_id"],
            identity=identity,
            ppi=parse_int(row, "ppi", row_number, ProtocolAuditError),
            raw_frgp_a=parse_int(row, "raw_frgp_a", row_number, ProtocolAuditError),
            raw_frgp_b=parse_int(row, "raw_frgp_b", row_number, ProtocolAuditError),
            path_a=Path(row["path_a"]),
            path_b=Path(row["path_b"]),
        )
    return records


def _identity_from_row(row: dict[str, str], row_number: int) -> Identity:
    canonical_position = parse_int(row, "canonical_finger_position", row_number, ProtocolAuditError)
    return (row["dataset"], row["subject_id"], canonical_position)


def _validate_constant(row: dict[str, str], column: str, expected: str, row_number: int) -> None:
    if row[column] != expected:
        raise ProtocolAuditError(
            f"Invalid {column} {row[column]!r} at CSV line {row_number}; expected {expected!r}."
        )


def _validate_plain_roll_is_intersection(
    plain_roll_identities: set[Identity],
    expected_intersection: set[Identity],
) -> None:
    missing = expected_intersection - plain_roll_identities
    extra = plain_roll_identities - expected_intersection
    if missing:
        raise ProtocolAuditError(
            "plain_roll is missing identities from plain_self and roll_self intersection: "
            f"{format_identities(missing)}."
        )
    if extra:
        raise ProtocolAuditError(
            "plain_roll contains identities outside plain_self and roll_self intersection: "
            f"{format_identities(extra)}."
        )


def _validate_source_consistency(
    plain_self_records: dict[Identity, _SelfRecord],
    roll_self_records: dict[Identity, _SelfRecord],
    plain_roll_records: dict[Identity, _PlainRollRecord],
) -> None:
    for identity, plain_roll_record in sorted(plain_roll_records.items()):
        plain_self_record = plain_self_records[identity]
        roll_self_record = roll_self_records[identity]
        if (
            plain_roll_record.path_a.resolve() == roll_self_record.path.resolve()
            and plain_roll_record.path_b.resolve() == plain_self_record.path.resolve()
        ):
            raise ProtocolAuditError(
                "plain_roll path_a/path_b appear swapped for identity "
                f"{identity} at CSV line {plain_roll_record.row_number}: "
                "path_a matches roll_self and path_b matches plain_self."
            )
        _validate_plain_source(identity, plain_roll_record, plain_self_record)
        _validate_roll_source(identity, plain_roll_record, roll_self_record)
        if plain_roll_record.ppi != plain_self_record.ppi:
            raise ProtocolAuditError(
                "PPI mismatch between plain_roll and plain_self for identity "
                f"{identity} at CSV line {plain_roll_record.row_number}: "
                f"{plain_roll_record.ppi} != {plain_self_record.ppi}."
            )
        if plain_roll_record.ppi != roll_self_record.ppi:
            raise ProtocolAuditError(
                "PPI mismatch between plain_roll and roll_self for identity "
                f"{identity} at CSV line {plain_roll_record.row_number}: "
                f"{plain_roll_record.ppi} != {roll_self_record.ppi}."
            )


def _validate_plain_source(
    identity: Identity,
    plain_roll_record: _PlainRollRecord,
    plain_self_record: _SelfRecord,
) -> None:
    if plain_roll_record.path_a.resolve() != plain_self_record.path.resolve():
        _raise_path_mismatch(
            identity,
            plain_roll_record,
            "path_a",
            "plain_self",
            plain_self_record.path,
            plain_roll_record.path_a,
        )
    if plain_roll_record.raw_frgp_a != plain_self_record.raw_frgp:
        raise ProtocolAuditError(
            "raw_frgp_a mismatch between plain_roll and plain_self for identity "
            f"{identity} at CSV line {plain_roll_record.row_number}: "
            f"{plain_roll_record.raw_frgp_a} != {plain_self_record.raw_frgp}."
        )


def _validate_roll_source(
    identity: Identity,
    plain_roll_record: _PlainRollRecord,
    roll_self_record: _SelfRecord,
) -> None:
    if plain_roll_record.path_b.resolve() != roll_self_record.path.resolve():
        _raise_path_mismatch(
            identity,
            plain_roll_record,
            "path_b",
            "roll_self",
            roll_self_record.path,
            plain_roll_record.path_b,
        )
    if plain_roll_record.raw_frgp_b != roll_self_record.raw_frgp:
        raise ProtocolAuditError(
            "raw_frgp_b mismatch between plain_roll and roll_self for identity "
            f"{identity} at CSV line {plain_roll_record.row_number}: "
            f"{plain_roll_record.raw_frgp_b} != {roll_self_record.raw_frgp}."
        )


def _raise_path_mismatch(
    identity: Identity,
    plain_roll_record: _PlainRollRecord,
    column: str,
    expected_protocol: str,
    expected_path: Path,
    actual_path: Path,
) -> None:
    swapped_hint = ""
    if column == "path_a" and actual_path.resolve() == plain_roll_record.path_b.resolve():
        swapped_hint = " This looks like path_a/path_b were swapped."
    elif column == "path_b" and actual_path.resolve() == plain_roll_record.path_a.resolve():
        swapped_hint = " This looks like path_a/path_b were swapped."

    raise ProtocolAuditError(
        f"{column} mismatch between plain_roll and {expected_protocol} for identity "
        f"{identity} at CSV line {plain_roll_record.row_number}: "
        f"expected {expected_path}, got {actual_path}.{swapped_hint}"
    )


def _canonical_counts(identities: set[Identity]) -> dict[str, int]:
    counts = Counter(identity[2] for identity in identities)
    return {f"{position:02d}": counts[position] for position in range(1, 11)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_json(report: ProtocolAuditReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
