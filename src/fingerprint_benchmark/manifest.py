"""Read pair manifests produced by the discovery layer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


class ManifestReadError(ValueError):
    """Raised when a benchmark manifest cannot be read safely."""


@dataclass(frozen=True)
class PairRecord:
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

    def image_metadata_a(self) -> dict[str, Any]:
        return self._image_metadata("a", self.raw_frgp_a, self.path_a)

    def image_metadata_b(self) -> dict[str, Any]:
        return self._image_metadata("b", self.raw_frgp_b, self.path_b)

    def _image_metadata(self, side: str, raw_frgp: int, path: Path) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "side": side,
            "dataset": self.dataset,
            "protocol": self.protocol,
            "subject_id": self.subject_id,
            "canonical_finger_position": self.canonical_finger_position,
            "ppi": self.ppi,
            "raw_frgp": raw_frgp,
            "path": str(path),
        }


def read_pair_manifest(manifest_path: Path) -> list[PairRecord]:
    if not manifest_path.is_file():
        raise ManifestReadError(f"Manifest file does not exist: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise ManifestReadError(
                f"Manifest schema mismatch. Expected columns {MANIFEST_COLUMNS}, got {reader.fieldnames}."
            )
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ManifestReadError(f"Extra unnamed CSV values at line {row_number}.")
            rows.append(_pair_record(row, row_number))

    pair_ids = [row.pair_id for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ManifestReadError("Manifest contains duplicate pair_id values.")
    return rows


def _pair_record(row: dict[str, str], row_number: int) -> PairRecord:
    for column in MANIFEST_COLUMNS:
        if row.get(column) in (None, ""):
            raise ManifestReadError(f"Missing value for {column!r} at CSV line {row_number}.")
    try:
        canonical_finger_position = int(row["canonical_finger_position"])
        ppi = int(row["ppi"])
        raw_frgp_a = int(row["raw_frgp_a"])
        raw_frgp_b = int(row["raw_frgp_b"])
    except ValueError as exc:
        raise ManifestReadError(f"Integer field is invalid at CSV line {row_number}.") from exc

    return PairRecord(
        pair_id=row["pair_id"],
        dataset=row["dataset"],
        protocol=row["protocol"],
        subject_id=row["subject_id"],
        canonical_finger_position=canonical_finger_position,
        ppi=ppi,
        raw_frgp_a=raw_frgp_a,
        raw_frgp_b=raw_frgp_b,
        path_a=Path(row["path_a"]),
        path_b=Path(row["path_b"]),
    )
