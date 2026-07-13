"""Discovery and filename validation for NIST SD300b and SD300c PNG images."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .canonical_fingers import (
    CANONICAL_FINGER_POSITIONS,
    canonical_finger_position,
    is_plain_multi_finger_capture,
)


DEFAULT_DATA_ROOT = Path(r"C:\fingerprint-datasets")

FILENAME_RE = re.compile(
    r"^(?P<subject_id>\d{8})_"
    r"(?P<impression_type>plain|roll)_"
    r"(?P<ppi>\d{3,4})_"
    r"(?P<frgp>\d{2})"
    r"(?P<extension>\.png)$"
)

FRGP_NAMES = {
    1: "right_thumb",
    2: "right_index",
    3: "right_middle",
    4: "right_ring",
    5: "right_little",
    6: "left_thumb",
    7: "left_index",
    8: "left_middle",
    9: "left_ring",
    10: "left_little",
    11: "plain_right_thumb",
    12: "plain_left_thumb",
    13: "plain_right_four_fingers",
    14: "plain_left_four_fingers",
}

EXPECTED_FRGP_BY_IMPRESSION = {
    "roll": frozenset(range(1, 11)),
    "plain": frozenset({2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14}),
}


class SchemaValidationError(ValueError):
    """Raised when a filename or path does not match the SD300 schema."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    ppi: int
    relative_image_root: Path

    def impression_dir(self, data_root: Path, impression_type: str) -> Path:
        return data_root / self.relative_image_root / impression_type


DATASETS = {
    "sd300b": DatasetSpec(
        name="sd300b",
        ppi=1000,
        relative_image_root=Path("NIST") / "sd300b" / "images" / "1000" / "png",
    ),
    "sd300c": DatasetSpec(
        name="sd300c",
        ppi=2000,
        relative_image_root=Path("NIST") / "sd300c" / "images" / "2000" / "png",
    ),
}


@dataclass(frozen=True)
class ParsedFilename:
    subject_id: str
    impression_type: str
    ppi: int
    frgp: int
    extension: str

    @property
    def finger_position(self) -> str:
        return FRGP_NAMES.get(self.frgp, f"unknown_frgp_{self.frgp:02d}")


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    subject_id: str
    impression_type: str
    ppi: int
    frgp: int
    finger_position: str
    absolute_path: Path

    def as_dict(self) -> dict[str, str | int]:
        return {
            "dataset": self.dataset,
            "subject_id": self.subject_id,
            "impression_type": self.impression_type,
            "ppi": self.ppi,
            "frgp": self.frgp,
            "finger_position": self.finger_position,
            "absolute_path": str(self.absolute_path),
        }


@dataclass(frozen=True)
class ScanError:
    path: Path
    message: str


@dataclass(frozen=True)
class ScanResult:
    records: list[ImageRecord]
    errors: list[ScanError]

    def audit_summary(self) -> dict[str, object]:
        subjects = {record.subject_id for record in self.records}
        plain_count = sum(1 for record in self.records if record.impression_type == "plain")
        roll_count = sum(1 for record in self.records if record.impression_type == "roll")
        frgp_counts = Counter(record.frgp for record in self.records)
        finger_counts = Counter(record.finger_position for record in self.records)
        canonical_plain_counts = _canonical_counts(self.records, "plain")
        canonical_roll_counts = _canonical_counts(self.records, "roll")
        plain_multi_finger_captures = sum(
            1
            for record in self.records
            if is_plain_multi_finger_capture(record.impression_type, record.frgp)
        )

        return {
            "plain_images": plain_count,
            "roll_images": roll_count,
            "subjects": len(subjects),
            "finger_positions": dict(sorted(finger_counts.items())),
            "frgp": {f"{frgp:02d}": frgp_counts[frgp] for frgp in sorted(frgp_counts)},
            "canonical_single_finger_plain": canonical_plain_counts,
            "canonical_single_finger_roll": canonical_roll_counts,
            "plain_multi_finger_captures_not_pairable": plain_multi_finger_captures,
            "invalid_files": len(self.errors),
        }


def _canonical_counts(records: Iterable[ImageRecord], impression_type: str) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for record in records:
        if record.impression_type != impression_type:
            continue

        canonical_position = canonical_finger_position(record.impression_type, record.frgp)
        if canonical_position is not None:
            counts[canonical_position] += 1

    return {
        f"{position:02d}": counts[position]
        for position in CANONICAL_FINGER_POSITIONS
    }


def parse_image_filename(filename: str) -> ParsedFilename:
    match = FILENAME_RE.fullmatch(filename)
    if not match:
        raise SchemaValidationError(
            "Invalid SD300 image filename. Expected "
            "SUBJECT_IMPRESSION_PPI_FRGP.png, for example "
            "00001000_roll_1000_01.png."
        )

    parsed = ParsedFilename(
        subject_id=match.group("subject_id"),
        impression_type=match.group("impression_type"),
        ppi=int(match.group("ppi")),
        frgp=int(match.group("frgp")),
        extension=match.group("extension"),
    )

    if parsed.frgp not in FRGP_NAMES:
        raise SchemaValidationError(
            f"Unsupported FRGP code {parsed.frgp:02d} in filename {filename}."
        )

    return parsed


def validate_image_path(path: Path, spec: DatasetSpec, expected_impression: str) -> ImageRecord:
    if expected_impression not in EXPECTED_FRGP_BY_IMPRESSION:
        raise SchemaValidationError(f"Unsupported impression type {expected_impression!r}.")

    if path.parent.name != expected_impression:
        raise SchemaValidationError(
            f"File {path} is not directly under an expected {expected_impression!r} directory."
        )

    parsed = parse_image_filename(path.name)

    if parsed.impression_type != expected_impression:
        raise SchemaValidationError(
            f"Filename impression {parsed.impression_type!r} does not match directory "
            f"{expected_impression!r} for {path.name}."
        )

    if parsed.ppi != spec.ppi:
        raise SchemaValidationError(
            f"Filename PPI {parsed.ppi} does not match dataset {spec.name} PPI "
            f"{spec.ppi} for {path.name}."
        )

    expected_frgp = EXPECTED_FRGP_BY_IMPRESSION[expected_impression]
    if parsed.frgp not in expected_frgp:
        allowed = ", ".join(f"{frgp:02d}" for frgp in sorted(expected_frgp))
        raise SchemaValidationError(
            f"FRGP {parsed.frgp:02d} is not valid for {expected_impression!r}; "
            f"allowed codes are {allowed}."
        )

    return ImageRecord(
        dataset=spec.name,
        subject_id=parsed.subject_id,
        impression_type=parsed.impression_type,
        ppi=parsed.ppi,
        frgp=parsed.frgp,
        finger_position=parsed.finger_position,
        absolute_path=path if path.is_absolute() else path.resolve(),
    )


def _candidate_files(directory: Path) -> Iterable[Path]:
    return (path for path in directory.rglob("*") if path.is_file())


def scan_dataset(data_root: Path, spec: DatasetSpec) -> ScanResult:
    records: list[ImageRecord] = []
    errors: list[ScanError] = []

    for impression_type in ("plain", "roll"):
        impression_dir = spec.impression_dir(data_root, impression_type)
        if not impression_dir.is_dir():
            errors.append(
                ScanError(
                    path=impression_dir,
                    message=f"Missing expected {spec.name} {impression_type!r} directory.",
                )
            )
            continue

        for path in sorted(_candidate_files(impression_dir)):
            try:
                records.append(validate_image_path(path, spec, impression_type))
            except SchemaValidationError as exc:
                errors.append(ScanError(path=path, message=str(exc)))

    return ScanResult(records=records, errors=errors)


def scan_all_datasets(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, ScanResult]:
    return {
        dataset_name: scan_dataset(data_root, spec)
        for dataset_name, spec in DATASETS.items()
    }
