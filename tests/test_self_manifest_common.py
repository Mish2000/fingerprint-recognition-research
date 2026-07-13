from dataclasses import dataclass
from pathlib import Path

import pytest

from fingerprint_data_discovery.self_manifest_common import (
    MANIFEST_COLUMNS,
    ensure_unique_pair_ids,
    format_identities,
    parse_int,
    read_manifest_rows,
    validate_completeness,
    validate_required_values,
    write_manifest_atomic,
    write_validated_manifest,
)


class CommonManifestError(ValueError):
    pass


@dataclass(frozen=True)
class DummyPair:
    pair_id: str
    path: Path

    def as_csv_row(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "dataset": "sd300b",
            "protocol": "dummy_self",
            "subject_id": "00001000",
            "canonical_finger_position": "1",
            "ppi": "1000",
            "raw_frgp_a": "1",
            "raw_frgp_b": "1",
            "path_a": str(self.path),
            "path_b": str(self.path),
        }


def test_manifest_columns_are_the_self_protocol_schema():
    assert MANIFEST_COLUMNS == [
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


def test_write_and_read_manifest_rows_with_schema_verification(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pair = DummyPair(pair_id="pair-1", path=tmp_path / "image.png")

    write_manifest_atomic([pair], manifest_path, CommonManifestError)
    rows = read_manifest_rows(manifest_path, CommonManifestError)

    assert rows == [pair.as_csv_row()]


def test_read_manifest_rows_rejects_schema_mismatch(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("pair_id,dataset\npair-1,sd300b\n", encoding="utf-8")

    with pytest.raises(CommonManifestError, match="schema mismatch"):
        read_manifest_rows(manifest_path, CommonManifestError)


def test_required_values_and_int_parsing_fail_explicitly():
    row = {column: "1" for column in MANIFEST_COLUMNS}
    row["subject_id"] = ""

    with pytest.raises(CommonManifestError, match="Missing value"):
        validate_required_values(row, 2, CommonManifestError)

    row["subject_id"] = "00001000"
    row["ppi"] = "not-an-int"
    with pytest.raises(CommonManifestError, match="must be an integer"):
        parse_int(row, "ppi", 2, CommonManifestError)


def test_pair_id_collisions_fail_before_writing(tmp_path):
    pairs = [
        DummyPair(pair_id="duplicate", path=tmp_path / "a.png"),
        DummyPair(pair_id="duplicate", path=tmp_path / "b.png"),
    ]

    with pytest.raises(CommonManifestError, match="Pair ID collision"):
        ensure_unique_pair_ids(pairs, CommonManifestError)


def test_write_validated_manifest_does_not_replace_target_when_validation_fails(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_bytes(b"existing bytes\n")

    def fail_validation(candidate_path):
        assert candidate_path.is_file()
        raise CommonManifestError("forced validation failure")

    with pytest.raises(CommonManifestError, match="forced validation failure"):
        write_validated_manifest(
            [DummyPair(pair_id="pair-1", path=tmp_path / "image.png")],
            manifest_path,
            fail_validation,
            CommonManifestError,
        )

    assert manifest_path.read_bytes() == b"existing bytes\n"
    assert list(tmp_path.glob(".manifest.csv.*.tmp")) == []


def test_completeness_reports_missing_and_extra_identities():
    expected = {("sd300b", "00001000", 1)}
    actual = {("sd300b", "00001000", 2)}

    with pytest.raises(CommonManifestError, match="missing expected anatomical identities"):
        validate_completeness(expected, actual, CommonManifestError)

    with pytest.raises(CommonManifestError, match="unexpected anatomical identities"):
        validate_completeness(set(), actual, CommonManifestError)


def test_identity_formatting_is_sorted_and_limited():
    identities = {
        ("sd300b", f"{subject_id:08d}", 1)
        for subject_id in range(12)
    }

    text = format_identities(identities)

    assert text.startswith("(sd300b, 00000000, 01), (sd300b, 00000001, 01)")
    assert text.endswith("... and 2 more")
