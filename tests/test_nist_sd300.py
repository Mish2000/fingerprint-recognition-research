from pathlib import Path

import pytest

from fingerprint_data_discovery.nist_sd300 import (
    DATASETS,
    ImageRecord,
    ScanResult,
    SchemaValidationError,
    parse_image_filename,
    validate_image_path,
)


def test_parse_valid_plain_filename():
    parsed = parse_image_filename("00001000_plain_1000_13.png")

    assert parsed.subject_id == "00001000"
    assert parsed.impression_type == "plain"
    assert parsed.ppi == 1000
    assert parsed.frgp == 13
    assert parsed.finger_position == "plain_right_four_fingers"


def test_parse_valid_roll_filename():
    parsed = parse_image_filename("00001000_roll_2000_01.png")

    assert parsed.subject_id == "00001000"
    assert parsed.impression_type == "roll"
    assert parsed.ppi == 2000
    assert parsed.frgp == 1
    assert parsed.finger_position == "right_thumb"


def test_parse_rejects_non_schema_filename():
    with pytest.raises(SchemaValidationError, match="Expected SUBJECT_IMPRESSION_PPI_FRGP"):
        parse_image_filename("00001000_rolled_1000_01.png")


def test_validate_rejects_ppi_mismatch():
    path = Path("C:/fingerprint-datasets/NIST/sd300b/images/1000/png/roll/00001000_roll_2000_01.png")

    with pytest.raises(SchemaValidationError, match="does not match dataset sd300b PPI 1000"):
        validate_image_path(path, DATASETS["sd300b"], "roll")


def test_validate_rejects_impression_directory_mismatch():
    path = Path("C:/fingerprint-datasets/NIST/sd300b/images/1000/png/plain/00001000_roll_1000_01.png")

    with pytest.raises(SchemaValidationError, match="does not match directory 'plain'"):
        validate_image_path(path, DATASETS["sd300b"], "plain")


def test_validate_rejects_invalid_frgp_for_roll():
    path = Path("C:/fingerprint-datasets/NIST/sd300b/images/1000/png/roll/00001000_roll_1000_13.png")

    with pytest.raises(SchemaValidationError, match="not valid for 'roll'"):
        validate_image_path(path, DATASETS["sd300b"], "roll")


def test_audit_summary_reports_canonical_counts_without_rewriting_raw_fields():
    records = [
        ImageRecord(
            dataset="sd300b",
            subject_id="00001000",
            impression_type="plain",
            ppi=1000,
            frgp=11,
            finger_position="plain_right_thumb",
            absolute_path=Path("C:/fingerprint-datasets/example_plain_11.png"),
        ),
        ImageRecord(
            dataset="sd300b",
            subject_id="00001000",
            impression_type="plain",
            ppi=1000,
            frgp=13,
            finger_position="plain_right_four_fingers",
            absolute_path=Path("C:/fingerprint-datasets/example_plain_13.png"),
        ),
        ImageRecord(
            dataset="sd300b",
            subject_id="00001000",
            impression_type="roll",
            ppi=1000,
            frgp=1,
            finger_position="right_thumb",
            absolute_path=Path("C:/fingerprint-datasets/example_roll_01.png"),
        ),
    ]

    summary = ScanResult(records=records, errors=[]).audit_summary()

    assert summary["finger_positions"]["plain_right_thumb"] == 1
    assert summary["finger_positions"]["right_thumb"] == 1
    assert summary["canonical_single_finger_plain"]["01"] == 1
    assert summary["canonical_single_finger_roll"]["01"] == 1
    assert summary["plain_multi_finger_captures_not_pairable"] == 1
