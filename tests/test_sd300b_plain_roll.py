from pathlib import Path
import subprocess
import sys

import pytest

import fingerprint_data_discovery.plain_roll_protocol as plain_roll_protocol
import fingerprint_data_discovery.sd300b_plain_roll as plain_roll
from fingerprint_data_discovery.canonical_fingers import CanonicalFingerMappingError
from fingerprint_data_discovery.nist_sd300 import ImageRecord, ScanError, ScanResult
from fingerprint_data_discovery.sd300b_plain_roll import (
    DATASET,
    PROTOCOL,
    ManifestGenerationError,
    ManifestValidationError,
    PlainRollPair,
    build_plain_roll_pairs,
    generate_manifest,
    make_pair_id,
    validate_manifest,
    write_manifest_atomic,
)


def test_builds_valid_plain_roll_pair():
    plain = _record("plain", subject_id="00001000", frgp=11)
    roll = _record("roll", subject_id="00001000", frgp=1)

    pairs = build_plain_roll_pairs(ScanResult(records=[roll, plain], errors=[]))

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.pair_id == "sd300b_plain_roll_00001000_01"
    assert pair.dataset == DATASET
    assert pair.protocol == PROTOCOL
    assert pair.subject_id == "00001000"
    assert pair.canonical_finger_position == 1
    assert pair.raw_frgp_a == 11
    assert pair.raw_frgp_b == 1
    assert pair.path_a == plain.absolute_path
    assert pair.path_b == roll.absolute_path


def test_right_thumb_pairs_plain_11_to_roll_1_as_canonical_1():
    pairs = build_plain_roll_pairs(
        ScanResult(
            records=[
                _record("plain", subject_id="00001000", frgp=11),
                _record("roll", subject_id="00001000", frgp=1),
            ],
            errors=[],
        )
    )

    assert [(pair.raw_frgp_a, pair.raw_frgp_b, pair.canonical_finger_position) for pair in pairs] == [
        (11, 1, 1)
    ]


def test_left_thumb_pairs_plain_12_to_roll_6_as_canonical_6():
    pairs = build_plain_roll_pairs(
        ScanResult(
            records=[
                _record("plain", subject_id="00001000", frgp=12),
                _record("roll", subject_id="00001000", frgp=6),
            ],
            errors=[],
        )
    )

    assert [(pair.raw_frgp_a, pair.raw_frgp_b, pair.canonical_finger_position) for pair in pairs] == [
        (12, 6, 6)
    ]


@pytest.mark.parametrize("frgp", [2, 3, 4, 5, 7, 8, 9, 10])
def test_non_thumb_plain_and_roll_direct_mappings(frgp):
    pairs = build_plain_roll_pairs(
        ScanResult(
            records=[
                _record("plain", subject_id="00001000", frgp=frgp),
                _record("roll", subject_id="00001000", frgp=frgp),
            ],
            errors=[],
        )
    )

    assert [(pair.raw_frgp_a, pair.raw_frgp_b, pair.canonical_finger_position) for pair in pairs] == [
        (frgp, frgp, frgp)
    ]


def test_plain_multi_finger_frgp_13_and_14_do_not_participate():
    records = [
        _record("plain", subject_id="00001000", frgp=13),
        _record("plain", subject_id="00001000", frgp=14),
        _record("plain", subject_id="00001000", frgp=2),
        _record("roll", subject_id="00001000", frgp=2),
        _record("roll", subject_id="00001000", frgp=3),
        _record("roll", subject_id="00001000", frgp=4),
    ]

    pairs = build_plain_roll_pairs(ScanResult(records=records, errors=[]))

    assert [(pair.canonical_finger_position, pair.raw_frgp_a, pair.raw_frgp_b) for pair in pairs] == [
        (2, 2, 2)
    ]


def test_plain_only_identity_does_not_create_pair():
    pairs = build_plain_roll_pairs(
        ScanResult(records=[_record("plain", subject_id="00001000", frgp=11)], errors=[])
    )

    assert pairs == []


def test_roll_only_identity_does_not_create_pair():
    pairs = build_plain_roll_pairs(
        ScanResult(records=[_record("roll", subject_id="00001000", frgp=1)], errors=[])
    )

    assert pairs == []


def test_pairs_are_created_exactly_from_identity_intersection():
    records = [
        _record("plain", subject_id="00001000", frgp=11),
        _record("roll", subject_id="00001000", frgp=1),
        _record("plain", subject_id="00001000", frgp=2),
        _record("roll", subject_id="00001000", frgp=3),
    ]

    pairs = build_plain_roll_pairs(ScanResult(records=records, errors=[]))

    assert [(pair.subject_id, pair.canonical_finger_position) for pair in pairs] == [
        ("00001000", 1)
    ]


def test_duplicate_plain_identity_is_rejected():
    records = [
        _record("plain", subject_id="00001000", frgp=11, suffix="a"),
        _record("plain", subject_id="00001000", frgp=11, suffix="b"),
        _record("roll", subject_id="00001000", frgp=1),
    ]

    with pytest.raises(ManifestGenerationError, match="Duplicate plain anatomical identities"):
        build_plain_roll_pairs(ScanResult(records=records, errors=[]))


def test_duplicate_roll_identity_is_rejected():
    records = [
        _record("plain", subject_id="00001000", frgp=11),
        _record("roll", subject_id="00001000", frgp=1, suffix="a"),
        _record("roll", subject_id="00001000", frgp=1, suffix="b"),
    ]

    with pytest.raises(ManifestGenerationError, match="Duplicate roll anatomical identities"):
        build_plain_roll_pairs(ScanResult(records=records, errors=[]))


def test_rejects_scan_errors_before_building_manifest():
    scan_result = ScanResult(
        records=[],
        errors=[ScanError(path=Path("C:/fingerprint-datasets/bad.png"), message="bad file")],
    )

    with pytest.raises(ManifestGenerationError, match="SD300b scan has 1 error"):
        build_plain_roll_pairs(scan_result)


def test_input_order_does_not_affect_pair_ids_or_sorting():
    records = [
        _record("roll", subject_id="00001002", frgp=7),
        _record("plain", subject_id="00001001", frgp=10),
        _record("plain", subject_id="00001002", frgp=7),
        _record("roll", subject_id="00001001", frgp=10),
        _record("roll", subject_id="00001001", frgp=1),
        _record("plain", subject_id="00001001", frgp=11),
    ]

    first = build_plain_roll_pairs(ScanResult(records=records, errors=[]))
    second = build_plain_roll_pairs(ScanResult(records=list(reversed(records)), errors=[]))

    expected = [
        ("sd300b_plain_roll_00001001_01", "00001001", 1),
        ("sd300b_plain_roll_00001001_10", "00001001", 10),
        ("sd300b_plain_roll_00001002_07", "00001002", 7),
    ]
    assert [(pair.pair_id, pair.subject_id, pair.canonical_finger_position) for pair in first] == expected
    assert [(pair.pair_id, pair.subject_id, pair.canonical_finger_position) for pair in second] == expected


def test_validator_accepts_complete_manifest(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=1, plain_path=plain_path, roll_path=roll_path)],
        manifest_path,
    )
    report = validate_manifest(manifest_path, data_root)

    assert report.row_count == 1
    assert report.eligible_plain_identity_count == 1
    assert report.eligible_roll_identity_count == 1
    assert report.expected_intersection_count == 1
    assert report.actual_identity_count == 1
    assert report.plain_only_identity_count == 0
    assert report.roll_only_identity_count == 0
    assert report.canonical_finger_counts[1] == 1


def test_validator_rejects_missing_intersection_identity(tmp_path):
    data_root = _make_data_root(tmp_path)
    first_plain = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    first_roll = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    _touch_plain_image(data_root, subject_id="00001000", frgp=2)
    _touch_roll_image(data_root, subject_id="00001000", frgp=2)
    manifest_path = tmp_path / "plain_roll.csv"

    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=1, plain_path=first_plain, roll_path=first_roll)],
        manifest_path,
    )

    with pytest.raises(ManifestValidationError, match="missing expected anatomical identities"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_extra_identity(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"
    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=1, plain_path=plain_path, roll_path=roll_path)],
        manifest_path,
    )

    monkeypatch.setattr(plain_roll_protocol, "_expected_records_by_identity", lambda *args: ({}, {}))

    with pytest.raises(ManifestValidationError, match="unexpected anatomical identities"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_swapped_a_b_paths(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"
    pair = _pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=1, plain_path=roll_path, roll_path=plain_path)

    write_manifest_atomic([pair], manifest_path)

    with pytest.raises(ManifestValidationError, match="path_a .* not under the SD300b plain directory"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_wrong_plain_source_path(tmp_path):
    data_root = _make_data_root(tmp_path)
    expected_plain = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    wrong_plain = _touch_plain_image(data_root, subject_id="00001001", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    assert expected_plain != wrong_plain
    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=1, plain_path=wrong_plain, roll_path=roll_path)],
        manifest_path,
    )

    with pytest.raises(ManifestValidationError, match="Subject mismatch for path_a|wrong plain source path"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_wrong_roll_source_path(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    expected_roll = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    wrong_roll = _touch_roll_image(data_root, subject_id="00001001", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    assert expected_roll != wrong_roll
    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=1, plain_path=plain_path, roll_path=wrong_roll)],
        manifest_path,
    )

    with pytest.raises(ManifestValidationError, match="Subject mismatch for path_b|wrong roll source path"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_subject_mismatch_between_plain_and_roll(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001001", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=1, plain_path=plain_path, roll_path=roll_path)],
        manifest_path,
    )

    with pytest.raises(ManifestValidationError, match="Subject mismatch for path_b"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_canonical_mismatch(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    write_manifest_atomic(
        [_pair("00001000", canonical_position=2, plain_frgp=11, roll_frgp=1, plain_path=plain_path, roll_path=roll_path)],
        manifest_path,
    )

    with pytest.raises(ManifestValidationError, match="Canonical mapping mismatch"):
        validate_manifest(manifest_path, data_root)


def test_validator_wraps_invalid_plain_frgp_as_manifest_validation_error(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=1, roll_frgp=1, plain_path=plain_path, roll_path=roll_path)],
        manifest_path,
    )

    with pytest.raises(
        ManifestValidationError,
        match="raw FRGP is not valid for plain at CSV line 2",
    ) as exc_info:
        validate_manifest(manifest_path, data_root)
    assert isinstance(exc_info.value.__cause__, CanonicalFingerMappingError)


def test_validator_wraps_invalid_roll_frgp_as_manifest_validation_error(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=11, roll_frgp=11, plain_path=plain_path, roll_path=roll_path)],
        manifest_path,
    )

    with pytest.raises(
        ManifestValidationError,
        match="raw FRGP is not valid for roll at CSV line 2",
    ) as exc_info:
        validate_manifest(manifest_path, data_root)
    assert isinstance(exc_info.value.__cause__, CanonicalFingerMappingError)


def test_generation_validation_failure_does_not_replace_existing_manifest(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path)
    _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"
    original_bytes = b"existing manifest bytes\n"
    manifest_path.write_bytes(original_bytes)

    def fail_validation(manifest_path, data_root):
        raise ManifestValidationError("forced validation failure")

    monkeypatch.setattr(plain_roll, "validate_manifest", fail_validation)

    with pytest.raises(ManifestValidationError, match="forced validation failure"):
        generate_manifest(data_root, manifest_path)

    assert manifest_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".plain_roll.csv.*.tmp")) == []


def test_repeated_generation_on_same_data_produces_identical_bytes(tmp_path):
    data_root = _make_data_root(tmp_path)
    _touch_plain_image(data_root, subject_id="00001002", frgp=7)
    _touch_roll_image(data_root, subject_id="00001002", frgp=7)
    _touch_plain_image(data_root, subject_id="00001001", frgp=11)
    _touch_roll_image(data_root, subject_id="00001001", frgp=1)
    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"

    generate_manifest(data_root, first_manifest)
    generate_manifest(data_root, second_manifest)

    assert first_manifest.read_bytes() == second_manifest.read_bytes()


def test_plain_roll_cli_reports_invalid_manifest_without_traceback(tmp_path):
    data_root = _make_data_root(tmp_path)
    plain_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    roll_path = _touch_roll_image(data_root, subject_id="00001000", frgp=1)
    manifest_path = tmp_path / "plain_roll.csv"

    write_manifest_atomic(
        [_pair("00001000", canonical_position=1, plain_frgp=1, roll_frgp=1, plain_path=plain_path, roll_path=roll_path)],
        manifest_path,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300b_plain_roll",
            "validate",
            "--data-root",
            str(data_root),
            "--manifest",
            str(manifest_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("Error:")
    assert "raw FRGP is not valid for plain at CSV line 2" in result.stderr
    assert "Traceback" not in result.stderr


def _record(impression_type: str, subject_id: str, frgp: int, suffix: str = "") -> ImageRecord:
    stem_suffix = f"_{suffix}" if suffix else ""
    return ImageRecord(
        dataset=DATASET,
        subject_id=subject_id,
        impression_type=impression_type,
        ppi=1000,
        frgp=frgp,
        finger_position=f"test_{impression_type}_{frgp}",
        absolute_path=Path(
            f"C:/fingerprint-datasets/{subject_id}_{impression_type}_1000_{frgp:02d}{stem_suffix}.png"
        ),
    )


def _make_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "fingerprint-datasets"
    (data_root / "NIST" / "sd300b" / "images" / "1000" / "png" / "plain").mkdir(
        parents=True
    )
    (data_root / "NIST" / "sd300b" / "images" / "1000" / "png" / "roll").mkdir(
        parents=True
    )
    return data_root


def _touch_plain_image(data_root: Path, subject_id: str, frgp: int) -> Path:
    path = (
        data_root
        / "NIST"
        / "sd300b"
        / "images"
        / "1000"
        / "png"
        / "plain"
        / f"{subject_id}_plain_1000_{frgp:02d}.png"
    )
    path.write_bytes(b"not a real image; filename validation only")
    return path


def _touch_roll_image(data_root: Path, subject_id: str, frgp: int) -> Path:
    path = (
        data_root
        / "NIST"
        / "sd300b"
        / "images"
        / "1000"
        / "png"
        / "roll"
        / f"{subject_id}_roll_1000_{frgp:02d}.png"
    )
    path.write_bytes(b"not a real image; filename validation only")
    return path


def _pair(
    subject_id: str,
    canonical_position: int,
    plain_frgp: int,
    roll_frgp: int,
    plain_path: Path,
    roll_path: Path,
) -> PlainRollPair:
    return PlainRollPair(
        pair_id=make_pair_id(subject_id, canonical_position),
        dataset=DATASET,
        protocol=PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=1000,
        raw_frgp_a=plain_frgp,
        raw_frgp_b=roll_frgp,
        path_a=plain_path,
        path_b=roll_path,
    )
