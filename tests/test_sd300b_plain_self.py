from pathlib import Path
import subprocess
import sys

import pytest

import fingerprint_data_discovery.sd300b_plain_self as plain_self
from fingerprint_data_discovery.canonical_fingers import CanonicalFingerMappingError
from fingerprint_data_discovery.nist_sd300 import ImageRecord, ScanError, ScanResult
from fingerprint_data_discovery.sd300b_plain_self import (
    DATASET,
    PROTOCOL,
    ManifestGenerationError,
    ManifestValidationError,
    PlainSelfPair,
    build_plain_self_pairs,
    generate_manifest,
    make_pair_id,
    validate_manifest,
    write_manifest_atomic,
)


def test_builds_valid_plain_self_pair():
    record = _record(subject_id="00001000", frgp=11)

    pairs = build_plain_self_pairs(ScanResult(records=[record], errors=[]))

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.pair_id == "sd300b_plain_self_00001000_01"
    assert pair.dataset == DATASET
    assert pair.protocol == PROTOCOL
    assert pair.subject_id == "00001000"
    assert pair.canonical_finger_position == 1
    assert pair.raw_frgp_a == 11
    assert pair.raw_frgp_b == 11
    assert pair.path_a == pair.path_b == record.absolute_path


def test_excludes_plain_multi_finger_frgp_13_and_14():
    records = [
        _record(subject_id="00001000", frgp=13),
        _record(subject_id="00001000", frgp=14),
        _record(subject_id="00001000", frgp=2),
    ]

    pairs = build_plain_self_pairs(ScanResult(records=records, errors=[]))

    assert len(pairs) == 1
    assert pairs[0].canonical_finger_position == 2
    assert pairs[0].raw_frgp_a == 2


def test_rejects_duplicate_anatomical_identity():
    records = [
        _record(subject_id="00001000", frgp=11, suffix="a"),
        _record(subject_id="00001000", frgp=11, suffix="b"),
    ]

    with pytest.raises(ManifestGenerationError, match="Duplicate plain single-finger"):
        build_plain_self_pairs(ScanResult(records=records, errors=[]))


def test_rejects_scan_errors_before_building_manifest():
    scan_result = ScanResult(
        records=[],
        errors=[ScanError(path=Path("C:/fingerprint-datasets/bad.png"), message="bad file")],
    )

    with pytest.raises(ManifestGenerationError, match="SD300b scan has 1 error"):
        build_plain_self_pairs(scan_result)


def test_pair_id_determinism_is_independent_of_input_order():
    records = [
        _record(subject_id="00001002", frgp=7),
        _record(subject_id="00001001", frgp=11),
    ]

    first = build_plain_self_pairs(ScanResult(records=records, errors=[]))
    second = build_plain_self_pairs(ScanResult(records=list(reversed(records)), errors=[]))

    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]


def test_sorting_determinism_is_by_subject_then_canonical_position():
    records = [
        _record(subject_id="00001002", frgp=7),
        _record(subject_id="00001001", frgp=10),
        _record(subject_id="00001001", frgp=11),
    ]

    pairs = build_plain_self_pairs(ScanResult(records=records, errors=[]))

    assert [(pair.subject_id, pair.canonical_finger_position) for pair in pairs] == [
        ("00001001", 1),
        ("00001001", 10),
        ("00001002", 7),
    ]


def test_validator_accepts_complete_manifest(tmp_path):
    data_root = _make_data_root(tmp_path)
    first_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    second_path = _touch_plain_image(data_root, subject_id="00001000", frgp=2)
    _touch_plain_image(data_root, subject_id="00001000", frgp=13)
    manifest_path = tmp_path / "plain_self.csv"
    pairs = [
        _pair(subject_id="00001000", canonical_position=1, frgp=11, path=first_path),
        _pair(subject_id="00001000", canonical_position=2, frgp=2, path=second_path),
    ]

    write_manifest_atomic(pairs, manifest_path)
    report = validate_manifest(manifest_path, data_root)

    assert report.row_count == 2
    assert report.expected_identity_count == 2
    assert report.actual_identity_count == 2
    assert report.canonical_finger_counts[1] == 1
    assert report.canonical_finger_counts[2] == 1


def test_validator_rejects_manifest_missing_eligible_identity(tmp_path):
    data_root = _make_data_root(tmp_path)
    first_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    _touch_plain_image(data_root, subject_id="00001000", frgp=2)
    manifest_path = tmp_path / "plain_self.csv"

    write_manifest_atomic(
        [_pair(subject_id="00001000", canonical_position=1, frgp=11, path=first_path)],
        manifest_path,
    )

    with pytest.raises(ManifestValidationError, match="missing expected anatomical identities"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_manifest_with_extra_identity(tmp_path):
    data_root = _make_data_root(tmp_path)
    image_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    manifest_path = tmp_path / "plain_self.csv"
    pairs = [
        _pair(subject_id="00001000", canonical_position=1, frgp=11, path=image_path),
        _pair(subject_id="00001001", canonical_position=1, frgp=11, path=image_path),
    ]

    write_manifest_atomic(pairs, manifest_path)

    with pytest.raises(ManifestValidationError, match="unexpected anatomical identities"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_correct_identity_with_wrong_source_path(tmp_path):
    data_root = _make_data_root(tmp_path)
    first_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    second_path = _touch_plain_image(data_root, subject_id="00001000", frgp=2)
    manifest_path = tmp_path / "plain_self.csv"
    pairs = [
        _pair(subject_id="00001000", canonical_position=1, frgp=11, path=second_path),
        _pair(subject_id="00001000", canonical_position=2, frgp=2, path=second_path),
    ]

    assert first_path != second_path
    write_manifest_atomic(pairs, manifest_path)

    with pytest.raises(ManifestValidationError, match="wrong source path"):
        validate_manifest(manifest_path, data_root)


def test_validator_rejects_corrupt_manifest(tmp_path):
    data_root = _make_data_root(tmp_path)
    image_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    manifest_path = tmp_path / "plain_self.csv"
    pair = _pair(subject_id="00001000", canonical_position=2, frgp=11, path=image_path)

    write_manifest_atomic([pair], manifest_path)

    with pytest.raises(ManifestValidationError, match="Canonical mapping mismatch"):
        validate_manifest(manifest_path, data_root)


def test_validator_wraps_invalid_plain_raw_frgp_as_manifest_validation_error(tmp_path):
    data_root = _make_data_root(tmp_path)
    image_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    manifest_path = tmp_path / "plain_self.csv"
    pair = _pair(subject_id="00001000", canonical_position=1, frgp=1, path=image_path)

    write_manifest_atomic([pair], manifest_path)

    with pytest.raises(
        ManifestValidationError,
        match="raw FRGP is not valid for plain at CSV line 2",
    ) as exc_info:
        validate_manifest(manifest_path, data_root)

    assert isinstance(exc_info.value.__cause__, CanonicalFingerMappingError)


def test_plain_self_cli_reports_invalid_plain_raw_frgp_without_traceback(tmp_path):
    data_root = _make_data_root(tmp_path)
    image_path = _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    manifest_path = tmp_path / "plain_self.csv"
    pair = _pair(subject_id="00001000", canonical_position=1, frgp=1, path=image_path)

    write_manifest_atomic([pair], manifest_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300b_plain_self",
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


def test_generation_validation_failure_does_not_replace_existing_manifest(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path)
    _touch_plain_image(data_root, subject_id="00001000", frgp=11)
    manifest_path = tmp_path / "plain_self.csv"
    original_bytes = b"existing manifest bytes\n"
    manifest_path.write_bytes(original_bytes)

    def fail_validation(manifest_path, data_root):
        raise ManifestValidationError("forced validation failure")

    monkeypatch.setattr(plain_self, "validate_manifest", fail_validation)

    with pytest.raises(ManifestValidationError, match="forced validation failure"):
        generate_manifest(data_root, manifest_path)

    assert manifest_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".plain_self.csv.*.tmp")) == []


def test_repeated_generation_on_same_data_produces_identical_bytes(tmp_path):
    data_root = _make_data_root(tmp_path)
    _touch_plain_image(data_root, subject_id="00001002", frgp=7)
    _touch_plain_image(data_root, subject_id="00001001", frgp=11)
    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"

    generate_manifest(data_root, first_manifest)
    generate_manifest(data_root, second_manifest)

    assert first_manifest.read_bytes() == second_manifest.read_bytes()


def _record(subject_id: str, frgp: int, suffix: str = "") -> ImageRecord:
    stem_suffix = f"_{suffix}" if suffix else ""
    return ImageRecord(
        dataset=DATASET,
        subject_id=subject_id,
        impression_type="plain",
        ppi=1000,
        frgp=frgp,
        finger_position=f"test_frgp_{frgp}",
        absolute_path=Path(f"C:/fingerprint-datasets/{subject_id}_plain_1000_{frgp:02d}{stem_suffix}.png"),
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


def _pair(subject_id: str, canonical_position: int, frgp: int, path: Path) -> PlainSelfPair:
    return PlainSelfPair(
        pair_id=make_pair_id(subject_id, canonical_position),
        dataset=DATASET,
        protocol=PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=1000,
        raw_frgp_a=frgp,
        raw_frgp_b=frgp,
        path_a=path,
        path_b=path,
    )
