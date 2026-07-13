import csv
from pathlib import Path
import subprocess
import sys

import pytest

from fingerprint_data_discovery.nist_sd300 import ImageRecord, ScanResult
from fingerprint_data_discovery.protocol_dataset import SD300B_CONTEXT, SD300C_CONTEXT
from fingerprint_data_discovery.self_manifest_common import MANIFEST_COLUMNS
import fingerprint_data_discovery.plain_roll_protocol as plain_roll_protocol
import fingerprint_data_discovery.roll_self_protocol as roll_self_protocol
import fingerprint_data_discovery.sd300b_plain_roll as sd300b_plain_roll
import fingerprint_data_discovery.sd300b_roll_self as sd300b_roll_self
import fingerprint_data_discovery.sd300c_plain_roll as sd300c_plain_roll
import fingerprint_data_discovery.sd300c_plain_self as sd300c_plain_self
import fingerprint_data_discovery.sd300c_protocol_audit as sd300c_protocol_audit
import fingerprint_data_discovery.sd300c_roll_self as sd300c_roll_self


ROLL_CONTEXT_CASES = [
    (SD300B_CONTEXT, sd300b_roll_self),
    (SD300C_CONTEXT, sd300c_roll_self),
]

PLAIN_ROLL_CONTEXT_CASES = [
    (SD300B_CONTEXT, sd300b_plain_roll),
    (SD300C_CONTEXT, sd300c_plain_roll),
]


def test_sd300b_and_sd300c_wrappers_use_separate_contexts_and_paths():
    assert sd300b_roll_self.DATASET_CONTEXT is SD300B_CONTEXT
    assert sd300b_plain_roll.DATASET_CONTEXT is SD300B_CONTEXT
    assert sd300c_roll_self.DATASET_CONTEXT is SD300C_CONTEXT
    assert sd300c_plain_roll.DATASET_CONTEXT is SD300C_CONTEXT
    assert sd300c_roll_self.DEFAULT_MANIFEST_PATH == Path("protocols") / "sd300c" / "roll_self.csv"
    assert sd300c_plain_roll.DEFAULT_MANIFEST_PATH == Path("protocols") / "sd300c" / "plain_roll.csv"
    assert sd300c_roll_self.make_pair_id("00001000", 1).startswith("sd300c_roll_self_")
    assert sd300c_plain_roll.make_pair_id("00001000", 1).startswith("sd300c_plain_roll_")


@pytest.mark.parametrize(("context", "module"), ROLL_CONTEXT_CASES)
def test_roll_self_all_canonical_positions_work_in_both_contexts(context, module):
    records = [
        _record(context, "roll", subject_id=f"000010{frgp:02d}", frgp=frgp)
        for frgp in range(1, 11)
    ]

    pairs = module.build_roll_self_pairs(ScanResult(records=list(reversed(records)), errors=[]))

    assert [pair.canonical_finger_position for pair in pairs] == list(range(1, 11))
    assert all(pair.pair_id.startswith(f"{context.name}_roll_self_") for pair in pairs)
    assert all(pair.ppi == context.expected_ppi for pair in pairs)


@pytest.mark.parametrize(("context", "module"), ROLL_CONTEXT_CASES)
def test_roll_self_duplicate_identity_is_rejected_in_both_contexts(context, module):
    records = [
        _record(context, "roll", "00001000", 1, "a"),
        _record(context, "roll", "00001000", 1, "b"),
    ]

    with pytest.raises(module.ManifestGenerationError, match="Duplicate roll anatomical identities"):
        module.build_roll_self_pairs(ScanResult(records=records, errors=[]))


def test_roll_self_validation_rejects_wrong_dataset_and_wrong_ppi(tmp_path):
    data_root = _make_data_root(tmp_path, SD300B_CONTEXT, SD300C_CONTEXT)
    sd300b_path = _touch_image(data_root, SD300B_CONTEXT, "roll", "00001000", 1)
    _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    manifest_path = tmp_path / "roll_self.csv"
    sd300b_roll_self.write_manifest_atomic(
        [_roll_pair(sd300b_roll_self, "00001000", 1, 1, sd300b_path)],
        manifest_path,
    )

    with pytest.raises(sd300c_roll_self.ManifestValidationError, match="Invalid dataset"):
        sd300c_roll_self.validate_manifest(manifest_path, data_root)

    sd300c_path = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001001", 1)
    sd300c_roll_self.write_manifest_atomic(
        [_roll_pair(sd300c_roll_self, "00001001", 1, 1, sd300c_path, ppi=1000)],
        manifest_path,
    )
    with pytest.raises(sd300c_roll_self.ManifestValidationError, match="PPI mismatch"):
        sd300c_roll_self.validate_manifest(manifest_path, data_root)


def test_roll_self_missing_extra_determinism_and_failure_safety(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    first_path = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 2)
    manifest_path = tmp_path / "roll_self.csv"
    sd300c_roll_self.write_manifest_atomic(
        [_roll_pair(sd300c_roll_self, "00001000", 1, 1, first_path)],
        manifest_path,
    )
    with pytest.raises(sd300c_roll_self.ManifestValidationError, match="missing expected anatomical identities"):
        sd300c_roll_self.validate_manifest(manifest_path, data_root)

    monkeypatch.setattr(roll_self_protocol, "_expected_pairs_by_identity", lambda *args: {})
    with pytest.raises(sd300c_roll_self.ManifestValidationError, match="unexpected anatomical identities"):
        sd300c_roll_self.validate_manifest(manifest_path, data_root)
    monkeypatch.undo()

    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"
    sd300c_roll_self.generate_manifest(data_root, first_manifest)
    sd300c_roll_self.generate_manifest(data_root, second_manifest)
    assert first_manifest.read_bytes() == second_manifest.read_bytes()

    original_bytes = b"existing manifest bytes\n"
    manifest_path.write_bytes(original_bytes)

    def fail_validation(manifest_path, data_root):
        raise sd300c_roll_self.ManifestValidationError("forced validation failure")

    monkeypatch.setattr(sd300c_roll_self, "validate_manifest", fail_validation)
    with pytest.raises(sd300c_roll_self.ManifestValidationError, match="forced validation failure"):
        sd300c_roll_self.generate_manifest(data_root, manifest_path)
    assert manifest_path.read_bytes() == original_bytes


@pytest.mark.parametrize(("context", "module"), PLAIN_ROLL_CONTEXT_CASES)
def test_plain_roll_intersection_thumb_and_multifinger_semantics(context, module):
    records = [
        _record(context, "plain", "00001000", 11),
        _record(context, "roll", "00001000", 1),
        _record(context, "plain", "00001000", 12),
        _record(context, "roll", "00001000", 6),
        _record(context, "plain", "00001001", 13),
        _record(context, "plain", "00001001", 14),
        _record(context, "roll", "00001001", 2),
        _record(context, "plain", "00001002", 3),
    ]

    pairs = module.build_plain_roll_pairs(ScanResult(records=list(reversed(records)), errors=[]))

    assert [
        (pair.subject_id, pair.canonical_finger_position, pair.raw_frgp_a, pair.raw_frgp_b)
        for pair in pairs
    ] == [
        ("00001000", 1, 11, 1),
        ("00001000", 6, 12, 6),
    ]


@pytest.mark.parametrize(("context", "module"), PLAIN_ROLL_CONTEXT_CASES)
def test_plain_roll_duplicate_identities_are_rejected_in_both_contexts(context, module):
    duplicate_plain = [
        _record(context, "plain", "00001000", 11, "a"),
        _record(context, "plain", "00001000", 11, "b"),
        _record(context, "roll", "00001000", 1),
    ]
    duplicate_roll = [
        _record(context, "plain", "00001000", 11),
        _record(context, "roll", "00001000", 1, "a"),
        _record(context, "roll", "00001000", 1, "b"),
    ]

    with pytest.raises(module.ManifestGenerationError, match="Duplicate plain anatomical identities"):
        module.build_plain_roll_pairs(ScanResult(records=duplicate_plain, errors=[]))
    with pytest.raises(module.ManifestGenerationError, match="Duplicate roll anatomical identities"):
        module.build_plain_roll_pairs(ScanResult(records=duplicate_roll, errors=[]))


def test_plain_roll_validation_rejects_wrong_dataset_orientation_frgp_and_canonical(tmp_path):
    data_root = _make_data_root(tmp_path, SD300B_CONTEXT, SD300C_CONTEXT)
    b_plain = _touch_image(data_root, SD300B_CONTEXT, "plain", "00001000", 11)
    b_roll = _touch_image(data_root, SD300B_CONTEXT, "roll", "00001000", 1)
    c_plain = _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 11)
    c_roll = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    manifest_path = tmp_path / "plain_roll.csv"

    sd300b_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300b_plain_roll, "00001000", 1, 11, 1, b_plain, b_roll)],
        manifest_path,
    )
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="Invalid dataset"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)

    sd300c_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300c_plain_roll, "00001000", 1, 11, 1, c_roll, c_plain)],
        manifest_path,
    )
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="path_a .* plain directory"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)

    sd300c_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300c_plain_roll, "00001000", 1, 2, 1, c_plain, c_roll)],
        manifest_path,
    )
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="Canonical mapping mismatch"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)

    sd300c_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300c_plain_roll, "00001000", 2, 11, 1, c_plain, c_roll)],
        manifest_path,
    )
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="Canonical mapping mismatch"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)


def test_plain_roll_completeness_determinism_and_failure_safety(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    plain_one = _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 11)
    roll_one = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 2)
    _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 2)
    manifest_path = tmp_path / "plain_roll.csv"
    sd300c_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300c_plain_roll, "00001000", 1, 11, 1, plain_one, roll_one)],
        manifest_path,
    )

    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="missing expected anatomical identities"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)

    monkeypatch.setattr(plain_roll_protocol, "_expected_records_by_identity", lambda *args: ({}, {}))
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="unexpected anatomical identities"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)
    monkeypatch.undo()

    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"
    sd300c_plain_roll.generate_manifest(data_root, first_manifest)
    sd300c_plain_roll.generate_manifest(data_root, second_manifest)
    assert first_manifest.read_bytes() == second_manifest.read_bytes()

    original_bytes = b"existing manifest bytes\n"
    manifest_path.write_bytes(original_bytes)

    def fail_validation(manifest_path, data_root):
        raise sd300c_plain_roll.ManifestValidationError("forced validation failure")

    monkeypatch.setattr(sd300c_plain_roll, "validate_manifest", fail_validation)
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="forced validation failure"):
        sd300c_plain_roll.generate_manifest(data_root, manifest_path)
    assert manifest_path.read_bytes() == original_bytes


def test_sd300c_protocol_audit_consistent_manifests_and_deterministic_json(tmp_path):
    data_root, plain_self_manifest, roll_self_manifest, plain_roll_manifest = _write_consistent_sd300c_manifests(tmp_path)

    report = sd300c_protocol_audit.build_protocol_audit_report(
        data_root,
        plain_self_manifest,
        roll_self_manifest,
        plain_roll_manifest,
    )
    first_report = tmp_path / "first.json"
    second_report = tmp_path / "second.json"
    sd300c_protocol_audit.save_report_json(report, first_report)
    sd300c_protocol_audit.save_report_json(report, second_report)

    assert report.dataset == "sd300c"
    assert report.plain_self_count == 2
    assert report.roll_self_count == 2
    assert report.plain_roll_count == 1
    assert report.expected_intersection_count == 1
    assert report.plain_only_count == 1
    assert report.roll_only_count == 1
    assert report.intersection_consistent is True
    assert first_report.read_bytes() == second_report.read_bytes()


def test_sd300c_protocol_audit_rejects_inconsistencies(tmp_path, monkeypatch):
    data_root, plain_self_manifest, roll_self_manifest, plain_roll_manifest = _write_consistent_sd300c_manifests(tmp_path)
    rows = _read_rows(plain_roll_manifest)
    rows[0]["path_a"], rows[0]["path_b"] = rows[0]["path_b"], rows[0]["path_a"]
    _write_rows(plain_roll_manifest, rows)
    monkeypatch.setattr(sd300c_protocol_audit.sd300c_plain_roll, "validate_manifest", lambda *args: object())

    with pytest.raises(sd300c_protocol_audit.ProtocolAuditError, match="path_a/path_b appear swapped"):
        sd300c_protocol_audit.build_protocol_audit_report(
            data_root,
            plain_self_manifest,
            roll_self_manifest,
            plain_roll_manifest,
        )

    def fail_validator(*args):
        raise sd300c_plain_self.ManifestValidationError("forced validator failure")

    monkeypatch.setattr(sd300c_protocol_audit.sd300c_plain_self, "validate_manifest", fail_validator)
    with pytest.raises(sd300c_protocol_audit.ProtocolAuditError, match="Dedicated validator failed"):
        sd300c_protocol_audit.build_protocol_audit_report(
            data_root,
            plain_self_manifest,
            roll_self_manifest,
            plain_roll_manifest,
        )


def test_sd300c_cli_smoke_and_error_paths(tmp_path):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 11)
    _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    manifest_dir = tmp_path / "manifests"
    plain_self_manifest = manifest_dir / "plain_self.csv"
    roll_self_manifest = manifest_dir / "roll_self.csv"
    plain_roll_manifest = manifest_dir / "plain_roll.csv"
    for module_name, output_path in [
        ("fingerprint_data_discovery.sd300c_plain_self", plain_self_manifest),
        ("fingerprint_data_discovery.sd300c_roll_self", roll_self_manifest),
        ("fingerprint_data_discovery.sd300c_plain_roll", plain_roll_manifest),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                module_name,
                "generate",
                "--data-root",
                str(data_root),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0

    audit_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300c_protocol_audit",
            "--data-root",
            str(data_root),
            "--plain-self-manifest",
            str(plain_self_manifest),
            "--roll-self-manifest",
            str(roll_self_manifest),
            "--plain-roll-manifest",
            str(plain_roll_manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert audit_result.returncode == 0

    bad_rows = _read_rows(plain_roll_manifest)
    bad_rows[0]["raw_frgp_b"] = "2"
    _write_rows(plain_roll_manifest, bad_rows)
    bad_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300c_plain_roll",
            "validate",
            "--data-root",
            str(data_root),
            "--manifest",
            str(plain_roll_manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_result.returncode == 1
    assert bad_result.stderr.startswith("Error:")
    assert "Traceback" not in bad_result.stderr


def _write_consistent_sd300c_manifests(tmp_path: Path):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 11)
    _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    _touch_image(data_root, SD300C_CONTEXT, "plain", "00001001", 2)
    _touch_image(data_root, SD300C_CONTEXT, "roll", "00001002", 3)
    manifest_dir = tmp_path / "manifests"
    plain_self_manifest = manifest_dir / "plain_self.csv"
    roll_self_manifest = manifest_dir / "roll_self.csv"
    plain_roll_manifest = manifest_dir / "plain_roll.csv"
    sd300c_plain_self.generate_manifest(data_root, plain_self_manifest)
    sd300c_roll_self.generate_manifest(data_root, roll_self_manifest)
    sd300c_plain_roll.generate_manifest(data_root, plain_roll_manifest)
    return data_root, plain_self_manifest, roll_self_manifest, plain_roll_manifest


def _record(context, impression_type: str, subject_id: str, frgp: int, suffix: str = "") -> ImageRecord:
    stem_suffix = f"_{suffix}" if suffix else ""
    return ImageRecord(
        dataset=context.name,
        subject_id=subject_id,
        impression_type=impression_type,
        ppi=context.expected_ppi,
        frgp=frgp,
        finger_position=f"test_frgp_{frgp}",
        absolute_path=Path(
            f"C:/fingerprint-datasets/{context.name}_{subject_id}_{impression_type}_{context.expected_ppi}_{frgp:02d}{stem_suffix}.png"
        ),
    )


def _make_data_root(tmp_path: Path, *contexts) -> Path:
    data_root = tmp_path / "fingerprint-datasets"
    for context in contexts:
        (data_root / context.spec.relative_image_root / "plain").mkdir(parents=True, exist_ok=True)
        (data_root / context.spec.relative_image_root / "roll").mkdir(parents=True, exist_ok=True)
    return data_root


def _touch_image(data_root: Path, context, impression_type: str, subject_id: str, frgp: int) -> Path:
    path = (
        data_root
        / context.spec.relative_image_root
        / impression_type
        / f"{subject_id}_{impression_type}_{context.expected_ppi}_{frgp:02d}.png"
    )
    path.write_bytes(b"not a real image; filename validation only")
    return path


def _roll_pair(module, subject_id: str, canonical_position: int, frgp: int, path: Path, ppi: int | None = None):
    row_ppi = module.DATASET_CONTEXT.expected_ppi if ppi is None else ppi
    return module.RollSelfPair(
        pair_id=module.make_pair_id(subject_id, canonical_position),
        dataset=module.DATASET,
        protocol=module.PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=row_ppi,
        raw_frgp_a=frgp,
        raw_frgp_b=frgp,
        path_a=path,
        path_b=path,
    )


def _plain_roll_pair(
    module,
    subject_id: str,
    canonical_position: int,
    plain_frgp: int,
    roll_frgp: int,
    plain_path: Path,
    roll_path: Path,
):
    return module.PlainRollPair(
        pair_id=module.make_pair_id(subject_id, canonical_position),
        dataset=module.DATASET,
        protocol=module.PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=module.DATASET_CONTEXT.expected_ppi,
        raw_frgp_a=plain_frgp,
        raw_frgp_b=roll_frgp,
        path_a=plain_path,
        path_b=roll_path,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
