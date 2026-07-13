import csv
from pathlib import Path
import subprocess
import sys

import pytest

import fingerprint_data_discovery.sd300b_plain_roll as plain_roll
import fingerprint_data_discovery.sd300b_plain_self as plain_self
import fingerprint_data_discovery.sd300b_protocol_audit as protocol_audit
import fingerprint_data_discovery.sd300b_roll_self as roll_self
from fingerprint_data_discovery.self_manifest_common import MANIFEST_COLUMNS
from fingerprint_data_discovery.sd300b_protocol_audit import (
    ProtocolAuditError,
    build_protocol_audit_report,
    save_report_json,
)


def test_three_consistent_manifests_pass_audit(tmp_path):
    fixture = _write_consistent_manifests(tmp_path)

    report = build_protocol_audit_report(
        fixture.data_root,
        fixture.plain_self_manifest,
        fixture.roll_self_manifest,
        fixture.plain_roll_manifest,
    )

    assert report.dataset == "sd300b"
    assert report.plain_self_count == 3
    assert report.roll_self_count == 3
    assert report.plain_roll_count == 2
    assert report.expected_intersection_count == 2
    assert report.plain_only_count == 1
    assert report.roll_only_count == 1
    assert report.intersection_consistent is True
    assert report.source_consistency_checked_pairs == 2
    assert report.canonical_pair_counts["plain_self"]["03"] == 1
    assert report.canonical_pair_counts["roll_self"]["04"] == 1
    assert report.canonical_pair_counts["plain_roll"]["01"] == 1
    assert report.manifest_sha256["plain_self"] == _sha256(fixture.plain_self_manifest)


def test_plain_roll_identities_must_equal_plain_and_roll_intersection(tmp_path):
    fixture = _write_consistent_manifests(tmp_path)

    report = build_protocol_audit_report(
        fixture.data_root,
        fixture.plain_self_manifest,
        fixture.roll_self_manifest,
        fixture.plain_roll_manifest,
    )

    assert report.plain_roll_count == report.expected_intersection_count
    assert report.intersection_consistent is True


def test_missing_plain_roll_identity_fails(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    _write_rows(fixture.plain_roll_manifest, _read_rows(fixture.plain_roll_manifest)[:1])

    with pytest.raises(ProtocolAuditError, match="missing identities"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_extra_plain_roll_identity_fails(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    rows = _read_rows(fixture.plain_roll_manifest)
    rows.append(
        _plain_roll_row(
            "00001001",
            canonical_position=3,
            ppi=1000,
            plain_frgp=3,
            roll_frgp=3,
            plain_path=fixture.paths["plain_only"],
            roll_path=fixture.paths["roll_only"],
        )
    )
    _write_rows(fixture.plain_roll_manifest, rows)

    with pytest.raises(ProtocolAuditError, match="outside plain_self and roll_self intersection"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_plain_roll_path_a_mismatch_with_plain_self_fails(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    rows = _read_rows(fixture.plain_roll_manifest)
    rows[0]["path_a"] = str(fixture.paths["intersection_plain_02"])
    _write_rows(fixture.plain_roll_manifest, rows)

    with pytest.raises(ProtocolAuditError, match="path_a mismatch"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_plain_roll_path_b_mismatch_with_roll_self_fails(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    rows = _read_rows(fixture.plain_roll_manifest)
    rows[0]["path_b"] = str(fixture.paths["intersection_roll_02"])
    _write_rows(fixture.plain_roll_manifest, rows)

    with pytest.raises(ProtocolAuditError, match="path_b mismatch"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_plain_roll_raw_frgp_a_mismatch_fails(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    rows = _read_rows(fixture.plain_roll_manifest)
    rows[0]["raw_frgp_a"] = "2"
    _write_rows(fixture.plain_roll_manifest, rows)

    with pytest.raises(ProtocolAuditError, match="raw_frgp_a mismatch"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_plain_roll_raw_frgp_b_mismatch_fails(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    rows = _read_rows(fixture.plain_roll_manifest)
    rows[0]["raw_frgp_b"] = "2"
    _write_rows(fixture.plain_roll_manifest, rows)

    with pytest.raises(ProtocolAuditError, match="raw_frgp_b mismatch"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_plain_roll_ppi_inconsistency_fails(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    rows = _read_rows(fixture.plain_roll_manifest)
    rows[0]["ppi"] = "999"
    _write_rows(fixture.plain_roll_manifest, rows)

    with pytest.raises(ProtocolAuditError, match="PPI mismatch"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_swapped_plain_roll_a_b_paths_fail(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)
    _disable_plain_roll_validator(monkeypatch)
    rows = _read_rows(fixture.plain_roll_manifest)
    rows[0]["path_a"], rows[0]["path_b"] = rows[0]["path_b"], rows[0]["path_a"]
    _write_rows(fixture.plain_roll_manifest, rows)

    with pytest.raises(ProtocolAuditError, match="path_a/path_b appear swapped"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_dedicated_validator_failure_stops_cross_audit(tmp_path, monkeypatch):
    fixture = _write_consistent_manifests(tmp_path)

    def fail_validator(manifest_path, data_root):
        raise plain_self.ManifestValidationError("forced plain_self failure")

    monkeypatch.setattr(protocol_audit.sd300b_plain_self, "validate_manifest", fail_validator)

    with pytest.raises(ProtocolAuditError, match="Dedicated validator failed for plain_self"):
        build_protocol_audit_report(
            fixture.data_root,
            fixture.plain_self_manifest,
            fixture.roll_self_manifest,
            fixture.plain_roll_manifest,
        )


def test_counts_are_computed_from_manifest_rows(tmp_path):
    fixture = _write_single_intersection_manifest_set(tmp_path)

    report = build_protocol_audit_report(
        fixture.data_root,
        fixture.plain_self_manifest,
        fixture.roll_self_manifest,
        fixture.plain_roll_manifest,
    )

    assert report.plain_self_count == 1
    assert report.roll_self_count == 1
    assert report.plain_roll_count == 1
    assert report.plain_only_count == 0
    assert report.roll_only_count == 0
    assert report.canonical_pair_counts["plain_roll"]["01"] == 1
    assert report.canonical_pair_counts["plain_roll"]["02"] == 0


def test_json_report_repeated_inputs_are_byte_identical(tmp_path):
    fixture = _write_consistent_manifests(tmp_path)
    report = build_protocol_audit_report(
        fixture.data_root,
        fixture.plain_self_manifest,
        fixture.roll_self_manifest,
        fixture.plain_roll_manifest,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    save_report_json(report, first_path)
    save_report_json(report, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_cli_inconsistency_exits_1_without_traceback(tmp_path):
    fixture = _write_consistent_manifests(tmp_path)
    _write_rows(fixture.plain_roll_manifest, _read_rows(fixture.plain_roll_manifest)[:1])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300b_protocol_audit",
            "--data-root",
            str(fixture.data_root),
            "--plain-self-manifest",
            str(fixture.plain_self_manifest),
            "--roll-self-manifest",
            str(fixture.roll_self_manifest),
            "--plain-roll-manifest",
            str(fixture.plain_roll_manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("Error:")
    assert "Traceback" not in result.stderr


class _Fixture:
    def __init__(
        self,
        data_root: Path,
        plain_self_manifest: Path,
        roll_self_manifest: Path,
        plain_roll_manifest: Path,
        paths: dict[str, Path],
    ) -> None:
        self.data_root = data_root
        self.plain_self_manifest = plain_self_manifest
        self.roll_self_manifest = roll_self_manifest
        self.plain_roll_manifest = plain_roll_manifest
        self.paths = paths


def _write_consistent_manifests(tmp_path: Path) -> _Fixture:
    data_root = _make_data_root(tmp_path)
    paths = {
        "intersection_plain_01": _touch_plain_image(data_root, "00001000", 11),
        "intersection_roll_01": _touch_roll_image(data_root, "00001000", 1),
        "intersection_plain_02": _touch_plain_image(data_root, "00001000", 2),
        "intersection_roll_02": _touch_roll_image(data_root, "00001000", 2),
        "plain_only": _touch_plain_image(data_root, "00001001", 3),
        "roll_only": _touch_roll_image(data_root, "00001002", 4),
    }
    return _write_fixture_manifests(
        tmp_path,
        data_root,
        paths,
        plain_pairs=[
            _plain_self_pair("00001000", 1, 11, paths["intersection_plain_01"]),
            _plain_self_pair("00001000", 2, 2, paths["intersection_plain_02"]),
            _plain_self_pair("00001001", 3, 3, paths["plain_only"]),
        ],
        roll_pairs=[
            _roll_self_pair("00001000", 1, 1, paths["intersection_roll_01"]),
            _roll_self_pair("00001000", 2, 2, paths["intersection_roll_02"]),
            _roll_self_pair("00001002", 4, 4, paths["roll_only"]),
        ],
        plain_roll_pairs=[
            _plain_roll_pair(
                "00001000",
                1,
                11,
                1,
                paths["intersection_plain_01"],
                paths["intersection_roll_01"],
            ),
            _plain_roll_pair(
                "00001000",
                2,
                2,
                2,
                paths["intersection_plain_02"],
                paths["intersection_roll_02"],
            ),
        ],
    )


def _write_single_intersection_manifest_set(tmp_path: Path) -> _Fixture:
    data_root = _make_data_root(tmp_path)
    paths = {
        "plain": _touch_plain_image(data_root, "00001000", 11),
        "roll": _touch_roll_image(data_root, "00001000", 1),
    }
    return _write_fixture_manifests(
        tmp_path,
        data_root,
        paths,
        plain_pairs=[_plain_self_pair("00001000", 1, 11, paths["plain"])],
        roll_pairs=[_roll_self_pair("00001000", 1, 1, paths["roll"])],
        plain_roll_pairs=[
            _plain_roll_pair("00001000", 1, 11, 1, paths["plain"], paths["roll"])
        ],
    )


def _write_fixture_manifests(
    tmp_path: Path,
    data_root: Path,
    paths: dict[str, Path],
    plain_pairs: list[plain_self.PlainSelfPair],
    roll_pairs: list[roll_self.RollSelfPair],
    plain_roll_pairs: list[plain_roll.PlainRollPair],
) -> _Fixture:
    manifest_dir = tmp_path / "manifests"
    plain_self_manifest = manifest_dir / "plain_self.csv"
    roll_self_manifest = manifest_dir / "roll_self.csv"
    plain_roll_manifest = manifest_dir / "plain_roll.csv"

    plain_self.write_manifest_atomic(plain_pairs, plain_self_manifest)
    roll_self.write_manifest_atomic(roll_pairs, roll_self_manifest)
    plain_roll.write_manifest_atomic(plain_roll_pairs, plain_roll_manifest)
    return _Fixture(data_root, plain_self_manifest, roll_self_manifest, plain_roll_manifest, paths)


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


def _plain_self_pair(subject_id: str, canonical_position: int, frgp: int, path: Path) -> plain_self.PlainSelfPair:
    return plain_self.PlainSelfPair(
        pair_id=plain_self.make_pair_id(subject_id, canonical_position),
        dataset=plain_self.DATASET,
        protocol=plain_self.PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=1000,
        raw_frgp_a=frgp,
        raw_frgp_b=frgp,
        path_a=path,
        path_b=path,
    )


def _roll_self_pair(subject_id: str, canonical_position: int, frgp: int, path: Path) -> roll_self.RollSelfPair:
    return roll_self.RollSelfPair(
        pair_id=roll_self.make_pair_id(subject_id, canonical_position),
        dataset=roll_self.DATASET,
        protocol=roll_self.PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=1000,
        raw_frgp_a=frgp,
        raw_frgp_b=frgp,
        path_a=path,
        path_b=path,
    )


def _plain_roll_pair(
    subject_id: str,
    canonical_position: int,
    plain_frgp: int,
    roll_frgp: int,
    plain_path: Path,
    roll_path: Path,
) -> plain_roll.PlainRollPair:
    return plain_roll.PlainRollPair(
        pair_id=plain_roll.make_pair_id(subject_id, canonical_position),
        dataset=plain_roll.DATASET,
        protocol=plain_roll.PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=1000,
        raw_frgp_a=plain_frgp,
        raw_frgp_b=roll_frgp,
        path_a=plain_path,
        path_b=roll_path,
    )


def _plain_roll_row(
    subject_id: str,
    canonical_position: int,
    ppi: int,
    plain_frgp: int,
    roll_frgp: int,
    plain_path: Path,
    roll_path: Path,
) -> dict[str, str]:
    return _plain_roll_pair(
        subject_id,
        canonical_position,
        plain_frgp,
        roll_frgp,
        plain_path,
        roll_path,
    ).as_csv_row()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _disable_plain_roll_validator(monkeypatch) -> None:
    monkeypatch.setattr(protocol_audit.sd300b_plain_roll, "validate_manifest", lambda *args: object())


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
