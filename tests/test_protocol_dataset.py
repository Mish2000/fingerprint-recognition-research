from pathlib import Path

import pytest

from fingerprint_data_discovery.nist_sd300 import DATASETS
from fingerprint_data_discovery.protocol_dataset import (
    ProtocolDatasetContext,
    SD300B_CONTEXT,
    SD300C_CONTEXT,
)
import fingerprint_data_discovery.sd300b_plain_roll as plain_roll
import fingerprint_data_discovery.sd300b_plain_self as plain_self
import fingerprint_data_discovery.sd300b_protocol_audit as protocol_audit
import fingerprint_data_discovery.sd300b_roll_self as roll_self


def test_matching_context_name_and_spec_is_created_successfully():
    context = ProtocolDatasetContext(name="sd300b", spec=DATASETS["sd300b"])

    assert context.name == "sd300b"
    assert context.spec is DATASETS["sd300b"]


def test_sd300b_name_with_sd300c_spec_is_rejected():
    with pytest.raises(ValueError, match="context.name='sd300b'.*spec.name='sd300c'"):
        ProtocolDatasetContext(name="sd300b", spec=DATASETS["sd300c"])


def test_sd300c_name_with_sd300b_spec_is_rejected():
    with pytest.raises(ValueError, match="context.name='sd300c'.*spec.name='sd300b'"):
        ProtocolDatasetContext(name="sd300c", spec=DATASETS["sd300b"])


def test_sd300b_context_uses_existing_dataset_spec():
    assert SD300B_CONTEXT.name == "sd300b"
    assert SD300B_CONTEXT.spec is DATASETS["sd300b"]


def test_sd300c_context_uses_existing_dataset_spec():
    assert SD300C_CONTEXT.name == "sd300c"
    assert SD300C_CONTEXT.spec is DATASETS["sd300c"]


def test_expected_ppi_values_come_from_dataset_specs():
    assert SD300B_CONTEXT.expected_ppi == DATASETS["sd300b"].ppi
    assert SD300C_CONTEXT.expected_ppi == DATASETS["sd300c"].ppi


def test_manifest_path_construction_is_deterministic():
    assert SD300B_CONTEXT.protocol_output_dir == Path("protocols") / "sd300b"
    assert SD300B_CONTEXT.manifest_path("plain_self") == Path("protocols") / "sd300b" / "plain_self.csv"
    assert SD300B_CONTEXT.manifest_path("plain_self") == SD300B_CONTEXT.manifest_path("plain_self")
    assert SD300C_CONTEXT.manifest_path("plain_self") == Path("protocols") / "sd300c" / "plain_self.csv"


def test_pair_id_construction_preserves_existing_format():
    assert (
        SD300B_CONTEXT.pair_id("plain_roll", "00001000", 1)
        == "sd300b_plain_roll_00001000_01"
    )
    assert (
        SD300C_CONTEXT.pair_id("plain_roll", "00001000", 10)
        == "sd300c_plain_roll_00001000_10"
    )


def test_sd300b_public_make_pair_id_functions_preserve_values():
    assert plain_self.make_pair_id("00001000", 1) == "sd300b_plain_self_00001000_01"
    assert roll_self.make_pair_id("00001000", 1) == "sd300b_roll_self_00001000_01"
    assert plain_roll.make_pair_id("00001000", 1) == "sd300b_plain_roll_00001000_01"


def test_sd300b_default_manifest_paths_are_unchanged():
    assert plain_self.DEFAULT_MANIFEST_PATH == Path("protocols") / "sd300b" / "plain_self.csv"
    assert roll_self.DEFAULT_MANIFEST_PATH == Path("protocols") / "sd300b" / "roll_self.csv"
    assert plain_roll.DEFAULT_MANIFEST_PATH == Path("protocols") / "sd300b" / "plain_roll.csv"
    assert protocol_audit.DEFAULT_REPORT_PATH == Path("protocols") / "sd300b" / "protocol_audit.json"


def test_sd300c_context_does_not_create_artifacts(tmp_path):
    context = ProtocolDatasetContext(
        name=SD300C_CONTEXT.name,
        spec=SD300C_CONTEXT.spec,
        protocols_root=tmp_path / "protocols",
    )

    manifest_path = context.manifest_path("plain_self")

    assert manifest_path == tmp_path / "protocols" / "sd300c" / "plain_self.csv"
    assert not manifest_path.exists()
    assert not manifest_path.parent.exists()


def test_repeated_generation_of_sd300b_manifests_is_byte_identical(tmp_path):
    data_root = _make_data_root(tmp_path)
    _touch_plain_image(data_root, "00001000", 11)
    _touch_roll_image(data_root, "00001000", 1)
    _touch_plain_image(data_root, "00001000", 2)
    _touch_roll_image(data_root, "00001000", 2)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    pairs = [
        (plain_self.generate_manifest, "plain_self.csv"),
        (roll_self.generate_manifest, "roll_self.csv"),
        (plain_roll.generate_manifest, "plain_roll.csv"),
    ]

    for generate_manifest, filename in pairs:
        first_path = first_dir / filename
        second_path = second_dir / filename
        generate_manifest(data_root, first_path)
        generate_manifest(data_root, second_path)
        assert first_path.read_bytes() == second_path.read_bytes()


def test_protocol_audit_json_rerun_is_byte_identical(tmp_path):
    data_root = _make_data_root(tmp_path)
    _touch_plain_image(data_root, "00001000", 11)
    _touch_roll_image(data_root, "00001000", 1)
    manifest_dir = tmp_path / "manifests"
    plain_self_manifest = manifest_dir / "plain_self.csv"
    roll_self_manifest = manifest_dir / "roll_self.csv"
    plain_roll_manifest = manifest_dir / "plain_roll.csv"

    plain_self.generate_manifest(data_root, plain_self_manifest)
    roll_self.generate_manifest(data_root, roll_self_manifest)
    plain_roll.generate_manifest(data_root, plain_roll_manifest)

    first_report_path = tmp_path / "first_protocol_audit.json"
    second_report_path = tmp_path / "second_protocol_audit.json"
    first_report = protocol_audit.build_protocol_audit_report(
        data_root,
        plain_self_manifest,
        roll_self_manifest,
        plain_roll_manifest,
    )
    second_report = protocol_audit.build_protocol_audit_report(
        data_root,
        plain_self_manifest,
        roll_self_manifest,
        plain_roll_manifest,
    )

    protocol_audit.save_report_json(first_report, first_report_path)
    protocol_audit.save_report_json(second_report, second_report_path)

    assert first_report_path.read_bytes() == second_report_path.read_bytes()


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
