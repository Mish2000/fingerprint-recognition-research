from pathlib import Path

import pytest

import fingerprint_data_discovery.plain_roll_protocol as plain_roll_protocol
import fingerprint_data_discovery.plain_self_protocol as plain_self_protocol
import fingerprint_data_discovery.roll_self_protocol as roll_self_protocol
import fingerprint_data_discovery.sd300b_plain_roll as sd300b_plain_roll
import fingerprint_data_discovery.sd300b_plain_self as sd300b_plain_self
import fingerprint_data_discovery.sd300b_roll_self as sd300b_roll_self
import fingerprint_data_discovery.sd300c_plain_roll as sd300c_plain_roll
import fingerprint_data_discovery.sd300c_plain_self as sd300c_plain_self
import fingerprint_data_discovery.sd300c_roll_self as sd300c_roll_self
from fingerprint_data_discovery.protocol_dataset import SD300B_CONTEXT, SD300C_CONTEXT


def test_roll_self_rejects_cross_dataset_source_paths_even_when_dataset_field_matches(tmp_path):
    data_root = _make_data_root(tmp_path, SD300B_CONTEXT, SD300C_CONTEXT)
    sd300b_roll = _touch_image(data_root, SD300B_CONTEXT, "roll", "00001000", 1)
    sd300c_roll = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    manifest_path = tmp_path / "roll_self.csv"

    sd300c_roll_self.write_manifest_atomic(
        [_roll_pair(sd300c_roll_self, "00001000", 1, 1, sd300b_roll)],
        manifest_path,
    )
    with pytest.raises(sd300c_roll_self.ManifestValidationError, match="not under the SD300c roll directory"):
        sd300c_roll_self.validate_manifest(manifest_path, data_root)

    sd300b_roll_self.write_manifest_atomic(
        [_roll_pair(sd300b_roll_self, "00001000", 1, 1, sd300c_roll)],
        manifest_path,
    )
    with pytest.raises(sd300b_roll_self.ManifestValidationError, match="not under the SD300b roll directory"):
        sd300b_roll_self.validate_manifest(manifest_path, data_root)


def test_plain_roll_rejects_cross_dataset_source_paths_on_both_sides(tmp_path):
    data_root = _make_data_root(tmp_path, SD300B_CONTEXT, SD300C_CONTEXT)
    sd300b_plain = _touch_image(data_root, SD300B_CONTEXT, "plain", "00001000", 11)
    sd300b_roll = _touch_image(data_root, SD300B_CONTEXT, "roll", "00001000", 1)
    sd300c_plain = _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 11)
    sd300c_roll = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    manifest_path = tmp_path / "plain_roll.csv"

    sd300c_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300c_plain_roll, "00001000", 1, 11, 1, sd300b_plain, sd300c_roll)],
        manifest_path,
    )
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="path_a .* SD300c plain directory"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)

    sd300c_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300c_plain_roll, "00001000", 1, 11, 1, sd300c_plain, sd300b_roll)],
        manifest_path,
    )
    with pytest.raises(sd300c_plain_roll.ManifestValidationError, match="path_b .* SD300c roll directory"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)

    sd300b_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300b_plain_roll, "00001000", 1, 11, 1, sd300c_plain, sd300b_roll)],
        manifest_path,
    )
    with pytest.raises(sd300b_plain_roll.ManifestValidationError, match="path_a .* SD300b plain directory"):
        sd300b_plain_roll.validate_manifest(manifest_path, data_root)

    sd300b_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300b_plain_roll, "00001000", 1, 11, 1, sd300b_plain, sd300c_roll)],
        manifest_path,
    )
    with pytest.raises(sd300b_plain_roll.ManifestValidationError, match="path_b .* SD300b roll directory"):
        sd300b_plain_roll.validate_manifest(manifest_path, data_root)


def test_expected_identity_programming_errors_are_not_wrapped_as_plain_self_validation_errors(
    tmp_path,
    monkeypatch,
):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    image_path = _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 11)
    manifest_path = tmp_path / "plain_self.csv"
    sd300c_plain_self.write_manifest_atomic(
        [_plain_self_pair(sd300c_plain_self, "00001000", 1, 11, image_path)],
        manifest_path,
    )

    monkeypatch.setattr(
        plain_self_protocol,
        "build_plain_self_pairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("programming bug")),
    )

    with pytest.raises(RuntimeError, match="programming bug"):
        sd300c_plain_self.validate_manifest(manifest_path, data_root)


def test_expected_identity_programming_errors_are_not_wrapped_as_roll_self_validation_errors(
    tmp_path,
    monkeypatch,
):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    image_path = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    manifest_path = tmp_path / "roll_self.csv"
    sd300c_roll_self.write_manifest_atomic(
        [_roll_pair(sd300c_roll_self, "00001000", 1, 1, image_path)],
        manifest_path,
    )

    monkeypatch.setattr(
        roll_self_protocol,
        "build_roll_self_pairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("programming bug")),
    )

    with pytest.raises(RuntimeError, match="programming bug"):
        sd300c_roll_self.validate_manifest(manifest_path, data_root)


def test_expected_identity_programming_errors_are_not_wrapped_as_plain_roll_validation_errors(
    tmp_path,
    monkeypatch,
):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    plain_path = _touch_image(data_root, SD300C_CONTEXT, "plain", "00001000", 11)
    roll_path = _touch_image(data_root, SD300C_CONTEXT, "roll", "00001000", 1)
    manifest_path = tmp_path / "plain_roll.csv"
    sd300c_plain_roll.write_manifest_atomic(
        [_plain_roll_pair(sd300c_plain_roll, "00001000", 1, 11, 1, plain_path, roll_path)],
        manifest_path,
    )

    monkeypatch.setattr(
        plain_roll_protocol,
        "_eligible_records_by_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("programming bug")),
    )

    with pytest.raises(RuntimeError, match="programming bug"):
        sd300c_plain_roll.validate_manifest(manifest_path, data_root)


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


def _plain_self_pair(module, subject_id: str, canonical_position: int, frgp: int, path: Path):
    return module.PlainSelfPair(
        pair_id=module.make_pair_id(subject_id, canonical_position),
        dataset=module.DATASET,
        protocol=module.PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=module.DATASET_CONTEXT.expected_ppi,
        raw_frgp_a=frgp,
        raw_frgp_b=frgp,
        path_a=path,
        path_b=path,
    )


def _roll_pair(module, subject_id: str, canonical_position: int, frgp: int, path: Path):
    return module.RollSelfPair(
        pair_id=module.make_pair_id(subject_id, canonical_position),
        dataset=module.DATASET,
        protocol=module.PROTOCOL,
        subject_id=subject_id,
        canonical_finger_position=canonical_position,
        ppi=module.DATASET_CONTEXT.expected_ppi,
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
