from pathlib import Path
import subprocess
import sys

import pytest

from fingerprint_data_discovery.canonical_fingers import CanonicalFingerMappingError
from fingerprint_data_discovery.nist_sd300 import ImageRecord, ScanResult
from fingerprint_data_discovery.protocol_dataset import SD300B_CONTEXT, SD300C_CONTEXT
import fingerprint_data_discovery.plain_self_protocol as plain_self_protocol
import fingerprint_data_discovery.sd300b_plain_self as sd300b_plain_self
import fingerprint_data_discovery.sd300c_plain_self as sd300c_plain_self


CONTEXT_CASES = [
    (SD300B_CONTEXT, sd300b_plain_self),
    (SD300C_CONTEXT, sd300c_plain_self),
]


@pytest.mark.parametrize(("context", "module"), CONTEXT_CASES)
def test_plain_self_semantics_work_with_dataset_context(context, module):
    record = _record(context, subject_id="00001000", frgp=11)

    pairs = module.build_plain_self_pairs(ScanResult(records=[record], errors=[]))

    assert len(pairs) == 1
    assert pairs[0].dataset == context.name
    assert pairs[0].protocol == "plain_self"
    assert pairs[0].canonical_finger_position == 1
    assert pairs[0].path_a == pairs[0].path_b == record.absolute_path


def test_pair_id_prefixes_are_dataset_specific():
    assert sd300b_plain_self.make_pair_id("00001000", 1).startswith("sd300b_plain_self_")
    assert sd300c_plain_self.make_pair_id("00001000", 1).startswith("sd300c_plain_self_")


def test_manifest_paths_are_dataset_specific():
    assert sd300b_plain_self.DEFAULT_MANIFEST_PATH == Path("protocols") / "sd300b" / "plain_self.csv"
    assert sd300c_plain_self.DEFAULT_MANIFEST_PATH == Path("protocols") / "sd300c" / "plain_self.csv"


def test_expected_ppi_values_come_from_contexts():
    assert sd300b_plain_self.DATASET_CONTEXT.expected_ppi == SD300B_CONTEXT.spec.ppi
    assert sd300c_plain_self.DATASET_CONTEXT.expected_ppi == SD300C_CONTEXT.spec.ppi


@pytest.mark.parametrize(("context", "module"), CONTEXT_CASES)
@pytest.mark.parametrize(
    ("frgp", "canonical_position"),
    [(11, 1), (12, 6)],
)
def test_plain_thumb_frgp_mappings_work_in_both_contexts(context, module, frgp, canonical_position):
    pairs = module.build_plain_self_pairs(
        ScanResult(records=[_record(context, subject_id="00001000", frgp=frgp)], errors=[])
    )

    assert [(pair.raw_frgp_a, pair.canonical_finger_position) for pair in pairs] == [
        (frgp, canonical_position)
    ]


@pytest.mark.parametrize(("context", "module"), CONTEXT_CASES)
def test_plain_multi_finger_frgp_13_and_14_do_not_create_pairs(context, module):
    pairs = module.build_plain_self_pairs(
        ScanResult(
            records=[
                _record(context, subject_id="00001000", frgp=13),
                _record(context, subject_id="00001000", frgp=14),
                _record(context, subject_id="00001000", frgp=2),
            ],
            errors=[],
        )
    )

    assert [(pair.canonical_finger_position, pair.raw_frgp_a) for pair in pairs] == [(2, 2)]


@pytest.mark.parametrize(("context", "module"), CONTEXT_CASES)
def test_duplicate_anatomical_identity_is_rejected_in_both_contexts(context, module):
    records = [
        _record(context, subject_id="00001000", frgp=11, path_suffix="a"),
        _record(context, subject_id="00001000", frgp=11, path_suffix="b"),
    ]

    with pytest.raises(module.ManifestGenerationError, match="Duplicate plain single-finger"):
        module.build_plain_self_pairs(ScanResult(records=records, errors=[]))


@pytest.mark.parametrize(("context", "module"), CONTEXT_CASES)
def test_input_order_does_not_affect_sorting_or_pair_ids(context, module):
    records = [
        _record(context, subject_id="00001002", frgp=7),
        _record(context, subject_id="00001001", frgp=11),
        _record(context, subject_id="00001001", frgp=10),
    ]

    first = module.build_plain_self_pairs(ScanResult(records=records, errors=[]))
    second = module.build_plain_self_pairs(ScanResult(records=list(reversed(records)), errors=[]))

    expected = [
        (module.make_pair_id("00001001", 1), "00001001", 1),
        (module.make_pair_id("00001001", 10), "00001001", 10),
        (module.make_pair_id("00001002", 7), "00001002", 7),
    ]
    assert [(pair.pair_id, pair.subject_id, pair.canonical_finger_position) for pair in first] == expected
    assert [(pair.pair_id, pair.subject_id, pair.canonical_finger_position) for pair in second] == expected


def test_sd300b_manifest_is_not_valid_under_sd300c_context(tmp_path):
    data_root = _make_data_root(tmp_path, SD300B_CONTEXT, SD300C_CONTEXT)
    sd300b_path = _touch_plain_image(data_root, SD300B_CONTEXT, "00001000", 11)
    _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "sd300b_plain_self.csv"
    sd300b_plain_self.write_manifest_atomic(
        [_pair(sd300b_plain_self, "00001000", 1, 11, sd300b_path)],
        manifest_path,
    )

    with pytest.raises(sd300c_plain_self.ManifestValidationError, match="Invalid dataset"):
        sd300c_plain_self.validate_manifest(manifest_path, data_root)


def test_sd300c_manifest_is_not_valid_under_sd300b_context(tmp_path):
    data_root = _make_data_root(tmp_path, SD300B_CONTEXT, SD300C_CONTEXT)
    _touch_plain_image(data_root, SD300B_CONTEXT, "00001000", 11)
    sd300c_path = _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "sd300c_plain_self.csv"
    sd300c_plain_self.write_manifest_atomic(
        [_pair(sd300c_plain_self, "00001000", 1, 11, sd300c_path)],
        manifest_path,
    )

    with pytest.raises(sd300b_plain_self.ManifestValidationError, match="Invalid dataset"):
        sd300b_plain_self.validate_manifest(manifest_path, data_root)


def test_wrong_ppi_is_rejected(tmp_path):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    image_path = _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "plain_self.csv"
    pair = _pair(sd300c_plain_self, "00001000", 1, 11, image_path, ppi=1000)

    sd300c_plain_self.write_manifest_atomic([pair], manifest_path)

    with pytest.raises(sd300c_plain_self.ManifestValidationError, match="PPI mismatch"):
        sd300c_plain_self.validate_manifest(manifest_path, data_root)


def test_wrong_dataset_source_path_is_rejected(tmp_path):
    data_root = _make_data_root(tmp_path, SD300B_CONTEXT, SD300C_CONTEXT)
    wrong_path = _touch_plain_image(data_root, SD300B_CONTEXT, "00001000", 11)
    _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "plain_self.csv"
    pair = _pair(sd300c_plain_self, "00001000", 1, 11, wrong_path)

    sd300c_plain_self.write_manifest_atomic([pair], manifest_path)

    with pytest.raises(sd300c_plain_self.ManifestValidationError, match="not under the SD300c plain directory"):
        sd300c_plain_self.validate_manifest(manifest_path, data_root)


def test_missing_identity_is_rejected(tmp_path):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    first_path = _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 2)
    manifest_path = tmp_path / "plain_self.csv"

    sd300c_plain_self.write_manifest_atomic(
        [_pair(sd300c_plain_self, "00001000", 1, 11, first_path)],
        manifest_path,
    )

    with pytest.raises(sd300c_plain_self.ManifestValidationError, match="missing expected anatomical identities"):
        sd300c_plain_self.validate_manifest(manifest_path, data_root)


def test_extra_identity_is_rejected(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    image_path = _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "plain_self.csv"

    sd300c_plain_self.write_manifest_atomic(
        [_pair(sd300c_plain_self, "00001000", 1, 11, image_path)],
        manifest_path,
    )
    monkeypatch.setattr(
        plain_self_protocol,
        "_expected_pairs_by_identity",
        lambda *args: {},
    )

    with pytest.raises(sd300c_plain_self.ManifestValidationError, match="unexpected anatomical identities"):
        sd300c_plain_self.validate_manifest(manifest_path, data_root)


@pytest.mark.parametrize(("context", "module"), CONTEXT_CASES)
def test_invalid_plain_frgp_does_not_leak_canonical_mapping_error(tmp_path, context, module):
    data_root = _make_data_root(tmp_path, context)
    image_path = _touch_plain_image(data_root, context, "00001000", 11)
    manifest_path = tmp_path / f"{context.name}_plain_self.csv"
    pair = _pair(module, "00001000", 1, 1, image_path)

    module.write_manifest_atomic([pair], manifest_path)

    with pytest.raises(module.ManifestValidationError, match="raw FRGP is not valid for plain") as exc_info:
        module.validate_manifest(manifest_path, data_root)
    assert isinstance(exc_info.value.__cause__, CanonicalFingerMappingError)


def test_sd300c_generation_validation_failure_does_not_replace_existing_manifest(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "plain_self.csv"
    original_bytes = b"existing manifest bytes\n"
    manifest_path.write_bytes(original_bytes)

    def fail_validation(manifest_path, data_root):
        raise sd300c_plain_self.ManifestValidationError("forced validation failure")

    monkeypatch.setattr(sd300c_plain_self, "validate_manifest", fail_validation)

    with pytest.raises(sd300c_plain_self.ManifestValidationError, match="forced validation failure"):
        sd300c_plain_self.generate_manifest(data_root, manifest_path)

    assert manifest_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".plain_self.csv.*.tmp")) == []


@pytest.mark.parametrize(("context", "module"), CONTEXT_CASES)
def test_repeated_generation_on_same_inputs_produces_identical_bytes(tmp_path, context, module):
    data_root = _make_data_root(tmp_path, context)
    _touch_plain_image(data_root, context, "00001002", 7)
    _touch_plain_image(data_root, context, "00001001", 11)
    first_manifest = tmp_path / f"{context.name}_first.csv"
    second_manifest = tmp_path / f"{context.name}_second.csv"

    module.generate_manifest(data_root, first_manifest)
    module.generate_manifest(data_root, second_manifest)

    assert first_manifest.read_bytes() == second_manifest.read_bytes()


def test_sd300c_cli_generate_and_validate_succeed(tmp_path):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "plain_self.csv"

    generate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300c_plain_self",
            "generate",
            "--data-root",
            str(data_root),
            "--output",
            str(manifest_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    validate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300c_plain_self",
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

    assert generate_result.returncode == 0
    assert validate_result.returncode == 0
    assert "Validated" in validate_result.stdout


def test_sd300c_cli_invalid_manifest_returns_error_without_traceback(tmp_path):
    data_root = _make_data_root(tmp_path, SD300C_CONTEXT)
    image_path = _touch_plain_image(data_root, SD300C_CONTEXT, "00001000", 11)
    manifest_path = tmp_path / "plain_self.csv"
    sd300c_plain_self.write_manifest_atomic(
        [_pair(sd300c_plain_self, "00001000", 1, 1, image_path)],
        manifest_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_data_discovery.sd300c_plain_self",
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
    assert "Traceback" not in result.stderr


def _record(
    context,
    subject_id: str,
    frgp: int,
    path_suffix: str = "",
) -> ImageRecord:
    suffix = f"_{path_suffix}" if path_suffix else ""
    return ImageRecord(
        dataset=context.name,
        subject_id=subject_id,
        impression_type="plain",
        ppi=context.expected_ppi,
        frgp=frgp,
        finger_position=f"test_frgp_{frgp}",
        absolute_path=Path(
            f"C:/fingerprint-datasets/{context.name}_{subject_id}_plain_{context.expected_ppi}_{frgp:02d}{suffix}.png"
        ),
    )


def _make_data_root(tmp_path: Path, *contexts) -> Path:
    data_root = tmp_path / "fingerprint-datasets"
    for context in contexts:
        (data_root / context.spec.relative_image_root / "plain").mkdir(parents=True, exist_ok=True)
        (data_root / context.spec.relative_image_root / "roll").mkdir(parents=True, exist_ok=True)
    return data_root


def _touch_plain_image(
    data_root: Path,
    context,
    subject_id: str,
    frgp: int,
) -> Path:
    path = (
        data_root
        / context.spec.relative_image_root
        / "plain"
        / f"{subject_id}_plain_{context.expected_ppi}_{frgp:02d}.png"
    )
    path.write_bytes(b"not a real image; filename validation only")
    return path


def _pair(
    module,
    subject_id: str,
    canonical_position: int,
    frgp: int,
    path: Path,
    ppi: int | None = None,
):
    row_ppi = module.DATASET_CONTEXT.expected_ppi if ppi is None else ppi
    return module.PlainSelfPair(
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
