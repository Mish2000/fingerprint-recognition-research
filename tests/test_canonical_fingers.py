import pytest

from fingerprint_data_discovery.canonical_fingers import (
    CANONICAL_FINGER_POSITIONS,
    CanonicalFingerMappingError,
    canonical_finger_position,
    is_single_finger_capture,
)


def test_canonical_positions_are_exactly_one_to_ten():
    assert CANONICAL_FINGER_POSITIONS == {
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
    }


@pytest.mark.parametrize("frgp", range(1, 11))
def test_roll_frgp_maps_directly_to_canonical_position(frgp):
    assert canonical_finger_position("roll", frgp) == frgp
    assert is_single_finger_capture("roll", frgp) is True


@pytest.mark.parametrize("frgp", [2, 3, 4, 5, 7, 8, 9, 10])
def test_plain_single_finger_frgp_maps_directly_to_canonical_position(frgp):
    assert canonical_finger_position("plain", frgp) == frgp
    assert is_single_finger_capture("plain", frgp) is True


def test_plain_right_thumb_frgp_maps_to_canonical_right_thumb():
    assert canonical_finger_position("plain", 11) == 1
    assert is_single_finger_capture("plain", 11) is True


def test_plain_left_thumb_frgp_maps_to_canonical_left_thumb():
    assert canonical_finger_position("plain", 12) == 6
    assert is_single_finger_capture("plain", 12) is True


@pytest.mark.parametrize("frgp", [13, 14])
def test_plain_multi_finger_captures_are_not_single_finger(frgp):
    assert canonical_finger_position("plain", frgp) is None
    assert is_single_finger_capture("plain", frgp) is False


@pytest.mark.parametrize(
    ("impression_type", "frgp", "message"),
    [
        ("rolled", 1, "Unsupported impression type"),
        ("plain", 1, "not valid for 'plain'"),
        ("roll", 11, "not valid for 'roll'"),
        ("plain", "01", "FRGP code must be an int"),
    ],
)
def test_invalid_inputs_fail_explicitly(impression_type, frgp, message):
    with pytest.raises(CanonicalFingerMappingError, match=message):
        canonical_finger_position(impression_type, frgp)
