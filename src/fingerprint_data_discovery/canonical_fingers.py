"""Canonical single-finger anatomical positions for NIST SD300 FRGP codes.

This module is intentionally a projection layer: it does not replace raw SD300
FRGP values or the readable raw finger-position labels used by discovery.
"""

from __future__ import annotations


CANONICAL_FINGER_POSITIONS = {
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

ROLL_CANONICAL_BY_FRGP = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
}

PLAIN_CANONICAL_BY_FRGP = {
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 1,
    12: 6,
}

PLAIN_MULTI_FINGER_FRGPS = frozenset({13, 14})

_CANONICAL_BY_IMPRESSION = {
    "roll": ROLL_CANONICAL_BY_FRGP,
    "plain": PLAIN_CANONICAL_BY_FRGP,
}

_VALID_FRGP_BY_IMPRESSION = {
    "roll": frozenset(ROLL_CANONICAL_BY_FRGP),
    "plain": frozenset(PLAIN_CANONICAL_BY_FRGP) | PLAIN_MULTI_FINGER_FRGPS,
}


class CanonicalFingerMappingError(ValueError):
    """Raised when an impression type or FRGP code cannot be mapped explicitly."""


def _validate_impression_and_frgp(impression_type: str, frgp: int) -> None:
    if impression_type not in _VALID_FRGP_BY_IMPRESSION:
        allowed = ", ".join(sorted(_VALID_FRGP_BY_IMPRESSION))
        raise CanonicalFingerMappingError(
            f"Unsupported impression type {impression_type!r}; expected one of {allowed}."
        )

    if type(frgp) is not int:
        raise CanonicalFingerMappingError(f"FRGP code must be an int, got {type(frgp).__name__}.")

    valid_frgp = _VALID_FRGP_BY_IMPRESSION[impression_type]
    if frgp not in valid_frgp:
        allowed = ", ".join(f"{code:02d}" for code in sorted(valid_frgp))
        raise CanonicalFingerMappingError(
            f"FRGP {frgp!r} is not valid for {impression_type!r}; allowed codes are {allowed}."
        )


def canonical_finger_position(impression_type: str, frgp: int) -> int | None:
    """Return the canonical position 1-10 for a single-finger capture.

    Valid multi-finger plain captures, FRGP 13 and 14, return ``None`` because
    they must not be projected into artificial single-finger records.
    """

    _validate_impression_and_frgp(impression_type, frgp)
    return _CANONICAL_BY_IMPRESSION[impression_type].get(frgp)


def is_single_finger_capture(impression_type: str, frgp: int) -> bool:
    """Return whether the SD300 capture explicitly represents one anatomical finger."""

    return canonical_finger_position(impression_type, frgp) is not None


def is_plain_multi_finger_capture(impression_type: str, frgp: int) -> bool:
    """Return whether the SD300 capture is a plain multi-finger capture."""

    _validate_impression_and_frgp(impression_type, frgp)
    return impression_type == "plain" and frgp in PLAIN_MULTI_FINGER_FRGPS
