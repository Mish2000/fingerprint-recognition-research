"""Deterministic descriptor post-processing."""

from __future__ import annotations

import numpy as np


def validate_descriptors(descriptors: np.ndarray) -> np.ndarray:
    array = np.asarray(descriptors)
    if array.ndim != 2 or array.shape[1] != 128:
        raise ValueError(f"SIFT descriptor array must have shape (N, 128); got {array.shape}.")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"SIFT descriptors must be floating point; got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError("SIFT descriptor array contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float32)


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    array = validate_descriptors(descriptors)
    norms = np.sum(np.abs(array), axis=1, keepdims=True, dtype=np.float64)
    normalized = np.divide(
        array,
        norms,
        out=np.zeros_like(array, dtype=np.float32),
        where=norms > 0.0,
    )
    transformed = np.sqrt(normalized, out=np.zeros_like(normalized), where=normalized >= 0.0)
    return validate_descriptors(transformed)


def process_descriptors(descriptors: np.ndarray, mode: str) -> np.ndarray:
    if mode == "standard":
        return validate_descriptors(descriptors)
    if mode == "rootsift":
        return rootsift(descriptors)
    raise ValueError(f"Unsupported descriptor mode: {mode!r}.")
