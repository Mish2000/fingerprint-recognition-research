"""Generic access to the protected, repository-native RootSIFT transform."""

# The implementation remains byte-protected at its historical import path.
from fingerprint_benchmark.sift.descriptors import (  # noqa: F401
    process_descriptors,
    rootsift,
    validate_descriptors,
)

__all__ = ["process_descriptors", "rootsift", "validate_descriptors"]
