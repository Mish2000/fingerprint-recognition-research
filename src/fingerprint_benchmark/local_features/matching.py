"""Detector-neutral access to the protected descriptor matcher."""

from fingerprint_benchmark.sift.matching import MatchRecord, MatchSet, match_descriptors

__all__ = ["MatchRecord", "MatchSet", "match_descriptors"]
