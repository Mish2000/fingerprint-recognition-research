"""Clean-room HarrisZ+ detector with RootSIFT geometric integration."""

from .config import HarrisZPlusConfig
from .cuda_detector import HarrisZPlusCUDADetector, detect_harriszplus_cuda
from .reference_cpu import HarrisZPlusReferenceCPU, detect_harriszplus_cpu
from .types import DetectedKeypoint, DetectorResult, ScaleSpec, SelectionCandidate

__all__ = [
    "HarrisZPlusConfig",
    "HarrisZPlusReferenceCPU",
    "HarrisZPlusCUDADetector",
    "DetectedKeypoint",
    "DetectorResult",
    "ScaleSpec",
    "SelectionCandidate",
    "detect_harriszplus_cpu",
    "detect_harriszplus_cuda",
]
