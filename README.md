# Fingerprint detector research

This repository is a minimal research base for comparing fingerprint detectors
under one common local-feature pipeline. It supports NIST SD300b and SD300c,
immutable protocol manifests, and the `pairwise-benchmark-v2` contract.

## Research methods

`detector_only_v1` fixes every stage after detection: PPI-aware support,
common dominant-gradient orientation, supplied-keypoint SIFT descriptors,
RootSIFT normalization, descriptor matching, affine geometry, and raw scoring.
Only detector locations are allowed to vary.

The first detector baseline is OpenCV GFTT with Harris scoring:

```python
from fingerprint_benchmark.detectors import (
    OpenCVGFTTHarrisDetector,
    OpenCVGFTTHarrisRootSIFTGeometricAdapter,
)
```

SourceAFIS is retained as a separate external end-to-end system baseline. It
does not participate in `detector_only_v1` and is executed through the local
Java sidecar and the generic benchmark runner.

## Installation

Create the pinned Python environment and install the package in editable mode:

```powershell
conda env create -f environment.yml
conda activate fingerprint-recognition-research
python -m pip install -e .
```

Build and test the SourceAFIS sidecar when that baseline is needed:

```powershell
Set-Location apps\sourceafis-sidecar
mvn test
mvn package
Set-Location ..\..
```

## Commands and tests

Dataset discovery and protocol validation commands are installed from
`pyproject.toml`. Inspect the generic benchmark and SourceAFIS commands with:

```powershell
fingerprint-benchmark --help
fingerprint-benchmark sourceafis-smoke --help
fingerprint-benchmark run-sourceafis --help
```

Run the repository test suite with:

```powershell
python -m pytest
```

The repository currently stores no benchmark results.

## Documentation

- [Benchmark contract](docs/benchmark_contract.md)
- [Detector-only protocol](docs/detector_only_protocol.md)
- [SourceAFIS integration](docs/sourceafis_integration_v2.md)
