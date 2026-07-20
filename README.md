# Fingerprint recognition research

This repository is a research base for full fingerprint matchers and for
controlled detector ablations. It supports NIST SD300b and SD300c, immutable
protocol manifests, and the `pairwise-benchmark-v2` contract.

## Research methods

### Full systems

- **SourceAFIS full** uses the native SourceAFIS template and verification
  pipeline through the pinned sidecar integration.
- **SIFT geometric full** is the restored `sift_geometric-v1` baseline with
  native SIFT detection, RootSIFT, mutual matching and affine verification.
- **GFTT-Harris--RootSIFT geometric full** is
  `gftt_harris_rootsift_geometric` / `gftt-harris-rootsift-geometric-v1`: a
  complete single-scale Harris-location matcher with fixed dominant-gradient
  orientation, supplied-keypoint SIFT, RootSIFT, mutual matching,
  PPI-normalized partial-affine RANSAC and raw inlier-count scoring.

The GFTT-Harris full method has no decision threshold and makes no claim of
final biometric accuracy. See
[`docs/gftt_harris_rootsift_geometric_full_v1.md`](docs/gftt_harris_rootsift_geometric_full_v1.md).

### Detector-only ablation

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

The second detector branch uses only the final minutia locations selected by
SourceAFIS 3.18.1, then passes those locations through the exact same
`detector_only_v1` support, orientation, RootSIFT, matching, geometry, and
inlier-count score:

```python
from fingerprint_benchmark.detectors import (
    SourceAfisFinalMinutiaeDetector,
    SourceAfisFinalMinutiaeRootSIFTGeometricAdapter,
)
```

Minutia type, SourceAFIS direction, template order, response sentinel, and
metadata are diagnostic only and do not enter the common representation.
SourceAFIS is also retained unchanged as a separate end-to-end baseline through
`/extract-template` and `/verify`; the detector branch does not replace it.
For detector-only work, OpenCV `IMREAD_GRAYSCALE` uint8 pixels are canonical and
are sent unchanged to SourceAFIS through `/extract-template-raw` and
`/extract-final-minutiae`. The preflight gates those two raw paths and retains
native encoded/raw equality only as a decoder-ingestion diagnostic.

The method-neutral screening cohort is `detector_only_joint_500_v1`: 500
identities, 50 per canonical finger position, one finger per subject, with the
same logical identities and impostor pairing in SD300b and SD300c. It is a
development/screening protocol, not held-out evaluation.

## Installation

Create the pinned Python environment and install the package in editable mode:

```powershell
conda env create -f environment.yml
conda activate fingerprint-recognition-research
python -m pip install -e .
```

The SourceAFIS JAR is built locally and is not stored in Git. Build and test
the sidecar before using managed SourceAFIS commands:

```powershell
mvn -f apps\sourceafis-sidecar\pom.xml test
mvn -f apps\sourceafis-sidecar\pom.xml package
```

`mvn package` creates the CLI's default JAR under
`apps\sourceafis-sidecar\target\`; Maven output is ignored by Git.

## Commands and tests

Dataset discovery and protocol validation commands are installed from
`pyproject.toml`. Inspect the generic benchmark and SourceAFIS commands with:

```powershell
fingerprint-benchmark --help
fingerprint-benchmark sourceafis-smoke --help
fingerprint-benchmark run-sourceafis --help
fingerprint-benchmark detector-joint500 --help
fingerprint-benchmark detector-joint500 build --check
fingerprint-benchmark detector-joint500 validate
fingerprint-benchmark gftt-harris-smoke --help
fingerprint-benchmark run-gftt-harris --help
fingerprint-benchmark gftt-harris-parity --help
fingerprint-benchmark gftt-harris-repeatability --help
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
- [Joint-500 screening protocol](docs/detector_only_joint_500_v1.md)
- [Full-system SIFT geometric baseline (restored)](docs/sift_geometric_full.md)
- [Full-system GFTT-Harris--RootSIFT geometric matcher](docs/gftt_harris_rootsift_geometric_full_v1.md)
