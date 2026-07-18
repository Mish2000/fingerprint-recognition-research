# Detector-only comparison protocol

## Scope

`detector_only_v1` compares **detectors only**. Every detector emits the same
minimal `DetectedPoint` contract, and every downstream operation is fixed:

```text
Detector
-> ranked (x, y) locations
-> one PPI-aware physical support rule
-> one dominant-gradient orientation assignment
-> one supplied-keypoint descriptor
-> one descriptor matcher
-> one PPI-normalized geometric verifier
-> geometric inlier count
```

The protocol produces a raw similarity score and never applies an acceptance
threshold. Threshold selection and biometric decisions belong to a separate
evaluation stage.

## Detector contract

`fingerprint_benchmark.detectors` exposes `DetectedPoint`, `DetectorResult`,
and the `Detector` protocol. A detector result contains a ranked point tuple,
detector name and version, its complete configuration, diagnostics, detector
wall time only, and implementation metadata. It contains no descriptor,
matching, geometry, or decision data.

The classical baseline is `opencv_gftt_harris`, version
`opencv-gftt-harris-v1`. It calls `cv2.goodFeaturesToTrack` directly with
`useHarrisDetector=True`; all corner thresholding, ranking, and spatial
selection remain inside OpenCV. OpenCV does not return response magnitudes from
this API, so `DetectedPoint.response` is the documented zero sentinel and the
returned tuple order is the authoritative rank.

## Fixed common components

The main protocol consumes only each point's `x` and `y` fields. Native
detector response, scale, angle, and metadata do not enter the representation.
Scale and angle may be retained in `DetectedPoint` solely for future research.

The fixed choices live in `DetectorOnlyProtocolConfig`:

- `reference_ppi` and `support_size_reference_px` define one physical support
  diameter; the native diameter is multiplied by `image_ppi/reference_ppi`.
- `maximum_keypoints` is the common cap applied to the detector's existing
  ranked order.
- `orientation_policy` assigns one orientation from the image for every
  detector in exactly the same way.
- `descriptor` selects the common supplied-keypoint descriptor and
  normalization. The v1 Harris method uses RootSIFT.
- `matching_mode` and `lowe_ratio` define the common descriptor matcher.
- `geometry_model` and `ransac_threshold_reference_px` define common
  PPI-normalized verification.

RootSIFT is a descriptor normalization, not the SIFT detector. Its historical
source file remains byte-protected under `fingerprint_benchmark.sift` for
reproducibility, while `fingerprint_benchmark.local_features.descriptors`
provides the detector-neutral public import. Matching, geometry, and scoring
are exposed the same way from `fingerprint_benchmark.local_features`. The
generic modules re-export the exact protected function objects; there is no
parallel implementation and legacy imports remain valid.

## Public Harris method

The complete public method is
`opencv_gftt_harris_rootsift_geometric`, version
`opencv-gftt-harris-rootsift-geometric-v1`:

```python
from fingerprint_benchmark.detectors import (
    OpenCVGFTTHarrisRootSIFTGeometricAdapter,
)

adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
```

The detector and common pipeline have separate configuration objects:
`OpenCVHarrisConfig` and `DetectorOnlyProtocolConfig`. A future descriptor can
therefore be selected in the protocol config without changing the detector.

## Research boundary

Native-keypoint pipelines that preserve a detector's own scale or orientation
answer a different question. They may be studied later under a different
protocol identity, but their results must not be mixed with
`detector_only_v1`.
