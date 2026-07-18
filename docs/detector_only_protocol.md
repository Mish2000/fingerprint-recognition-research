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

The SourceAFIS location detector is `sourceafis_final_minutiae`, version
`sourceafis-final-minutiae-3.18.1-v1`. It extracts the final selected minutia
set from the documented native template serialized by `toByteArray()`. Native
template order is a deterministic SourceAFIS shuffle, not a quality ranking, so
the detector sorts only for deterministic serialization by native Y, native X,
type, direction, and source index. It does not quality-truncate the final set;
exceeding the protocol maximum is an explicit failure.

## Fixed common components

The main protocol consumes only each point's `x` and `y` fields. Native
detector response, scale, angle, and metadata do not enter the representation.
Scale and angle may be retained in `DetectedPoint` solely for future research.

The fixed choices live in `DetectorOnlyProtocolConfig`:

- `reference_ppi` and `support_size_reference_px` define one physical support
  diameter; the native diameter is multiplied by `image_ppi/reference_ppi`.
- `maximum_keypoints` is the common cap applied to the detector's existing
  ranked order.
- `orientation_policy=common_dominant_gradient_v1` assigns one protocol-owned
  orientation from the image for every detector in exactly the same way. This
  is a shared dominant-gradient policy, not the SIFT detector and not a claim
  of full reproduction of SIFT's orientation assignment. The historical names
  `sift_dominant_gradient` and `sift_dominant_gradient_v1` remain temporary
  input aliases and are normalized to the canonical name in new metadata.
- `descriptor` selects the common supplied-keypoint descriptor and
  normalization. The v1 Harris method uses RootSIFT.
- `matching_mode` and `lowe_ratio` define the common descriptor matcher.
- `geometry_model` and `ransac_threshold_reference_px` define common
  PPI-normalized verification.

RootSIFT is a descriptor normalization, not the SIFT detector. Its authoritative
implementation lives under
`fingerprint_benchmark.local_features.descriptors.rootsift`. Matching, geometry,
and scoring are implemented directly in `fingerprint_benchmark.local_features`.

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
The public RootSIFT adapter rejects every non-RootSIFT descriptor immediately;
descriptor experiments must use the generic adapter with an explicit matching
method name and version.

The SourceAFIS-location public method is
`sourceafis_final_minutiae_rootsift_geometric`, version
`sourceafis-final-minutiae-3.18.1-rootsift-geometric-detector-only-v1`.
It has no threshold. Its raw score remains `geometric_inlier_count`, identical
to the Harris branch. SourceAFIS direction and type are retained only in
detector diagnostics; common support and `common_dominant_gradient_v1`
orientation are recomputed from the image.

SourceAFIS coordinates are mapped from the actual serialized-template image
dimensions to native image dimensions using pixel centers:

```text
x_native = ((x_scaled + 0.5) * native_width / scaled_width) - 0.5
y_native = ((y_scaled + 0.5) * native_height / scaled_height) - 0.5
```

The mapping is named `scaled_to_native_pixel_center_v1`. It performs no integer
rounding, theoretical-DPI substitution, or silent clipping.

## Implementation provenance

Adapters may implement `implementation_source_paths()` to declare their full
score-producing source graph without adding method-name conditionals to the
benchmark provenance layer. Each declared file is persisted under a sorted
repository-relative path with its SHA-256, and the ordered list receives a
deterministic component SHA-256. The complete component participates in the
run's implementation hash. The Harris adapter declares the detector sources,
all common representation, descriptor, matching, geometry, and scoring sources.
Detectors may additionally declare source paths; the generic detector-only
adapter merges them. The SourceAFIS-location branch declares its Python client,
lifecycle module and detector plus the sidecar POM, Java sources, and filtered
build properties. Its adapter also declares `sidecar_jar_sha256` as a required
runtime artifact. The generic provenance layer rejects a missing or malformed
SHA before warm-up and includes it in the implementation hash.

## Research boundary

Native-keypoint pipelines that preserve a detector's own scale or orientation
answer a different question. They may be studied later under a different
protocol identity, but their results must not be mixed with
`detector_only_v1`.
