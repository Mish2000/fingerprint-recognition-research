# Full-system SIFT geometric baseline (restored)

Restored historical full-system SIFT geometric baseline.

This document describes a method that already existed and was already evaluated.
Nothing here is new, improved, or re-tuned. The algorithm modules were recovered
byte-for-byte from the commit that produced the historical result bundles, and
the restoration is gated on exact parity against those bundles.

| Field | Value |
| --- | --- |
| `method_name` | `sift_geometric` |
| `method_version` | `sift-geometric-v1` |
| `score_direction` | `higher_is_more_similar` |
| Historical source commit | `e95ab6f3c5685df1222f32b7260b2e340e6b0d5e` |
| Restoration commit | the commit that adds this file, on `main` |
| Frozen `config_hash` | `51cd949cbfd84eebfe7cc5057f3621d6ec20be855905db743f42e8945adc541a` |

## Why `e95ab6f` is the authoritative source

The SIFT tree was deleted by `f58fe8b` ("Reduce repository to detector-only
research"). `e95ab6f` is the latest commit before that deletion at which the
code and the artifacts still agree exactly:

* All nine `src/fingerprint_benchmark/sift/*.py` files hash to the values the six
  historical bundles recorded in
  `implementation_hash_components.method_support_source_sha256`.
* `contract.py`, `runner.py`, `bundle.py`, `hashing.py`, `io.py`, `manifest.py`,
  `preflight.py` and `provenance.py` likewise hash to the values those bundles
  recorded.
* The six bundles, the tests, the docs and the frozen development config are all
  present and byte-identical to their state at the last pre-deletion commit.

The two later commits are not authoritative. `02935b3` rewrote
`sift/__init__.py` into a lazy-import shim, and `01f27ab` changed `contract.py`
and `provenance.py`; after either, at least one recorded source hash no longer
matches the artifacts.

## Pipeline

```text
grayscale image (cv2.IMREAD_GRAYSCALE)
  -> native image policy: no resize, no CLAHE, no normalization, no mask
  -> cv2.SIFT_create detection            (native SIFT keypoints)
  -> native SIFT scale (kp.size) and orientation (kp.angle)
  -> SIFT descriptors, RootSIFT post-processing (L1 normalize, then sqrt)
  -> BFMatcher(NORM_L2, crossCheck=False), knnMatch k=2, both directions
  -> Lowe ratio 0.75, mutual consistency filter
  -> cv2.estimateAffinePartial2D, RANSAC, PPI-normalized coordinates
  -> raw score = geometric inlier count
```

`compare()` performs no thresholding, no normalization and no accept/reject
decision. It returns a raw score only.

### Preprocessing

`image_policy = "native"`. The grayscale image is used at its native
resolution and native intensities; the only operation is
`np.ascontiguousarray`. `mask_mode = "none"`, so no valid-region mask is
computed. The `reference_reproduction` policy and the `valid_region` mask exist
in the restored configuration space but are not part of this frozen identity.

### SIFT parameters

| Parameter | Value |
| --- | --- |
| `nfeatures` | 3000 |
| `nOctaveLayers` | 3 |
| `contrastThreshold` | 0.04 |
| `edgeThreshold` | 10.0 |
| `sigma` | 1.6 |

Keypoints, scale and orientation all come from OpenCV SIFT itself via
`sift.detectAndCompute`. No external detector is involved.

### Descriptors and matching

| Property | Value |
| --- | --- |
| `descriptor_mode` | `rootsift` (L1 normalize, then element-wise square root) |
| Matcher | `cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)`, `knnMatch(k=2)` |
| `lowe_ratio` | 0.75, applied as `first.distance < ratio * second.distance` |
| `matching_mode` | `mutual` — forward matches kept only if the reverse pass agrees |
| `minimum_descriptors` | 2 |
| `minimum_geometry_matches` | 3 |

### Geometry

| Property | Value |
| --- | --- |
| Model | `affine_partial_2d` (`cv2.estimateAffinePartial2D`) |
| Coordinates | scaled by `reference_ppi / image_ppi`, `reference_ppi = 1000.0` |
| `ransacReprojThreshold` | 3.0 reference pixels |
| `confidence` | 0.99 |
| `maxIters` | 2000 |
| `refineIters` | 10 |
| RNG policy | `cv2.setRNGSeed(0)` immediately before every geometry fit |

### Score formula

```text
raw_score = geometric_inlier_count
```

The frozen `score_mode` is `geometric_inlier_count`: the score is the number of
RANSAC inliers, as a float. The other three modes in the restored scoring module
(`geometric_inlier_ratio`, `inliers_over_min_keypoints`,
`inliers_times_inlier_ratio_times_log1p_matches`) are computed and reported under
`score_components` for diagnostics, but are not the frozen score.

Note that the dataclass defaults in `config.py` are **not** the frozen identity.
The historical pilot selected `descriptor_mode="rootsift"` and
`score_mode="geometric_inlier_count"`, both of which differ from those defaults.
The frozen values are declared explicitly in `sift/restored.py::frozen_config`
and asserted against the historical bundle metadata in
`tests/test_sift_geometric_restoration.py`.

## Failure semantics

Technical failures and legitimate non-matches are kept strictly separate.

**Technical failure** — `status` is a failure status, `raw_score` is absent
(empty), `error_code` is explicit:

| `error_code` | Cause |
| --- | --- |
| `image_read_failure` | OpenCV could not read the image |
| `missing_or_invalid_ppi` | manifest PPI missing, non-finite or non-positive |
| `image_preparation_failure` | preprocessing raised |
| `missing_descriptors` | SIFT produced no descriptors |
| `too_few_descriptors` | fewer than `minimum_descriptors` |
| `invalid_descriptors` | descriptor validation failed |
| `descriptor_keypoint_mismatch` | keypoint/descriptor counts differ |
| `descriptor_matching_failure` | matching raised |
| `raw_score_failure` | score was non-finite or negative |
| `representation_identity_mismatch` | representation from another method/version |
| `representation_format_mismatch` | representation payload of the wrong type |

**Valid comparison with no match** — `status = ok`, `raw_score = 0.0`, and
`compare_diagnostics.geometry_failure_reason` carries the explicit reason:
`insufficient_geometry_matches`, `opencv_geometry_error`,
`model_estimation_failure` or `invalid_inlier_mask`. Zero inliers from a
successful fit are also `status = ok` with score `0.0`.

A zero score is never reported as a technical failure, and a technical failure is
never reported as a zero score.

## Difference from the SIFT descriptor inside `detector_only_v1`

Both methods end in a RootSIFT descriptor, mutual matching and an inlier-count
score, so they are easy to confuse. They differ in where the keypoints, scale and
orientation come from:

| | `sift_geometric-v1` | `detector_only_v1` |
| --- | --- | --- |
| Keypoint locations | native SIFT (DoG extrema) | external detector (Harris / SourceAFIS minutiae) |
| Scale / support | native SIFT `kp.size` | common fixed physical diameter scaled by PPI |
| Orientation | native SIFT `kp.angle` | common orientation policy |
| Descriptor | RootSIFT at native SIFT keypoints | RootSIFT at supplied keypoints |

`sift_geometric-v1` is a full-system method: it owns its own detection, scale and
orientation. It is not, and must not become, another `DetectorOnlyAdapter`
configuration.

## Integration with the current architecture

The nine algorithm modules are byte-identical to `e95ab6f`. Two adaptations were
needed because the surrounding architecture changed after the historical runs,
and both live in `src/fingerprint_benchmark/sift/restored.py`:

1. **Frozen configuration.** The historical runs loaded their config from
   `results/sift_geometric/development/sift_geometric_config.json`. Result
   artifacts are no longer tracked, so the same values are declared as code in
   `frozen_config()` and asserted against the historical bundle metadata.
2. **Provenance source declaration.** `provenance.py` replaced its hardcoded
   `sift_geometric` source-hash branch with the generic
   `ImplementationSourceProvider` capability, so
   `RestoredSiftGeometricAdapter` declares its own sources via
   `implementation_source_paths()`.

`RestoredSiftGeometricAdapter` subclasses the historical adapter and overrides
none of `metadata()`, `prepare()` or `compare()`; a test enforces this. Bundle
publication, manifest parsing, hashing, the runner and the result schema are all
the existing ones on `main` — nothing was duplicated.

### Identity: what reproduces and what does not

* **`config_hash` reproduces exactly**: `51cd949c…`, in all six bundles. This is
  the frozen algorithm and configuration identity.
* **`implementation_hash` does not reproduce**, by construction. It mixes in
  `contract.py` and `provenance.py` (both changed after the runs) and the removed
  per-method source branch. It is reported as a declared difference with its
  reason, never silently normalized. Restoration identity is instead carried by
  `adapter_declared_implementation_sources.component_sha256`.

## Parity procedure

```powershell
git worktree add --detach C:\fingerprint-recognition-research-sift-history e95ab6f

fingerprint-benchmark sift-geometric-parity `
    --historical-results-root C:\fingerprint-recognition-research-sift-history\results `
    --repository-root C:\fingerprint-recognition-research
```

For each of the six historical bundles the gate:

1. verifies the bundle declares `sift_geometric` / `sift-geometric-v1`, and that
   the protocol manifest still hashes to what the run recorded;
2. selects pairs deterministically by `index_i = floor(i * N / 10)` for
   `i = 0..9`, then adds one pair per outcome class (positive score, zero score
   with `status=ok`, technical failure) if the stride missed it;
3. re-runs the restored code on those pairs through the current runner's own
   `_execute_pair`, so row serialization is identical by construction;
4. requires **exact** equality — no tolerance — of pair id, status, error code,
   error message, raw score text, raw score numeric value, all deterministic
   diagnostics, method name/version, config values and `config_hash`.

Only wall-clock timings are excluded (diagnostics keys ending in `_ms`), plus
`implementation_hash` as the one declared difference above.

The report is written to
`results/restoration_preflight/sift_geometric_v1/parity_report.json`. It is not
tracked in git; `results/` is ignored.

## Parity result

Run on Python 3.11.15, NumPy 2.2.6, OpenCV 4.12.0 (opencv-python 4.12.0.88) —
the same versions the historical bundles recorded.

| Bundle | Pairs | Matched | Status |
| --- | --- | --- | --- |
| sd300b / plain_self | 12 | 12 | pass |
| sd300b / roll_self | 10 | 10 | pass |
| sd300b / plain_roll | 11 | 11 | pass |
| sd300c / plain_self | 10 | 10 | pass |
| sd300c / roll_self | 10 | 10 | pass |
| sd300c / plain_roll | 10 | 10 | pass |
| **Total** | **63** | **63** | **pass** |

Outcome coverage across the sample: 54 positive scores, 7 zero scores with
`status=ok`, 2 technical failures (`too_few_descriptors`). `config_hash`
reproduced in all six bundles. Zero mismatches.

Repeatability: one self pair, one Plain–Roll pair with a positive score and one
Plain–Roll pair with a zero score were each run in three fresh processes. Status,
error code, raw score and deterministic diagnostics were identical across all
three.

## Running it

```powershell
# single-pair smoke comparison
fingerprint-benchmark sift-geometric-smoke --manifest protocols\sd300b\plain_self.csv

# one full pairwise-benchmark-v2 bundle
fingerprint-benchmark run-sift-geometric --dataset sd300b --protocol plain_roll
```

Running `sift_geometric-v1` over the shared full-system cohort is deliberately
out of scope for the restoration and is planned separately.
