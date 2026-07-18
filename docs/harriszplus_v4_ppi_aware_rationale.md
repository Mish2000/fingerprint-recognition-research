# HarrisZ+ v4 PPI-aware rationale

## Why v4 is a separate method

`harriszplus_rootsift_geometric_v3` passed its frozen engineering preflight and
is technically valid under its published contract. Its postmortem nevertheless
found strong evidence for a scale mismatch: the same HarrisZ+ scale index at
2000 PPI covered approximately half the physical size that it covered at
1000 PPI. RANSAC was already PPI-normalized; the remaining mismatch was in
detector scale, orientation support, and descriptor support.

v4 is therefore a new method/configuration, not a silent correction to v3:

- method: `harriszplus_rootsift_geometric_ppi_aware`
- version: `harriszplus-rootsift-geometric-ppi-aware-v4`
- parent: `harriszplus-rootsift-geometric-v3`
- reference PPI: 1000
- runtime factor: `spatial_scale = manifest_ppi / 1000.0`

SD300b consequently uses `spatial_scale=1.0`, while SD300c uses
`spatial_scale=2.0`. The PPI is read from every manifest row and is stored with
the factor in the prepared representation and its deterministic hash.

The frozen 500-identity selection is not used for parameter choice. The
physical contract and engineering gates are frozen and executed on the same
ten development identities used by v3 before any v4 500-pair result may run.

## Scope of the change

The HarrisZ+ equations, five scale indices, thresholds, eigenvalue rule,
subpixel formula, rank/tie rules, 3000-point cap, native grayscale policy,
internal Lanczos doubling for indices 0 and 1, RootSIFT, Lowe ratio 0.75,
mutual BF-L2 matcher, partial-affine RANSAC, inlier-count score, threshold 4,
float32/determinism policy, cold-pair execution, and no-cache policy are
unchanged.

No full-image downsampling or upsampling is introduced. OpenCV SIFT computes
the unchanged descriptor on the original native image at explicitly supplied
PPI-scaled `KeyPoint.size` values.

## Spatial-parameter audit

| v3 quantity | v4 classification | v4 rule |
|---|---|---|
| differentiation sigma | scaled by `p` | `sigma_d_native = p * sigma_d_reference` |
| integration sigma | scaled by `p` | `sigma_i_native = p * sigma_i_reference` |
| Gaussian radius/size | scaled by `p` | scale the reference sampled support radius; preserve the physical truncation support |
| per-scale greedy suppression | scaled by `p` | scale the reference rounded distance |
| scale-0/1 duplicate radius | scaled by `p` | `p * 1.0` native pixels |
| OpenCV `KeyPoint.size` | scaled by `p` | twice the PPI-scaled output integration sigma |
| orientation weighting sigma/radius | scaled by `p` | derive from scaled keypoint support; scale the reference rounded radius |
| descriptor support estimate | scaled by `p` | derive continuously from scaled `KeyPoint.size` |
| descriptor-safe border margin | scaled by `p` | scale the reference descriptor radius plus one pixel; apply before the uniform cap |
| uniform-selection `q` | dimension-derived | use Eq. 14 on native height/width; do not multiply by `p` again |
| native image dimensions | dimension-derived | unchanged inputs; SD300c dimensions already encode its sampling density |
| internal 2× Lanczos array | fixed dimensionless factor | only indices 0 and 1; working sigma includes both `p` and the 2× internal factor |
| strict 7×7 local-maximum topology | dimensionless algorithmic topology | unchanged; it is the frozen response-selection neighborhood, not an independently tuned physical support |
| subpixel parabolic offset | dimensionless local pixel-cell offset | unchanged; the implementation has no clipping distance |
| response/edge/eigen thresholds, ratios, scale indices and ranking | dimensionless | unchanged |
| RANSAC threshold | already PPI-normalized | unchanged at 3 reference pixels |

The explicit Gaussian-radius policy is
`round(p * ceil(3 * sigma_reference_working))`. This is derived from the
PPI-scaled physical Gaussian support and avoids unequal B/C support caused by
re-rounding the two native radii independently.

Uniform selection is recorded explicitly as:

> uniform q is dimension-derived and not independently PPI-scaled

## Physical contract

Before matcher execution, v4 publishes
`physical_scale_contract_v4.json`. For differentiation/integration sigma,
Gaussian support, suppression, duplicate radius, keypoint size, orientation
radius, descriptor support, border margin, uniform `q`, and RANSAC, the
physical B/C values must satisfy:

`abs(mm_B - mm_C) <= max(0.001 mm, 1% of the reference value)`

Failure stops the workflow before matching and forbids the 500-identity pilot.
For RANSAC, 3 pixels at 1000 PPI and 6 pixels at 2000 PPI both represent
0.0762 mm.

## Border and orientation behavior

The orientation histogram remains the v3 signed-gradient, 36-bin,
Gaussian-weighted, circularly interpolated and smoothed construction with one
dominant angle. Only its sigma and integer support radius follow the PPI-scaled
keypoint support. Diagnostics retain the radius in pixels, radius in
millimetres, and histogram window size per scale.

Candidates whose estimated SIFT descriptor window crosses the native image
boundary are removed per scale after subpixel/eigen filtering and before
cross-scale duplicate removal and the uniform 3000-point cap. This makes border
handling explicit and prevents OpenCV border truncation from becoming an
uncontrolled PPI-dependent support change.

## Decision and interpretation

The adapter returns only `geometric_inlier_count`. Operational acceptance
remains:

`status == ok && geometric_inlier_count >= 4`

The threshold is inherited, not recalibrated. v3/v4 development and pilot
comparisons are diagnostic ablations only. Genuine acceptance must be shown
together with negative false accepts; neither development scores nor the 500
identities authorize tuning.
