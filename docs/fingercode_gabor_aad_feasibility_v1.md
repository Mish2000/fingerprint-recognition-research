# FingerCode Gabor/AAD feasibility preflight v1

Status: **NO-GO before implementation and before SD300 inference**

## Scope and decision

This preflight evaluated whether the classical Jain-Prabhakar-Hong-Pankanti FingerCode matcher can be implemented as a faithful, PPI-aware, deterministic full-system benchmark without inventing missing method details. The answer is no. The primary papers specify much of the tessellation, normalization, filter bank, AAD feature, and approximate rotation-template strategy, but they do not provide a complete operational contract for the reference frame, byte vector, and resampled rotation. The requested policy explicitly forbids replacing those elements with a modern core detector, orientation estimator, or guessed interpolation rule.

No matcher code, CLI registration, vector artifact, SD300 preparation run, or SD300 comparison run was created.

## Frozen repository baseline

- Starting branch: `main`
- Starting commit and `origin/main`: `e0fec284546f8b937fac3c570612fc4e2245937c`
- Baseline tests: 403 passed, 1 skipped
- Detector-only Joint-500 protocol SHA-256: `4d53ba3466524f6a0399e57f62edc1bac58fb2bb425e18bdbd4ef95373e7ec23`
- Detector build check and validation: pass
- SIFT historical parity: 63/63, zero mismatches
- GFTT-Harris full-system parity: 93/93, zero mismatches
- Runtime: Python 3.11.15, NumPy 2.2.6, OpenCV 4.12.0

Protected tree Git object IDs are frozen in `preflight_contract.json`.

## Primary sources

The audit used the complete 2000 IEEE Transactions on Image Processing paper from the MSU author archive and a complete seven-page reproduction of the 1999 IEEE CVPR paper from the USPTO PTAB public record. The IEEE records and DOI metadata were cross-checked. PDFs, page renders, and extracted text remained outside the repository. Their byte hashes, page ranges, equations, origins, and a rejected mislabeled MSU link are recorded in `primary_source_manifest.json`.

No external implementation, code, model, figure, or prose was copied into the repository.

## What the primary sources do specify

For the 640-component MSU setting at 500 PPI, the papers support:

- an excluded 20-pixel central circle;
- five retained bands, each 20 pixels wide;
- sixteen sectors per band and an outer radius of 120 pixels;
- sector-local normalization to target mean 100 and target variance 100;
- eight even-symmetric Gabor filters at 0, 22.5, 45, 67.5, 90, 112.5, 135, and 157.5 degrees;
- frequency `f=1/K`, with `K` approximately 10 pixels at 500 PPI and an example value `f=0.1`;
- `sigma_x=sigma_y=4.0`;
- a 33x33 spatial mask retaining coefficients with absolute value greater than 0.05;
- AAD as the mean absolute deviation of each filtered sector from its filtered-sector mean;
- Euclidean distance with lower values more similar;
- ten enrollment templates: five cyclic sector rotations plus five based on an image rotated by 11.25 degrees.

## Mandatory blockers

1. **Reference-point detector:** the multi-resolution outline permits different gradient operators; the empirically chosen integration regions are only drawn, without exact dimensions; segmentation, borders, grid anchoring, search geometry, and ties are not frozen.
2. **Reference orientation:** the CVPR paper defines a symmetry axis but says it is not used. The TIP implementation assumes upright acquisition and treats automatic orientation detection as future work.
3. **Canonical alignment:** there is no exact source-defined image canonicalization contract.
4. **Vector bytes:** 640 dimensions and one-byte storage are claimed, but the linear orientation/sector serialization and the quantization, clipping, and rounding rule are absent.
5. **Rotation compensation:** cyclic permutation is described, but the 11.25-degree image rotation lacks interpolation, canvas, center, and border rules.
6. **PPI support:** physical scaling is straightforward for scalar radii, widths, sigma, and wavelength, but the papers do not say whether a 33-sample odd mask scales by sample count or centered half-width.

Any one of the first five is independently sufficient for the requested NO-GO.

## PPI findings

The known source quantities were converted without resizing source images. At 1000/2000 PPI, respectively: 20 pixels becomes 40/80, 120 becomes 240/480, sigma 4 becomes 8/16, and wavelength 10 becomes 20/40. Frequency becomes 0.05/0.025 cycles per native pixel. Those arithmetic conversions have zero rounding error. A synthetic implementation test was not run because the algorithm contract is incomplete.

## Frozen smoke cohort

The cohort manifest contains exactly 200 existing images: the first five identities by frozen selection rank for each of ten canonical finger positions, crossed with SD300b/SD300c and plain/roll acquisition. It stores only repository-safe relative dataset paths and per-file SHA-256 hashes. Hashing source bytes is provenance work, not model inference. No image, crop, mask, vector, or visualization is stored in the repository.

## Required reports

- `reference_point_report.json`: `not_run`; zero of 25 requested images evaluated.
- `repeatability_report.json`: `not_run`; zero processes executed.
- `smoke_report.json`: preparation and inference both `not_run`.
- `physical_scale_contract.json`: known conversions recorded; synthetic checks `not_run`.

## Boundary preservation

The SourceAFIS sidecar, restored SIFT matcher, full GFTT-Harris matcher, detector-only Joint-500 protocol, existing benchmark v2 CLI, and `research/deepprint_style_preflight_v1/` were not modified.

## Conclusion

The classical papers are scientifically informative but not sufficient to reproduce the requested full matcher byte-for-byte without choosing behavior that the sources do not specify. Under the frozen no-substitution rule, implementation would misrepresent a modern interpretation as the historical method. The compliant result is therefore a documented NO-GO.

The next task is exactly: **Design a separate coreless Gabor texture matcher or proceed to the spectral/correlation family**.
