# PPI-normalized phase-only correlation feasibility v1

## Verdict

**NO-GO.** The repository-owned candidate `ppi_normalized_phase_only_correlation` / `ppi-normalized-phase-only-correlation-v1` stopped at the mandatory physical/PPI synthetic gate. No SD300 image was decoded by the candidate and no SD300 score was computed.

The candidate was a new global spectral/correlation baseline inspired by Phase-Only Correlation. It was never intended or represented as a byte-exact or behavior-exact reconstruction of Kuglin–Hines (1975), Takita et al. (2003), or Ito et al. (2004).

## Source audit and ownership

The primary-source audit is recorded in `research/poc_preflight_v1/primary_source_manifest.json`; the component-by-component ownership map is in `research/poc_preflight_v1/specification_and_design_map.json`. The audited PDFs and rendered pages remain outside the repository. No external code, paper PDF, dataset image, representation, FFT array, crop, or correlation surface is versioned.

The POC equation and translation-peak interpretation are source-defined. Band limitation, bounded rotation, overlap, and subpixel peak estimation are source-inspired. All numerical rules, PPI normalization, fixed canvas, exact band, search grid, overlap fraction, PSR definition, failure semantics, hashes, and determinism rules are marked `project-defined frozen choice`.

## Frozen candidate specification

The algorithm and physical-scale contracts were canonically hashed before candidate use on SD300:

| Contract | Canonical SHA-256 |
| --- | --- |
| `preflight_contract.json` | `be1431d7917563288b82f4c573962aebede96518fd519dc525e1859eb68da53b` |
| `physical_scale_contract.json` | `8db5b6738b4153324c97f122582f3880d0146aa6851e13fda44fd3558460050c` |

The candidate used grayscale `uint8`, float64/complex128 computation, 500 PPI normalization, `INTER_AREA` downsampling and `INTER_CUBIC` upsampling, a fixed 512×512 center-crop/zero-pad canvas with a valid mask, valid-pixel mean/RMS normalization, and a symmetric separable Hann window. The FFT shape was exactly 512×512 with NumPy's default backward normalization and no extra padding.

Cross-power was `C = Fa * conjugate(Fb)`, normalized to unit magnitude only above `1e-12`, zero elsewhere, with DC explicitly zeroed. The inclusive radial band was 0.5–4.0 cycles/mm. Rotation candidates were -20° through +20° in 2° steps, plus at most one clamped parabolic refinement candidate. Translation used the first C-order maximum, signed cyclic indices, and separable three-sample subpixel parabolas. Valid overlap was mask intersection divided by the smaller valid support; values below 0.25 received the valid low score 0.0. The sole raw score was PSR with an 11×11 cyclic exclusion square and population sidelobe statistics. No threshold existed.

## Synthetic evidence

Identity, positive and negative translation, positive rotation, and one combined transform passed their frozen recovery tolerances. In particular, the candidate recovered a +7/-5-pixel image displacement as the -6.998041/+5.000948-pixel displacement to apply to B, which proves the frozen sign convention.

The mandatory PPI case separately rasterized the same analytic physical field and transform at 500, 1000, and 2000 PPI. Rotation, translation in millimeters, integer peak location, and finiteness passed. The raw PSR values were 11.381758 at 500 PPI, 16.096358 at 1000 PPI, and 11.836468 at 2000 PPI. The 500-to-1000 difference was **4.714600**, exceeding the frozen maximum of **3.0**.

The tolerance was not relaxed, the pattern was not replaced, and the score was not modified after the result. Under the approved fail-fast rule, this is a physical/PPI-equivalence failure and therefore fixes NO-GO. Remaining synthetic cases were not run because they cannot reverse that verdict.

## Consequences

The candidate implementation used to execute the pre-SD300 synthetic probe was discarded. There is no published method package, no pairwise-benchmark-v2 adapter, and no runnable `poc-preflight`, `poc-smoke`, `poc-repeatability`, or `run-poc` CLI. The frozen 200-image FingerCode cohort was reused by exact entry identity only as an unexecuted cohort manifest. Preparation, repeatability, and the 400-comparison smoke are explicitly `not_run` and blocked by the physical/PPI gate.

No threshold calibration, parameter sweep on SD300, ROC, AUC, EER, FAR, TAR, acceptance decision, method comparison, or accuracy claim was produced. The existing benchmark contract and every protected method/preflight tree remain outside the candidate's change scope.

## Remaining limitation and exact next task

The main unresolved question is whether absolute PSR can be made sufficiently invariant to raster sampling without tuning on SD300. The exact next task is a new, separately specified **synthetic-only POC physical-scale study**: freeze a broader analytic-field suite and evaluation rule before execution, compare physically normalized peak-quality formulations across 500/1000/2000 PPI, and do not decode SD300 or reuse the rejected v1 identity.
