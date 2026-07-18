# HarrisZ+ RootSIFT geometric joint 500 pilot supervisor report

Operational rule: `accepted` only when `status == ok` and integer geometric-inlier score `>= 4`. This is a frozen pilot rule, not a HarrisZ+ FAR-calibrated threshold.

| שלב | SD300b | SD300c |
|---|---:|---:|
| PLAIN מול עצמו | accepted 500; rejected 0; failures 0 | accepted 500; rejected 0; failures 0 |
| ROLL מול עצמו | accepted 500; rejected 0; failures 0 | accepted 500; rejected 0; failures 0 |
| PLAIN מול ROLL המתאים | accepted 63; rejected 437; failures 0 | accepted 40; rejected 460; failures 0 |
| PLAIN מול ROLL של הנבדק הבא | incorrectly accepted 1; correctly rejected 499; failures 0 | incorrectly accepted 1; correctly rejected 499; failures 0 |

## 1. Record counts

| Dataset | PLAIN | ROLL |
|---|---:|---:|
| SD300B | 500 | 500 |
| SD300C | 500 | 500 |

## 2. PLAIN self-comparisons

| Dataset | Accepted | Rejected | Failures | Removed | Remaining |
|---|---:|---:|---:|---:|---:|
| SD300B | 500 | 0 | 0 | 0 | 500 |
| SD300C | 500 | 0 | 0 | 0 | 500 |

## 3. ROLL self-comparisons

| Dataset | Accepted | Rejected | Failures | Removed | Remaining |
|---|---:|---:|---:|---:|---:|
| SD300B | 500 | 0 | 0 | 0 | 500 |
| SD300C | 500 | 0 | 0 | 0 | 500 |

## 4. PLAIN-to-corresponding-ROLL record matching

Self filtering was performed independently for each dataset. No identity was replaced.

| Dataset | Survivors | Excluded after self | Ground truth same subject/finger | Ground truth wrong | HarrisZ+ matching | HarrisZ+ non-matching | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| SD300B | 500 | 0 | 500 | 0 | 63 | 437 | 0 |
| SD300C | 500 | 0 | 500 | 0 | 40 | 460 | 0 |

## 5. PLAIN versus corresponding ROLL

| Dataset | Total | Accepted | Rejected | Failures | Acceptance % | Score mean | Score median | Mean method compare ms | Mean total pair ms | Mean keypoints A/B | Mean mutual matches | Mean inliers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SD300B | 500 | 63 | 437 | 0 | 12.600000000 | 2.142000000 | 2.000000000 | 22.095900400 | 4233.560185200 | 2996.800000000/3000.000000000 | 6.014000000 | 2.142000000 |
| SD300C | 500 | 40 | 460 | 0 | 8.000000000 | 1.032000000 | 0.000000000 | 17.464633800 | 2704.788194400 | 2999.180000000/3000.000000000 | 2.414000000 | 1.032000000 |

## 6. PLAIN versus next subject's ROLL

Pairing method: next survivor within the same canonical finger position, ordered by `selection_index`, circular shift `1`.

| Dataset | Total | Incorrectly accepted | Correctly rejected | Failures | False-match % | Score mean | Score median | Mean method compare ms | Mean total pair ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SD300B | 500 | 1 | 499 | 0 | 0.200000000 | 1.022000000 | 2.000000000 | 23.746240400 | 1905.268723200 |
| SD300C | 500 | 1 | 499 | 0 | 0.200000000 | 0.126000000 | 0.000000000 | 17.406986400 | 2701.040675800 |

Timing note: HarrisZ+ preparation uses a CUDA detector. Earlier SourceAFIS and SIFT pilots may use different backends, so this report makes no direct CPU-efficiency ranking.
