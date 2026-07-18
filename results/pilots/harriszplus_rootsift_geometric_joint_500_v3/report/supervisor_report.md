# HarrisZ+ RootSIFT geometric joint 500 pilot supervisor report

Operational rule: `accepted` only when `status == ok` and integer geometric-inlier score `>= 4`. This is a frozen pilot rule, not a HarrisZ+ FAR-calibrated threshold.

| שלב | SD300b | SD300c |
|---|---:|---:|
| PLAIN מול עצמו | accepted 500; rejected 0; failures 0 | accepted 500; rejected 0; failures 0 |
| ROLL מול עצמו | accepted 500; rejected 0; failures 0 | accepted 500; rejected 0; failures 0 |
| PLAIN מול ROLL המתאים | accepted 64; rejected 436; failures 0 | accepted 1; rejected 499; failures 0 |
| PLAIN מול ROLL של הנבדק הבא | incorrectly accepted 3; correctly rejected 497; failures 0 | incorrectly accepted 0; correctly rejected 500; failures 0 |

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
| SD300B | 500 | 0 | 500 | 0 | 64 | 436 | 0 |
| SD300C | 500 | 0 | 500 | 0 | 1 | 499 | 0 |

## 5. PLAIN versus corresponding ROLL

| Dataset | Total | Accepted | Rejected | Failures | Acceptance % | Score mean | Score median | Mean method compare ms | Mean total pair ms | Mean keypoints A/B | Mean mutual matches | Mean inliers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SD300B | 500 | 64 | 436 | 0 | 12.800000000 | 2.148000000 | 2.000000000 | 20.520023200 | 1759.307524600 | 2996.962000000/3000.000000000 | 6.032000000 | 2.148000000 |
| SD300C | 500 | 1 | 499 | 0 | 0.200000000 | 1.472000000 | 2.000000000 | 17.804067400 | 4326.448917600 | 3000.000000000/3000.000000000 | 4.212000000 | 1.472000000 |

## 6. PLAIN versus next subject's ROLL

Pairing method: next survivor within the same canonical finger position, ordered by `selection_index`, circular shift `1`.

| Dataset | Total | Incorrectly accepted | Correctly rejected | Failures | False-match % | Score mean | Score median | Mean method compare ms | Mean total pair ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SD300B | 500 | 3 | 497 | 0 | 0.600000000 | 1.058000000 | 2.000000000 | 20.783217400 | 1757.314248000 |
| SD300C | 500 | 0 | 500 | 0 | 0.000000000 | 1.130000000 | 2.000000000 | 17.303792400 | 4332.735633000 |

Timing note: HarrisZ+ preparation uses a CUDA detector. Earlier SourceAFIS and SIFT pilots may use different backends, so this report makes no direct CPU-efficiency ranking.
