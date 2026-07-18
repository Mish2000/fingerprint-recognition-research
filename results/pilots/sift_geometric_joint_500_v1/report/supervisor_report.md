# SIFT geometric joint 500 pilot supervisor report

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

| Dataset | Survivors | Ground truth: same subject and finger | Ground truth: wrong | SIFT: matching | SIFT: non-matching |
|---|---:|---:|---:|---:|---:|
| SD300B | 500 | 500 | 0 | 284 | 216 |
| SD300C | 500 | 500 | 0 | 271 | 229 |

## 5. PLAIN versus corresponding ROLL

| Dataset | Total | Accepted | Rejected | Failures | Acceptance percentage | Mean comparison time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| SD300B | 500 | 284 | 216 | 0 | 56.800000000 | 22.771795200 |
| SD300C | 500 | 271 | 229 | 0 | 54.200000000 | 23.885775600 |

## 6. PLAIN versus next subject's ROLL

Pairing method: next surviving subject within the same canonical finger position, circular shift by one

| Dataset | Total | Incorrectly accepted | Correctly rejected | Failures | False-match percentage | Mean comparison time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| SD300B | 500 | 14 | 486 | 0 | 2.800000000 | 23.119792400 |
| SD300C | 500 | 15 | 485 | 0 | 3.000000000 | 23.785994600 |
