# SourceAFIS joint 500 pilot supervisor report

## 1. Record counts

| Dataset | PLAIN | ROLL |
|---|---:|---:|
| SD300B | 500 | 500 |
| SD300C | 500 | 500 |

## 2. PLAIN self-comparisons

| Dataset | Matched | Not matched | Removed | Remaining |
|---|---:|---:|---:|---:|
| SD300B | 500 | 0 | 0 | 500 |
| SD300C | 500 | 0 | 0 | 500 |

## 3. ROLL self-comparisons

| Dataset | Matched | Not matched | Removed | Remaining |
|---|---:|---:|---:|---:|
| SD300B | 500 | 0 | 0 | 500 |
| SD300C | 500 | 0 | 0 | 500 |

## 4. PLAIN-to-corresponding-ROLL record matching

The base selection came from a population that had already passed self-cleaning. The four pilot self runs revalidated that cleaning. Identities before repeat cleaning: 500; identities after repeat cleaning: 500.

| Dataset | Pairs after self | Ground truth: same subject and finger | Ground truth: different subject | SourceAFIS: same person | SourceAFIS: not same person |
|---|---:|---:|---:|---:|---:|
| SD300B | 500 | 500 | 0 | 358 | 142 |
| SD300C | 500 | 500 | 0 | 344 | 156 |

## 5. PLAIN versus corresponding ROLL

| Dataset | Matched | Not matched | Match percentage | Mean comparison time (ms) |
|---|---:|---:|---:|---:|
| SD300B | 358 | 142 | 71.600000000 | 4.821668800 |
| SD300C | 344 | 156 | 68.800000000 | 4.398587600 |

## 6. PLAIN versus next subject's ROLL

Pairing method: `next subject within the same canonical finger position, circular shift by one`

| Dataset | Pairs | Incorrectly matched | Correctly rejected | False-match percentage | Mean comparison time (ms) |
|---|---:|---:|---:|---:|---:|
| SD300B | 500 | 0 | 500 | 0.000000000 | 4.718205200 |
| SD300C | 500 | 0 | 500 | 0.000000000 | 3.783832200 |
