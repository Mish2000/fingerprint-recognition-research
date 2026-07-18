# SourceAFIS exact requested supervisor report

## 1. Single-finger record counts

| Dataset | plain_single_finger_records | roll_single_finger_records |
|---|---:|---:|
| SD300B | 8788 | 8871 |
| SD300C | 8788 | 8871 |

## 2. PLAIN self-comparisons

| Dataset | total | matches | non_matches_removed_from_derived_protocol | match_percentage | mean_comparison_time_ms |
|---|---:|---:|---:|---:|---:|
| SD300B | 8788 | 8613 | 175 | 98.008648157 | 6.394842023 |
| SD300C | 8788 | 8632 | 156 | 98.224852071 | 6.052149989 |

## 3. ROLL self-comparisons

| Dataset | total | matches | non_matches_removed_from_derived_protocol | match_percentage | mean_comparison_time_ms |
|---|---:|---:|---:|---:|---:|
| SD300B | 8871 | 8843 | 28 | 99.684364784 | 16.341807174 |
| SD300C | 8871 | 8847 | 24 | 99.729455529 | 14.728983710 |

## 4. Cleaned genuine PLAIN-ROLL pairs

| Dataset | pair_count | ground_truth_same_subject_same_finger_count | ground_truth_wrong_count |
|---|---:|---:|---:|
| SD300B | 8593 | 8593 | 0 |
| SD300C | 8614 | 8614 | 0 |

## 5. SourceAFIS results on genuine pairs

| Dataset | matched_by_sourceafis | not_matched_by_sourceafis | match_percentage | mean_comparison_time_ms |
|---|---:|---:|---:|---:|
| SD300B | 5784 | 2809 | 67.310601653 | 4.133063342 |
| SD300C | 5705 | 2909 | 66.229394010 | 3.971587091 |

## 6. SourceAFIS results on wrong pairs

Pairing method: `next subject within the same canonical finger position, circular shift by one`

| Dataset | pair_count | incorrectly_matched_by_sourceafis | correctly_rejected_by_sourceafis | false_match_percentage | mean_comparison_time_ms |
|---|---:|---:|---:|---:|---:|
| SD300B | 8593 | 9 | 8584 | 0.104736413 | 3.146589736 |
| SD300C | 8614 | 10 | 8604 | 0.116090086 | 3.116608533 |
