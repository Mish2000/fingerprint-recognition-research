# SIFT Geometric Supervisor Tables

Method: `sift_geometric`
Method version: `sift-geometric-v1`
Cohort: `sift_geometric_joint_self_accept_v1`
Included anatomical identities: 8777
Decision rule hash: `f3f03d964b20229cea8dfc5363f6b9558cae9fb60b4e4c0d5df868dcd1fc9914`

## Full manifests

| dataset | protocol | pairs | success | failure | accepted | accepted % | median score | zero % | geometry failure % | median total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sd300b | plain_roll | 8779 | 8778 | 1 | 4048 | 46.11 | 3.000 | 25.43 | 25.43 | 402.926 |
| sd300b | plain_self | 8788 | 8787 | 1 | 8786 | 99.98 | 3000.000 | 0.01 | 0.01 | 222.098 |
| sd300b | roll_self | 8871 | 8871 | 0 | 8871 | 100.00 | 3000.000 | 0.00 | 0.00 | 618.937 |
| sd300c | plain_roll | 8779 | 8779 | 0 | 3926 | 44.72 | 3.000 | 24.71 | 24.71 | 1381.933 |
| sd300c | plain_self | 8788 | 8788 | 0 | 8788 | 100.00 | 3000.000 | 0.00 | 0.00 | 671.734 |
| sd300c | roll_self | 8871 | 8871 | 0 | 8871 | 100.00 | 3000.000 | 0.00 | 0.00 | 2138.500 |

## Frozen evaluation subjects

These rows exclude every development subject used by pilot selection and threshold calibration.

| dataset | protocol | pairs | success | failure | accepted | accepted % | median score | zero % | geometry failure % | median total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sd300b | plain_roll | 6882 | 6881 | 1 | 3155 | 45.84 | 3.000 | 25.74 | 25.74 | 402.623 |
| sd300b | plain_self | 6888 | 6887 | 1 | 6887 | 99.99 | 3000.000 | 0.00 | 0.00 | 221.933 |
| sd300b | roll_self | 6954 | 6954 | 0 | 6954 | 100.00 | 3000.000 | 0.00 | 0.00 | 619.355 |
| sd300c | plain_roll | 6882 | 6882 | 0 | 3053 | 44.36 | 3.000 | 24.92 | 24.92 | 1383.561 |
| sd300c | plain_self | 6888 | 6888 | 0 | 6888 | 100.00 | 3000.000 | 0.00 | 0.00 | 670.566 |
| sd300c | roll_self | 6954 | 6954 | 0 | 6954 | 100.00 | 3000.000 | 0.00 | 0.00 | 2140.624 |

## SIFT-specific cohort

Membership uses only the four frozen self decisions. Plain-roll outcomes are retained and do not affect membership.

| dataset | protocol | pairs | success | failure | accepted | accepted % | median score | zero % | geometry failure % | median total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sd300b | plain_roll | 8777 | 8777 | 0 | 4048 | 46.12 | 3.000 | 25.42 | 25.42 | 402.927 |
| sd300b | plain_self | 8777 | 8777 | 0 | 8777 | 100.00 | 3000.000 | 0.00 | 0.00 | 222.146 |
| sd300b | roll_self | 8777 | 8777 | 0 | 8777 | 100.00 | 3000.000 | 0.00 | 0.00 | 619.063 |
| sd300c | plain_roll | 8777 | 8777 | 0 | 3926 | 44.73 | 3.000 | 24.69 | 24.69 | 1381.948 |
| sd300c | plain_self | 8777 | 8777 | 0 | 8777 | 100.00 | 3000.000 | 0.00 | 0.00 | 671.904 |
| sd300c | roll_self | 8777 | 8777 | 0 | 8777 | 100.00 | 3000.000 | 0.00 | 0.00 | 2138.770 |
