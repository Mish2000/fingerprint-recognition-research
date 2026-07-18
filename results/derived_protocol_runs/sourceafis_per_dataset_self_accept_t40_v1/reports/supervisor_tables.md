# Derived SourceAFIS supervisor tables

## SD300b derived plain_roll

| Measure | Value |
|---|---:|
| Subjects | 887 |
| Anatomical identities | 8593 |
| Thumb | 1740 |
| Index | 1718 |
| Middle | 1736 |
| Ring | 1727 |
| Little | 1672 |
| All self accepted count | 8593 |
| Plain-roll accepted | 5784 |
| Plain-roll rejected | 2809 |
| Score = 0 | 589 |
| 0 < score < 40 | 2220 |
| Accept percentage | 67.310602 |
| Mean method compare time (ms) | 4.133063 |
| Median method compare time (ms) | 2.969800 |
| P95 method compare time (ms) | 10.922300 |

## SD300c derived plain_roll

| Measure | Value |
|---|---:|
| Subjects | 886 |
| Anatomical identities | 8614 |
| Thumb | 1735 |
| Index | 1727 |
| Middle | 1737 |
| Ring | 1732 |
| Little | 1683 |
| All self accepted count | 8614 |
| Plain-roll accepted | 5705 |
| Plain-roll rejected | 2909 |
| Score = 0 | 615 |
| 0 < score < 40 | 2294 |
| Accept percentage | 66.229394 |
| Mean method compare time (ms) | 3.971587 |
| Median method compare time (ms) | 2.922050 |
| P95 method compare time (ms) | 10.115400 |

Every PLAIN image in these tables passed its dataset-specific `plain_self` test at threshold 40.
Every ROLL image passed its dataset-specific `roll_self` test at threshold 40.
PLAIN-to-ROLL comparison was performed only after those two self decisions and availability intersection.
A rejection in `plain_roll` remains an experimental false non-match and never removes the pair.
