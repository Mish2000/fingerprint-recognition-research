# Derived SIFT plain-roll supervisor tables

## SD300b derived SIFT plain_roll

| Measure | Value |
| --- | ---: |
| Subjects | 888 |
| Anatomical identities | 8777 |
| Thumb | 1750 |
| Index | 1759 |
| Middle | 1756 |
| Ring | 1767 |
| Little | 1745 |
| plain_self accepted identities | 8786 |
| roll_self accepted identities | 8871 |
| Final paired identities | 8777 |
| plain_roll accepted | 4048 |
| plain_roll rejected | 4729 |
| Accepted percentage | 46.120542326535265 |
| Failures | 0 |
| Mean raw score | 5.586646918081349 |
| Median raw score | 3.0 |
| Mean method compare time (ms) | 18.771009513501195 |
| Median method compare time (ms) | 17.4484 |
| P95 method compare time (ms) | 28.209 |
| Keypoints A/B mean; median | 2754.8611142759487 / 2992.014355702404; 3000.0 / 3000.0 |
| Candidate matches mean; median | 12.877748661273785; 6.0 |
| Geometric inliers mean; median | 5.586646918081349; 3.0 |

## SD300c derived SIFT plain_roll

| Measure | Value |
| --- | ---: |
| Subjects | 888 |
| Anatomical identities | 8779 |
| Thumb | 1750 |
| Index | 1760 |
| Middle | 1756 |
| Ring | 1767 |
| Little | 1746 |
| plain_self accepted identities | 8788 |
| roll_self accepted identities | 8871 |
| Final paired identities | 8779 |
| plain_roll accepted | 3926 |
| plain_roll rejected | 4853 |
| Accepted percentage | 44.72035539355279 |
| Failures | 0 |
| Mean raw score | 5.458594372935414 |
| Median raw score | 3.0 |
| Mean method compare time (ms) | 22.140477036108894 |
| Median method compare time (ms) | 20.9113 |
| P95 method compare time (ms) | 31.2939 |
| Keypoints A/B mean; median | 2850.0369062535597 / 2994.111288301629; 3000.0 / 3000.0 |
| Candidate matches mean; median | 12.604282947943958; 6.0 |
| Geometric inliers mean; median | 5.458594372935414; 3.0 |

Every included PLAIN identity passed its dataset-specific frozen SIFT `plain_self` decision.
Every included ROLL identity passed its dataset-specific frozen SIFT `roll_self` decision.
Only identities with both sides and a valid base `plain_roll` pair were included.
A `plain_roll` rejection remains an experimental result and never removes the pair.
