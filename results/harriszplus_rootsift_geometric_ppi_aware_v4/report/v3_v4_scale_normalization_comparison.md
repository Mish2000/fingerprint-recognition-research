# HarrisZ+ v3 מול v4 — scale normalization

דוח זה הוא ablation טכני נפרד ואינו דוח המנחה.

- Physical-scale contract: `PASS`
- v4 משנה רק את פרשנות הפרמטרים המרחביים לפי manifest PPI.
- אין סף ביצועים בדוח זה ולא בוצע tuning מתוצאות 500.
- allocated ו-reserved מוצגים בנפרד ואינם מסוכמים.

## תוצאות שמונת התנאים

| תנאי | גרסה | ok/total | accepted ≥4 | score median | candidates median | saturation | mutual median | inliers median | total ms median | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sd300b/plain_self | v3 | 500/500 | 500 | 3000.000 | 10317.500 | 1.000 | 3000.000 | 3000.000 | 1210.614 | 1135834624.000 | 4057989120.000 |
| sd300b/plain_self | v4 | 500/500 | 500 | 3000.000 | 10132.000 | 1.000 | 3000.000 | 3000.000 | 1469.835 | 1139921920.000 | 1258291200.000 |
| sd300b/roll_self | v3 | 500/500 | 500 | 3000.000 | 36783.000 | 1.000 | 3000.000 | 3000.000 | 2279.920 | 1351982080.000 | 6968836096.000 |
| sd300b/roll_self | v4 | 500/500 | 500 | 3000.000 | 36079.500 | 1.000 | 3000.000 | 3000.000 | 2892.150 | 1352137728.000 | 1547698176.000 |
| sd300b/plain_roll_genuine | v3 | 500/500 | 64 | 2.000 | 26996.000 | 1.000 | 5.000 | 2.000 | 1744.269 | 1351982080.000 | 6968836096.000 |
| sd300b/plain_roll_genuine | v4 | 500/500 | 63 | 2.000 | 26029.500 | 1.000 | 5.000 | 2.000 | 4084.437 | 1352137728.000 | 1547698176.000 |
| sd300b/plain_roll_negative | v3 | 500/500 | 3 | 2.000 | 26996.000 | 1.000 | 3.000 | 2.000 | 1729.824 | 1351982080.000 | 6968836096.000 |
| sd300b/plain_roll_negative | v4 | 500/500 | 1 | 2.000 | 26029.500 | 1.000 | 3.000 | 2.000 | 1832.756 | 1352137728.000 | 1547698176.000 |
| sd300c/plain_self | v3 | 500/500 | 500 | 3000.000 | 31456.000 | 1.000 | 3000.000 | 3000.000 | 2126.326 | 4507041280.000 | 27055357952.000 |
| sd300c/plain_self | v4 | 500/500 | 500 | 3000.000 | 12073.000 | 1.000 | 3000.000 | 3000.000 | 1766.619 | 4507042816.000 | 4991221760.000 |
| sd300c/roll_self | v3 | 500/500 | 500 | 3000.000 | 131551.000 | 1.000 | 3000.000 | 3000.000 | 6408.025 | 5417253376.000 | 34472984576.000 |
| sd300c/roll_self | v4 | 500/500 | 500 | 3000.000 | 45172.500 | 1.000 | 3000.000 | 3000.000 | 3703.923 | 5429470208.000 | 5966397440.000 |
| sd300c/plain_roll_genuine | v3 | 500/500 | 1 | 2.000 | 97242.000 | 1.000 | 4.000 | 2.000 | 4227.497 | 5417099264.000 | 6276775936.000 |
| sd300c/plain_roll_genuine | v4 | 500/500 | 40 | 0.000 | 32842.000 | 1.000 | 1.000 | 0.000 | 2632.261 | 5429470208.000 | 5966397440.000 |
| sd300c/plain_roll_negative | v3 | 500/500 | 0 | 2.000 | 97242.000 | 1.000 | 3.000 | 2.000 | 4213.245 | 5417099264.000 | 6276775936.000 |
| sd300c/plain_roll_negative | v4 | 500/500 | 1 | 0.000 | 32842.000 | 1.000 | 0.000 | 0.000 | 2620.899 | 5429470208.000 | 5966397440.000 |

## Physical support

| scale | parameter | B mm | C mm | delta mm | pass |
|---:|---|---:|---:|---:|---:|
| 0 | native_differentiation_sigma_px | 0.025400 | 0.025400 | 0.000000 | True |
| 0 | native_integration_sigma_px | 0.035921 | 0.035921 | 0.000000 | True |
| 0 | gaussian_support_diameter_native_px | 0.152400 | 0.152400 | 0.000000 | True |
| 0 | suppression_distance_native_px | 0.063500 | 0.063500 | 0.000000 | True |
| 0 | duplicate_radius_native_px | 0.025400 | 0.025400 | 0.000000 | True |
| 0 | opencv_keypoint_size_px | 0.071842 | 0.071842 | 0.000000 | True |
| 0 | orientation_radius_px | 0.152400 | 0.152400 | 0.000000 | True |
| 0 | descriptor_support_diameter_estimate_px | 0.762000 | 0.762000 | 0.000000 | True |
| 0 | border_margin_native_px | 0.406400 | 0.406400 | 0.000000 | True |
| 1 | native_differentiation_sigma_px | 0.025400 | 0.025400 | 0.000000 | True |
| 1 | native_integration_sigma_px | 0.035921 | 0.035921 | 0.000000 | True |
| 1 | gaussian_support_diameter_native_px | 0.228600 | 0.228600 | 0.000000 | True |
| 1 | suppression_distance_native_px | 0.076200 | 0.076200 | 0.000000 | True |
| 1 | duplicate_radius_native_px | 0.025400 | 0.025400 | 0.000000 | True |
| 1 | opencv_keypoint_size_px | 0.071842 | 0.071842 | 0.000000 | True |
| 1 | orientation_radius_px | 0.152400 | 0.152400 | 0.000000 | True |
| 1 | descriptor_support_diameter_estimate_px | 0.762000 | 0.762000 | 0.000000 | True |
| 1 | border_margin_native_px | 0.406400 | 0.406400 | 0.000000 | True |
| 2 | native_differentiation_sigma_px | 0.035921 | 0.035921 | 0.000000 | True |
| 2 | native_integration_sigma_px | 0.050800 | 0.050800 | 0.000000 | True |
| 2 | gaussian_support_diameter_native_px | 0.304800 | 0.304800 | 0.000000 | True |
| 2 | suppression_distance_native_px | 0.127000 | 0.127000 | 0.000000 | True |
| 2 | duplicate_radius_native_px | 0.025400 | 0.025400 | 0.000000 | True |
| 2 | opencv_keypoint_size_px | 0.101600 | 0.101600 | 0.000000 | True |
| 2 | orientation_radius_px | 0.228600 | 0.228600 | 0.000000 | True |
| 2 | descriptor_support_diameter_estimate_px | 1.077631 | 1.077631 | 0.000000 | True |
| 2 | border_margin_native_px | 0.558800 | 0.558800 | 0.000000 | True |
| 3 | native_differentiation_sigma_px | 0.050800 | 0.050800 | 0.000000 | True |
| 3 | native_integration_sigma_px | 0.071842 | 0.071842 | 0.000000 | True |
| 3 | gaussian_support_diameter_native_px | 0.457200 | 0.457200 | 0.000000 | True |
| 3 | suppression_distance_native_px | 0.152400 | 0.152400 | 0.000000 | True |
| 3 | duplicate_radius_native_px | 0.025400 | 0.025400 | 0.000000 | True |
| 3 | opencv_keypoint_size_px | 0.143684 | 0.143684 | 0.000000 | True |
| 3 | orientation_radius_px | 0.330200 | 0.330200 | 0.000000 | True |
| 3 | descriptor_support_diameter_estimate_px | 1.524000 | 1.524000 | 0.000000 | True |
| 3 | border_margin_native_px | 0.787400 | 0.787400 | 0.000000 | True |
| 4 | native_differentiation_sigma_px | 0.071842 | 0.071842 | 0.000000 | True |
| 4 | native_integration_sigma_px | 0.101600 | 0.101600 | 0.000000 | True |
| 4 | gaussian_support_diameter_native_px | 0.609600 | 0.609600 | 0.000000 | True |
| 4 | suppression_distance_native_px | 0.228600 | 0.228600 | 0.000000 | True |
| 4 | duplicate_radius_native_px | 0.025400 | 0.025400 | 0.000000 | True |
| 4 | opencv_keypoint_size_px | 0.203200 | 0.203200 | 0.000000 | True |
| 4 | orientation_radius_px | 0.457200 | 0.457200 | 0.000000 | True |
| 4 | descriptor_support_diameter_estimate_px | 2.155261 | 2.155261 | 0.000000 | True |
| 4 | border_margin_native_px | 1.092200 | 1.092200 | 0.000000 | True |

יש לקרוא genuine ו-negative יחד; genuine acceptance לבדו אינו בסיס לטענה ש-v4 טוב יותר.
