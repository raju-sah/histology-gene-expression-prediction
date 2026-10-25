| task | encoder | head | Pearson (ours) | 95% CI (samples) | published | diff |
|---|---|---|---|---|---|---|
| IDC | resnet50 | hest_ridge | 0.4739 | [0.432, 0.516] | 0.4739 | -0.0000 |
| IDC | resnet50 | ridge_cv | 0.4769 | [0.435, 0.518] | 0.4739 | +0.0030 |
| IDC | resnet50 | mlp | 0.4800 | [0.441, 0.519] | 0.4739 | +0.0061 |
| IDC | uni2_h | hest_ridge | 0.5899 | [0.514, 0.663] | 0.5898 | +0.0001 |
| IDC | uni2_h | ridge_cv | 0.5975 | [0.524, 0.671] | 0.5898 | +0.0077 |
| IDC | uni2_h | mlp | 0.5911 | [0.521, 0.661] | 0.5898 | +0.0013 |
| LUNG | resnet50 | hest_ridge | 0.4953 | [0.486, 0.505] | 0.4949 | +0.0004 |
| LUNG | resnet50 | ridge_cv | 0.5380 | [0.515, 0.561] | 0.4949 | +0.0431 |
| LUNG | resnet50 | mlp | 0.5037 | [0.501, 0.506] | 0.4949 | +0.0088 |
| LUNG | uni2_h | hest_ridge | 0.5593 | [0.547, 0.572] | 0.5587 | +0.0006 |
| LUNG | uni2_h | ridge_cv | 0.5922 | [0.580, 0.605] | 0.5587 | +0.0335 |
| LUNG | uni2_h | mlp | 0.5590 | [0.552, 0.566] | 0.5587 | +0.0003 |
| COAD | resnet50 | hest_ridge | 0.2503 | [0.224, 0.282] | 0.2500 | +0.0003 |
| COAD | resnet50 | ridge_cv | 0.3086 | [0.308, 0.365] | 0.2500 | +0.0586 |
| COAD | resnet50 | mlp | 0.2668 | [0.250, 0.289] | 0.2500 | +0.0168 |
| COAD | uni2_h | hest_ridge | 0.3026 | [0.264, 0.342] | 0.3015 | +0.0011 |
| COAD | uni2_h | ridge_cv | 0.3437 | [0.318, 0.388] | 0.3015 | +0.0422 |
| COAD | uni2_h | mlp | 0.2977 | [0.252, 0.333] | 0.3015 | -0.0038 |

**Flags**

- COAD/resnet50/ridge_cv: 0.3086 exceeds published 0.2500 by > 0.05: check leakage
