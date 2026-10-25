# What explains predictability (log1p targets)

Genes: 5001; encoders: conch_v15, uni2_h, virchow2; response for the main model: `r_virchow2` (highest median r).

## Spearman correlation with per-gene r (best encoder)

| feature | n | rho | p | partial rho (given mean expr.) |
|---|---|---|---|---|
| mean_log1p | 5001 | 0.883 | 0.0e+00 | n/a |
| var_log1p | 5001 | 0.825 | 0.0e+00 | -0.010 |
| frac_zero | 5001 | -0.875 | 0.0e+00 | -0.104 |
| morans_i | 5001 | 0.927 | 0.0e+00 | 0.597 |
| tau | 5001 | -0.382 | 1.0e-173 | -0.354 |
| agreement_mean | 5001 | 0.819 | 0.0e+00 | 0.765 |
| r_composition | 5001 | 0.983 | 0.0e+00 | 0.919 |

## Standardised OLS (HC3), R^2 = 0.939, n = 5001

| feature | coef (per SD) | SE | p | unique delta R^2 |
|---|---|---|---|---|
| mean_log1p | -0.033 | 0.002 | 5.5e-42 | 0.002 |
| frac_zero | -0.034 | 0.003 | 1.6e-22 | 0.002 |
| morans_i | 0.167 | 0.003 | 0.0e+00 | 0.050 |
| tau | 0.005 | 0.001 | 6.0e-07 | 0.000 |
| agreement_mean | 0.083 | 0.001 | 0.0e+00 | 0.070 |

## Median r by dominant cell type

| top_type               |   ('r_conch_v15', 'median') |   ('r_conch_v15', 'count') |   ('r_uni2_h', 'median') |   ('r_uni2_h', 'count') |   ('r_virchow2', 'median') |   ('r_virchow2', 'count') |   ('r_composition', 'median') |   ('r_composition', 'count') |
|:-----------------------|----------------------------:|---------------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|------------------------------:|-----------------------------:|
| b                      |                       0.383 |                        625 |                    0.404 |                     625 |                      0.43  |                       625 |                         0.415 |                          625 |
| endothelial            |                       0.461 |                        467 |                    0.478 |                     467 |                      0.534 |                       467 |                         0.534 |                          467 |
| epithelial             |                       0.651 |                       1338 |                    0.65  |                    1338 |                      0.7   |                      1338 |                         0.685 |                         1338 |
| fibroblast             |                       0.378 |                        436 |                    0.408 |                     436 |                      0.419 |                       436 |                         0.425 |                          436 |
| mast                   |                       0.206 |                        409 |                    0.221 |                     409 |                      0.235 |                       409 |                         0.216 |                          409 |
| myeloid                |                       0.485 |                        479 |                    0.532 |                     479 |                      0.567 |                       479 |                         0.56  |                          479 |
| plasma                 |                       0.628 |                        432 |                    0.633 |                     432 |                      0.677 |                       432 |                         0.674 |                          432 |
| smooth_muscle_pericyte |                       0.215 |                        458 |                    0.227 |                     458 |                      0.247 |                       458 |                         0.212 |                          458 |
| t_nk                   |                       0.391 |                        357 |                    0.425 |                     357 |                      0.463 |                       357 |                         0.423 |                          357 |
