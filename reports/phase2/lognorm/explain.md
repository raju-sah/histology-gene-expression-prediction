# What explains predictability (lognorm targets)

Genes: 5001; encoders: conch_v15, uni2_h, virchow2; response for the main model: `r_virchow2` (highest median r).

## Spearman correlation with per-gene r (best encoder)

| feature | n | rho | p | partial rho (given mean expr.) |
|---|---|---|---|---|
| mean_log1p | 5001 | 0.455 | 4.8e-254 | n/a |
| var_log1p | 5001 | 0.439 | 1.8e-234 | 0.042 |
| frac_zero | 5001 | -0.440 | 1.9e-236 | 0.031 |
| morans_i | 5001 | 0.506 | 0.0e+00 | 0.272 |
| tau | 5001 | 0.212 | 5.3e-52 | 0.378 |
| agreement_mean | 5001 | 0.645 | 0.0e+00 | 0.676 |
| r_composition | 5001 | 0.688 | 0.0e+00 | 0.597 |

## Standardised OLS (HC3), R^2 = 0.649, n = 5001

| feature | coef (per SD) | SE | p | unique delta R^2 |
|---|---|---|---|---|
| mean_log1p | 0.019 | 0.003 | 2.7e-08 | 0.003 |
| frac_zero | 0.001 | 0.005 | 8.2e-01 | 0.000 |
| morans_i | 0.033 | 0.004 | 1.6e-15 | 0.007 |
| tau | 0.031 | 0.001 | 9.1e-194 | 0.060 |
| agreement_mean | 0.065 | 0.001 | 0.0e+00 | 0.264 |

## Median r by dominant cell type

| top_type               |   ('r_conch_v15', 'median') |   ('r_conch_v15', 'count') |   ('r_uni2_h', 'median') |   ('r_uni2_h', 'count') |   ('r_virchow2', 'median') |   ('r_virchow2', 'count') |   ('r_composition', 'median') |   ('r_composition', 'count') |
|:-----------------------|----------------------------:|---------------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|------------------------------:|-----------------------------:|
| b                      |                       0.083 |                        625 |                    0.105 |                     625 |                      0.117 |                       625 |                         0.093 |                          625 |
| endothelial            |                       0.142 |                        467 |                    0.153 |                     467 |                      0.188 |                       467 |                         0.109 |                          467 |
| epithelial             |                       0.143 |                       1338 |                    0.167 |                    1338 |                      0.192 |                      1338 |                         0.219 |                         1338 |
| fibroblast             |                       0.116 |                        436 |                    0.147 |                     436 |                      0.153 |                       436 |                         0.097 |                          436 |
| mast                   |                       0.061 |                        409 |                    0.068 |                     409 |                      0.087 |                       409 |                         0.032 |                          409 |
| myeloid                |                       0.101 |                        479 |                    0.145 |                     479 |                      0.163 |                       479 |                         0.146 |                          479 |
| plasma                 |                       0.081 |                        432 |                    0.106 |                     432 |                      0.116 |                       432 |                         0.126 |                          432 |
| smooth_muscle_pericyte |                       0.065 |                        458 |                    0.075 |                     458 |                      0.082 |                       458 |                         0.02  |                          458 |
| t_nk                   |                       0.101 |                        357 |                    0.133 |                     357 |                      0.128 |                       357 |                         0.095 |                          357 |
