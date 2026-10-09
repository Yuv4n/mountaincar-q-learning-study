# Hyperparameter selection

Eight settings were compared before the final experiment: learning rate {0.1, 0.2}, discount {0.95, 0.99} and bins per dimension {20, 40}. Everything else was left at the values from the original script (25,000 episodes, epsilon falling linearly from 1 to 0 over the first half, Q-table initialised uniformly in [-2, 0]).

Each setting was trained with seeds 100 to 102 and evaluated greedily on environment seeds 5000 to 5099. These seeds are not used in the final experiment (training seeds 0 to 4, evaluation seeds 10000 to 10199). The selection rule was fixed in advance: highest mean validation return. Values are mean ± sample standard deviation over the three seeds. Per-setting outputs are in `results/tuning/` and the configs in `configs/tuning/`.

| learning rate | discount | bins | success rate | mean return |
|---|---|---|---|---|
| 0.2 | 0.99 | 40 | 1.000 ± 0.000 | -124.7 ± 0.5 |
| 0.2 | 0.95 | 40 | 1.000 ± 0.000 | -130.1 ± 3.0 |
| 0.1 | 0.99 | 40 | 0.970 ± 0.052 | -133.0 ± 4.3 |
| 0.1 | 0.99 | 20 | 1.000 ± 0.000 | -133.5 ± 4.9 |
| 0.1 | 0.95 | 40 | 0.950 ± 0.078 | -141.6 ± 8.8 |
| 0.1 | 0.95 | 20 | 1.000 ± 0.000 | -142.3 ± 3.2 |
| 0.2 | 0.95 | 20 | 1.000 ± 0.000 | -143.7 ± 13.0 |
| 0.2 | 0.99 | 20 | 1.000 ± 0.000 | -144.3 ± 2.0 |

The differences between settings are small compared with the seed-to-seed spread for some rows, and three seeds is a small sample, so this is a light tuning pass and not a claim that the chosen setting is clearly best. The original values (0.1, 0.95, 20 bins) already reach the goal on every validation episode. The setting with learning rate 0.2, discount 0.99 and 40 bins had the highest mean return and is used in `configs/final.json`.
