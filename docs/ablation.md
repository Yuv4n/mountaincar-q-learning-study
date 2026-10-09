# Effect of the original script's two defects

The 2021 script had two problems: the bin width was computed as `high - low / bins`, and the epsilon-greedy action was overwritten by a greedy one. The corrected code can switch each defect back on (`legacy_bin_width`, `ignore_exploration` in `qlearning.py`), so the effect was measured instead of assumed.

All runs use the original hyperparameters (learning rate 0.1, discount 0.95, 20 bins, 25,000 episodes), the same five training seeds (0 to 4) and the same 200 evaluation seeds (10000 to 10199) as the final experiment. Values are mean ± sample standard deviation over the five seeds. Configs are in `configs/` and outputs in `results/reference/`.

| variant | goal reached | mean return |
|---|---|---|
| corrected code | 100.0% ± 0.0% | -135.0 ± 7.2 |
| overwritten exploratory action restored | 99.9% ± 0.2% | -130.5 ± 9.8 |
| bin-width error restored | 0.0% ± 0.0% | -200.0 ± 0.0 |
| both defects restored | 0.0% ± 0.0% | -200.0 ± 0.0 |

The bin-width error alone was enough to stop learning in this setup: the greedy policies reached the goal in none of the 1,000 evaluation episodes. Restoring the overwritten action did not hurt. The Q-table starts uniformly in [-2, 0] and every step costs -1, so untried actions keep looking better than tried ones and a purely greedy agent still ends up visiting most of the table. The two corrected-code figures differ by less than the seed-to-seed spread, so there is no evidence here that epsilon-greedy helps on this task.

The "both defects" variant is the original behaviour apart from the Gymnasium interface and the corrected terminal and truncation handling. I did not run the unmodified 2021 script under legacy Gym.

The corrected code with the original hyperparameters reaches the goal as reliably as the final configuration but about 9 steps slower on average (-135.0 against -126.2). Those two runs share seeds, but the comparison is five seeds on each side and the final configuration was chosen on separate validation seeds.
