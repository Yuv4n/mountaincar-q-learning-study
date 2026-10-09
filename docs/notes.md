# Notes on the original script and the rewrite

The 2021 version of `train.py` was reviewed against the current code before it was replaced. Each item below was checked directly.

- Exploration: the epsilon-greedy branch chose an action and the next line overwrote it with `argmax`, so the agent never explored. Confirmed in the source.
- Bin width: `high - low / bins` divides only `low` by the bin count. For MountainCar this gives widths of 0.66 and 0.0735 instead of 0.09 and 0.007. Sweeping the observation range with those widths produces position indices 0 to 2 and velocity indices 0 to 1, so only a few cells of the 20 x 20 table are reachable. I did not run the original code under legacy Gym, so I make no claim about how well it would have trained.
- Upper boundary: with the correct width an observation at the maximum maps to index n_bins, one past the end of the table. The rewrite clips the index.
- Terminal handling: the original ignored the difference between reaching the flag and hitting the 200-step limit, and only updated the table for the goal case in a way that depended on the reward scale. The rewrite uses Gymnasium's `terminated` and `truncated` flags.
- Compatibility: `np.int` is gone from current NumPy, the old `reset()` and four-value `step()` are replaced in Gymnasium, and `env.goal_position` is no longer reachable through the wrapper. `env.unwrapped.goal_position` is not needed now.

Design choices in the rewrite:

- The Q-table is still initialised uniformly in [-2, 0] as in the original. With a reward of -1 per step this makes unvisited actions look better than visited ones, which encourages trying them.
- Training seeds drive the Q-table initialisation, the exploration generator and the first environment reset. Evaluation uses different environment seeds, with a separate generator for the random baseline.
- Checkpoint evaluations for the learning curve use their own seeds (20000 to 20019) so they do not overlap the final evaluation seeds.

The earlier illustration of the update rule was left out because its redistribution rights are unknown.
