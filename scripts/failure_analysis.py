"""List evaluation episodes where a saved greedy policy misses the 200-step limit.

For each such episode the policy is replayed without the time limit to see
whether it is stuck or just slow.

    python scripts/failure_analysis.py results/final/q_table_seed3.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qlearning as ql  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("q_table", help="path to a saved q_table .npy file")
    p.add_argument("--eval-seed-start", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=1000, help="step cap for the replay without a time limit")
    args = p.parse_args()

    q = np.load(args.q_table)
    n_bins = q.shape[0]
    env = gym.make(ql.ENV_ID).unwrapped  # no TimeLimit wrapper
    low = env.observation_space.low.astype(np.float64)
    width = ql.bin_widths(low, env.observation_space.high, n_bins)

    slow = 0
    for seed in range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes):
        obs, _ = env.reset(seed=seed)
        start = float(obs[0])
        steps, done = 0, False
        while not done and steps < args.max_steps:
            obs, _, done, _, _ = env.step(int(np.argmax(q[ql.discretize(obs, low, width, n_bins)])))
            steps += 1
        if steps > 200:
            slow += 1
            outcome = f"reaches the goal after {steps} steps" if done else f"no goal within {args.max_steps} steps"
            print(f"seed {seed}: start position {start:.3f}, {outcome}")
    print(f"{slow} of {args.eval_episodes} episodes exceed 200 steps")


if __name__ == "__main__":
    main()
