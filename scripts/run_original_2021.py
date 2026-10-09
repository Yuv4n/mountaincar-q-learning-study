"""Run the unmodified 2021 script (legacy/original_train.py) under legacy Gym and evaluate what it learned.

The script itself is executed as written. This wrapper only seeds NumPy and the
environment before the run, counts the goal messages it prints, and afterwards
runs the script's own state conversion with a greedy policy on held-out seeds.
It needs the legacy environment in requirements-legacy.txt (Python 3.9):

    python scripts/run_original_2021.py --seed 0 --out results/legacy/seed0.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import runpy

import gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(os.path.dirname(HERE), "legacy", "original_train.py")


class _Tee(io.TextIOBase):
    def __init__(self):
        self.goal_episodes = []

    def write(self, text):
        for line in text.splitlines():
            if line.startswith("We made it on episode"):
                self.goal_episodes.append(int(line.rsplit(" ", 1)[1]))
        return len(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed-start", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    np.random.seed(args.seed)
    real_make = gym.make

    def seeded_make(*a, **k):
        env = real_make(*a, **k)
        env.seed(args.seed)
        return env

    gym.make = seeded_make
    tee = _Tee()
    with contextlib.redirect_stdout(tee):
        g = runpy.run_path(ORIGINAL, run_name="original_2021")
    gym.make = real_make

    q_table, to_state = g["q_table"], g["get_discrete_state"]
    env = real_make("MountainCar-v0")
    returns, successes = [], []
    for i in range(args.eval_episodes):
        env.seed(args.eval_seed_start + i)
        obs = env.reset()
        total, done = 0.0, False
        while not done:
            obs, reward, done, _ = env.step(int(np.argmax(q_table[to_state(obs)])))
            total += reward
        returns.append(total)
        successes.append(bool(obs[0] >= env.unwrapped.goal_position))
    result = {
        "seed": args.seed,
        "gym": gym.__version__,
        "numpy": np.__version__,
        "training_episodes_reaching_goal": len(tee.goal_episodes),
        "first_goal_episode": tee.goal_episodes[0] if tee.goal_episodes else None,
        "greedy_eval": {"seeds": [args.eval_seed_start, args.eval_seed_start + args.eval_episodes - 1],
                        "episodes": args.eval_episodes, "success_rate": float(np.mean(successes)),
                        "mean_return": float(np.mean(returns))},
    }
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
