"""Train one Q-learning agent on MountainCar-v0 and optionally evaluate it.

Example:
    python train.py --seed 0 --out runs/single
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np

import qlearning as ql


def build_parser() -> argparse.ArgumentParser:
    d = ql.Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="seed for Q-table init, exploration and the training env (default: 0)")
    p.add_argument("--episodes", type=int, default=d.episodes, help="training episodes (default: %(default)s)")
    p.add_argument("--n-bins", type=int, default=d.n_bins, help="bins per observation dimension (default: %(default)s)")
    p.add_argument("--learning-rate", type=float, default=d.learning_rate, help="Q-learning step size alpha (default: %(default)s)")
    p.add_argument("--discount", type=float, default=d.discount, help="discount factor gamma (default: %(default)s)")
    p.add_argument("--epsilon-start", type=float, default=d.epsilon_start, help="initial exploration rate (default: %(default)s)")
    p.add_argument("--epsilon-end", type=float, default=d.epsilon_end, help="final exploration rate (default: %(default)s)")
    p.add_argument("--epsilon-decay-fraction", type=float, default=d.epsilon_decay_fraction,
                   help="fraction of episodes over which epsilon decays linearly (default: %(default)s)")
    p.add_argument("--q-init-low", type=float, default=d.q_init_low, help="lower bound of the random Q init (default: %(default)s)")
    p.add_argument("--q-init-high", type=float, default=d.q_init_high, help="upper bound of the random Q init (default: %(default)s)")
    p.add_argument("--curve-eval-every", type=int, default=d.curve_eval_every,
                   help="greedy checkpoint evaluation interval in episodes, 0 to disable (default: %(default)s)")
    p.add_argument("--curve-eval-episodes", type=int, default=d.curve_eval_episodes,
                   help="episodes per greedy checkpoint evaluation (default: %(default)s)")
    p.add_argument("--curve-eval-seed-start", type=int, default=d.curve_eval_seed_start,
                   help="first environment seed of the checkpoint evaluations (default: %(default)s)")
    p.add_argument("--eval-episodes", type=int, default=100, help="greedy evaluation episodes after training (default: %(default)s)")
    p.add_argument("--eval-seed-start", type=int, default=50000,
                   help="first environment seed of the post-training evaluation. The default is a scratch range; "
                        "the reported results use 10000 to 10199, so avoid that range when exploring (default: %(default)s)")
    p.add_argument("--render-every", type=int, default=0,
                   help="open a render window every N training episodes, 0 for headless (default: 0)")
    p.add_argument("--out", default=None, help="directory for q_table.npy and result.json (default: do not save)")
    return p


def config_from_args(args: argparse.Namespace) -> ql.Config:
    return ql.Config(
        n_bins=args.n_bins, episodes=args.episodes, learning_rate=args.learning_rate,
        discount=args.discount, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay_fraction=args.epsilon_decay_fraction, q_init_low=args.q_init_low,
        q_init_high=args.q_init_high, curve_eval_every=args.curve_eval_every,
        curve_eval_episodes=args.curve_eval_episodes, curve_eval_seed_start=args.curve_eval_seed_start)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    def progress(ep, info):
        print(f"episode {ep:6d}  epsilon {info['epsilon']:.3f}  "
              f"train success (last 500) {info['train_success']:.3f}  greedy success {info['greedy_success']:.2f}")

    q, _ = ql.train(cfg, args.seed, render_every=args.render_every, progress=progress)
    eval_seeds = [args.eval_seed_start + i for i in range(args.eval_episodes)]
    greedy = ql.evaluate(q, cfg.n_bins, eval_seeds)
    random_policy = ql.evaluate_random(eval_seeds)
    print(f"greedy policy: success {greedy['success_rate']:.2f}, mean return {greedy['mean_return']:.1f}")
    print(f"random policy: success {random_policy['success_rate']:.2f}, mean return {random_policy['mean_return']:.1f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, "q_table.npy"), q)
        with open(os.path.join(args.out, "result.json"), "w") as f:
            json.dump({"seed": args.seed, "config": cfg.to_dict(), "eval_seeds": [eval_seeds[0], eval_seeds[-1]],
                       "greedy": {k: v for k, v in greedy.items() if k not in ("returns", "successes")},
                       "random": {k: v for k, v in random_policy.items() if k not in ("returns", "successes")}},
                      f, indent=2)


if __name__ == "__main__":
    main()
