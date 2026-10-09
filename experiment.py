"""Multi-seed experiment: train, evaluate the frozen greedy policies, compare to a random baseline.

Everything needed to reproduce a run lives in the JSON config, for example:
    python experiment.py --config configs/final.json --out results/final
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

import qlearning as ql


def load_config(path: str) -> Dict:
    with open(path) as f:
        spec = json.load(f)
    spec["hyperparameters"] = ql.Config(**spec["hyperparameters"]).to_dict()  # validates keys
    return spec


def seed_list(spec: Dict, key: str) -> List[int]:
    start, count = spec[key]["start"], spec[key]["count"]
    return list(range(start, start + count))


def _run_seed(args):
    spec, seed = args
    cfg = ql.Config(**spec["hyperparameters"])
    t0 = time.time()
    q, hist = ql.train(cfg, seed)
    eval_seeds = seed_list(spec, "eval_seeds")
    greedy = ql.evaluate(q, cfg.n_bins, eval_seeds)
    return seed, q, hist, greedy, time.time() - t0


def summarise(per_seed: List[Dict], key: str) -> Dict:
    vals = np.array([r[key] for r in per_seed])
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_curves(histories: List[Dict], cfg: ql.Config, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = 500
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    train = np.array([moving_average(h["train_success"].astype(float), window) for h in histories])
    x = np.arange(window, cfg.episodes + 1)
    ax = axes[0]
    for row in train:
        ax.plot(x, row, color="tab:blue", alpha=0.25, linewidth=0.8)
    ax.plot(x, train.mean(axis=0), color="tab:blue", linewidth=2, label="mean over seeds")
    ax.set_title("Training episodes (epsilon-greedy, exploring)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Fraction of episodes reaching the goal")
    ax.legend(loc="upper left")

    ck = np.array([h["ckpt_success"] for h in histories])
    cx = histories[0]["ckpt_episode"]
    ax = axes[1]
    for row in ck:
        ax.plot(cx, row, color="tab:green", alpha=0.25, linewidth=0.8)
    ax.plot(cx, ck.mean(axis=0), color="tab:green", linewidth=2, label="mean over seeds")
    ax.set_title("Greedy policy checkpoints (no exploration, no updates)")
    ax.set_xlabel("Training episode at checkpoint")
    ax.legend(loc="upper left")
    for a in axes:
        a.set_ylim(-0.02, 1.02)
        a.grid(alpha=0.3)

    fig.suptitle(f"MountainCar-v0 tabular Q-learning, {len(histories)} seeds "
                 f"(left: {window}-episode moving average; right: {cfg.curve_eval_episodes} episodes per checkpoint)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment JSON (hyperparameters, train_seeds, eval_seeds)")
    p.add_argument("--out", required=True, help="output directory for results.json, q_tables and learning_curve.png")
    p.add_argument("--workers", type=int, default=0, help="parallel processes, 0 for one per training seed (default: 0)")
    p.add_argument("--no-plot", action="store_true", help="skip the learning-curve PNG (used for tuning)")
    args = p.parse_args(argv)

    spec = load_config(args.config)
    cfg = ql.Config(**spec["hyperparameters"])
    train_seeds = seed_list(spec, "train_seeds")
    eval_seeds = seed_list(spec, "eval_seeds")
    assert not set(train_seeds) & set(eval_seeds), "train and evaluation seeds must be disjoint"
    os.makedirs(args.out, exist_ok=True)

    workers = args.workers or len(train_seeds)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(_run_seed, [(spec, s) for s in train_seeds]))

    random_eval = ql.evaluate_random(eval_seeds, policy_seed=spec.get("random_policy_seed", 0))
    heuristic_eval = ql.evaluate_heuristic(eval_seeds)
    per_seed, histories = [], []
    for seed, q, hist, greedy, elapsed in outputs:
        np.save(os.path.join(args.out, f"q_table_seed{seed}.npy"), q)
        per_seed.append({"train_seed": seed, "success_rate": greedy["success_rate"],
                         "mean_return": greedy["mean_return"], "mean_steps": greedy["mean_steps"],
                         "train_seconds": round(elapsed, 1)})
        histories.append(hist)
        print(f"seed {seed}: success {greedy['success_rate']:.2f}  mean return {greedy['mean_return']:.1f}  ({elapsed:.0f}s)")

    results = {
        "description": spec.get("description", ""),
        "config": spec,
        "evaluation_seeds": {"start": eval_seeds[0], "end": eval_seeds[-1], "episodes": len(eval_seeds)},
        "greedy_per_seed": per_seed,
        "greedy_across_seeds": {"success_rate": summarise(per_seed, "success_rate"),
                                "mean_return": summarise(per_seed, "mean_return")},
        "random_baseline": {k: v for k, v in random_eval.items() if k not in ("returns", "successes")},
        "velocity_heuristic_baseline": {k: v for k, v in heuristic_eval.items() if k not in ("returns", "successes")},
        "velocity_heuristic_description": "push right when velocity >= 0, left otherwise; no learning",
        "std_note": "sample standard deviation (ddof=1) across training seeds",
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out, "history.npz"), **{
        f"seed{s}_{k}": v for (s, _, h, _, _), _ in zip(outputs, histories) for k, v in h.items()})
    if not args.no_plot:
        plot_curves(histories, cfg, os.path.join(args.out, "learning_curve.png"))
    g = results["greedy_across_seeds"]
    print(f"greedy across seeds: success {g['success_rate']['mean']:.3f} +/- {g['success_rate']['std']:.3f}, "
          f"return {g['mean_return']['mean']:.1f} +/- {g['mean_return']['std']:.1f}")
    print(f"random baseline: success {random_eval['success_rate']:.3f}, return {random_eval['mean_return']:.1f}")
    print(f"velocity heuristic: success {heuristic_eval['success_rate']:.3f}, return {heuristic_eval['mean_return']:.1f}")


if __name__ == "__main__":
    main()
