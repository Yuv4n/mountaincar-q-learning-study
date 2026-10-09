"""Tabular Q-learning for MountainCar-v0 (Gymnasium).

The two continuous observations (position, velocity) are binned into an
n_bins x n_bins grid and a Q-value is kept for each of the three actions.
Importing this module has no side effects; use train.py or experiment.py to run it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

ENV_ID = "MountainCar-v0"


@dataclass
class Config:
    n_bins: int = 40
    episodes: int = 25000
    learning_rate: float = 0.2
    discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.0
    # Epsilon falls linearly to epsilon_end over this fraction of the episodes.
    epsilon_decay_fraction: float = 0.5
    q_init_low: float = -2.0
    q_init_high: float = 0.0
    # Greedy-policy checkpoints used for the learning curve.
    curve_eval_every: int = 500
    curve_eval_episodes: int = 20
    curve_eval_seed_start: int = 20000

    def to_dict(self) -> Dict:
        return asdict(self)


def bin_widths(low: np.ndarray, high: np.ndarray, n_bins: int) -> np.ndarray:
    """Width of one bin in each observation dimension."""
    return (np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)) / n_bins


def discretize(obs, low, width, n_bins: int) -> Tuple[int, ...]:
    """Map a continuous observation to a table index, clipped to [0, n_bins - 1]."""
    idx = np.floor((np.asarray(obs, dtype=np.float64) - low) / width).astype(np.intp)
    return tuple(int(i) for i in np.clip(idx, 0, n_bins - 1))


def epsilon_at(episode: int, cfg: Config) -> float:
    """Linear epsilon schedule: epsilon_start -> epsilon_end over the decay window."""
    decay_episodes = max(1, int(cfg.episodes * cfg.epsilon_decay_fraction))
    frac = min(1.0, episode / decay_episodes)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy: a uniformly random action with probability epsilon, else argmax."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


def q_update(q, state, action, reward, next_state, terminated, learning_rate, discount) -> float:
    """One Q-learning update, in place. Returns the new Q-value.

    A terminal transition has no future, so the target is the reward alone. A
    time-limit truncation is not a real end of the task, so it still bootstraps
    from the next state.
    """
    target = reward if terminated else reward + discount * float(np.max(q[next_state]))
    q[state + (action,)] = (1 - learning_rate) * q[state + (action,)] + learning_rate * target
    return float(q[state + (action,)])


def init_q_table(cfg: Config, n_actions: int, rng: np.random.Generator) -> np.ndarray:
    shape = (cfg.n_bins, cfg.n_bins, n_actions)
    return rng.uniform(cfg.q_init_low, cfg.q_init_high, size=shape)


def _grid(env, n_bins):
    low = env.observation_space.low.astype(np.float64)
    high = env.observation_space.high.astype(np.float64)
    return low, bin_widths(low, high, n_bins)


def evaluate(q, n_bins: int, seeds: List[int]) -> Dict:
    """Run the frozen greedy policy once per seed. The Q-table is never modified."""
    env = gym.make(ENV_ID)
    low, width = _grid(env, n_bins)
    returns, steps, successes = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        total, n, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = env.step(
                int(np.argmax(q[discretize(obs, low, width, n_bins)])))
            total += reward
            n += 1
        returns.append(total)
        steps.append(n)
        successes.append(bool(terminated))
    env.close()
    return _summary(returns, steps, successes)


def evaluate_random(seeds: List[int], policy_seed: int = 0) -> Dict:
    """Uniformly random policy on the same environment seeds as evaluate()."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(policy_seed)
    returns, steps, successes = [], [], []
    for seed in seeds:
        env.reset(seed=int(seed))
        total, n, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            total += reward
            n += 1
        returns.append(total)
        steps.append(n)
        successes.append(bool(terminated))
    env.close()
    return _summary(returns, steps, successes)


def evaluate_heuristic(seeds: List[int]) -> Dict:
    """Hand-coded baseline: push in the direction of the current velocity (right when at rest)."""
    env = gym.make(ENV_ID)
    returns, steps, successes = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        total, n, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = env.step(2 if obs[1] >= 0 else 0)
            total += reward
            n += 1
        returns.append(total)
        steps.append(n)
        successes.append(bool(terminated))
    env.close()
    return _summary(returns, steps, successes)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _summary(returns, steps, successes) -> Dict:
    return {
        "episodes": len(returns),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(steps)),
        "returns": [float(r) for r in returns],
        "successes": [bool(s) for s in successes],
    }


def train(cfg: Config, seed: int, render_every: int = 0,
          progress: Optional[Callable[[int, Dict], None]] = None) -> Tuple[np.ndarray, Dict]:
    """Train from scratch with one seed. Returns (q_table, history).

    history holds per-episode training statistics (exploratory behaviour) and,
    every cfg.curve_eval_every episodes, a greedy evaluation on seeds that are
    separate from the training stream.
    """
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID)
    low, width = _grid(env, cfg.n_bins)
    q = init_q_table(cfg, env.action_space.n, rng)
    render_env = gym.make(ENV_ID, render_mode="human") if render_every else None

    train_return = np.zeros(cfg.episodes)
    train_success = np.zeros(cfg.episodes, dtype=bool)
    ckpt_episode, ckpt_success, ckpt_return = [], [], []
    curve_seeds = [cfg.curve_eval_seed_start + i for i in range(cfg.curve_eval_episodes)]

    for episode in range(cfg.episodes):
        epsilon = epsilon_at(episode, cfg)
        # Render episodes are drawn from a second environment that mirrors the
        # training environment's state, so rendering does not change the run.
        mirror = render_env is not None and episode % render_every == 0
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        if mirror:
            render_env.reset()
        state = discretize(obs, low, width, cfg.n_bins)
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            action = select_action(q[state], epsilon, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            if mirror:
                render_env.unwrapped.state = env.unwrapped.state
                render_env.render()
            next_state = discretize(obs, low, width, cfg.n_bins)
            q_update(q, state, action, reward, next_state, terminated,
                     cfg.learning_rate, cfg.discount)
            state = next_state
            total += reward
        train_return[episode] = total
        train_success[episode] = terminated

        if cfg.curve_eval_every and (episode + 1) % cfg.curve_eval_every == 0:
            ev = evaluate(q, cfg.n_bins, curve_seeds)
            ckpt_episode.append(episode + 1)
            ckpt_success.append(ev["success_rate"])
            ckpt_return.append(ev["mean_return"])
            if progress:
                progress(episode + 1, {"epsilon": epsilon, "greedy_success": ev["success_rate"],
                                       "train_success": float(train_success[max(0, episode - 499):episode + 1].mean())})
    env.close()
    if render_env is not None:
        render_env.close()
    history = {
        "train_return": train_return,
        "train_success": train_success,
        "ckpt_episode": np.array(ckpt_episode),
        "ckpt_success": np.array(ckpt_success),
        "ckpt_return": np.array(ckpt_return),
    }
    return q, history
