import os
import subprocess
import sys

import numpy as np
import pytest

import qlearning as ql

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOW = np.array([-1.2, -0.07])
HIGH = np.array([0.6, 0.07])
N = 20
WIDTH = ql.bin_widths(LOW, HIGH, N)


def test_bin_width_divides_range_before_bins():
    np.testing.assert_allclose(WIDTH, [0.09, 0.007])


def test_lower_and_upper_bounds_map_to_valid_indices():
    assert ql.discretize(LOW, LOW, WIDTH, N) == (0, 0)
    # The observation-space maximum would index N without clipping.
    assert ql.discretize(HIGH, LOW, WIDTH, N) == (N - 1, N - 1)


def test_out_of_range_observations_are_clipped():
    assert ql.discretize([-5.0, -5.0], LOW, WIDTH, N) == (0, 0)
    assert ql.discretize([5.0, 5.0], LOW, WIDTH, N) == (N - 1, N - 1)


def test_bins_cover_the_whole_grid():
    pos = np.linspace(LOW[0], HIGH[0], 2000)
    idx = {ql.discretize([p, 0.0], LOW, WIDTH, N)[0] for p in pos}
    assert idx == set(range(N))


def test_epsilon_zero_is_always_greedy():
    rng = np.random.default_rng(0)
    q = np.array([0.1, 0.9, 0.3])
    assert all(ql.select_action(q, 0.0, rng) == 1 for _ in range(200))


def test_epsilon_one_explores_every_action():
    rng = np.random.default_rng(0)
    q = np.array([0.1, 0.9, 0.3])
    assert {ql.select_action(q, 1.0, rng) for _ in range(200)} == {0, 1, 2}


def test_epsilon_fraction_of_random_actions():
    rng = np.random.default_rng(1)
    q = np.array([0.0, 1.0, 0.0])
    picks = [ql.select_action(q, 0.5, rng) for _ in range(20000)]
    # Greedy half plus a third of the random half should land on action 1.
    assert np.mean(np.array(picks) == 1) == pytest.approx(0.5 + 0.5 / 3, abs=0.02)


def test_epsilon_schedule():
    cfg = ql.Config(episodes=100, epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_fraction=0.5)
    assert ql.epsilon_at(0, cfg) == 1.0
    assert ql.epsilon_at(25, cfg) == pytest.approx(0.5)
    assert ql.epsilon_at(50, cfg) == 0.0
    assert ql.epsilon_at(99, cfg) == 0.0


def _table():
    q = np.zeros((N, N, 3))
    q[(5, 5)] = [1.0, 2.0, 3.0]  # next state, max 3.0
    return q


def test_terminal_update_uses_reward_only():
    q = _table()
    new = ql.q_update(q, (1, 1), 0, -1.0, (5, 5), True, 0.5, 0.9)
    assert new == pytest.approx(0.5 * 0.0 + 0.5 * -1.0)


def test_truncated_update_still_bootstraps():
    q = _table()
    # Truncation is not terminated, so the caller passes terminated=False.
    new = ql.q_update(q, (1, 1), 0, -1.0, (5, 5), False, 0.5, 0.9)
    assert new == pytest.approx(0.5 * (-1.0 + 0.9 * 3.0))


def test_update_changes_only_the_chosen_entry():
    q = _table()
    before = q.copy()
    ql.q_update(q, (1, 1), 2, -1.0, (5, 5), False, 0.1, 0.95)
    diff = np.argwhere(q != before)
    assert diff.tolist() == [[1, 1, 2]]


def test_importing_modules_does_not_train():
    code = "import qlearning, train, experiment"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout == ""


def test_evaluate_does_not_modify_q_table():
    q = np.random.default_rng(0).uniform(-2, 0, (N, N, 3))
    before = q.copy()
    q.flags.writeable = False
    result = ql.evaluate(q, N, [1, 2])
    np.testing.assert_array_equal(q, before)
    assert result["episodes"] == 2


def test_truncation_is_reported_distinctly_from_termination():
    # A random-ish greedy policy on a zero table never reaches the goal in 200 steps.
    result = ql.evaluate(np.zeros((N, N, 3)), N, [0])
    assert result["successes"] == [False]
    assert result["mean_steps"] == 200


SMALL = ql.Config(episodes=60, curve_eval_every=30, curve_eval_episodes=3)


def test_same_seed_gives_identical_q_table():
    q1, _ = ql.train(SMALL, seed=3)
    q2, _ = ql.train(SMALL, seed=3)
    q3, _ = ql.train(SMALL, seed=4)
    np.testing.assert_array_equal(q1, q2)
    assert not np.array_equal(q1, q3)


def test_rendering_does_not_change_training(monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    headless, _ = ql.train(SMALL, seed=5)
    rendered, _ = ql.train(SMALL, seed=5, render_every=20)
    np.testing.assert_array_equal(headless, rendered)


def test_velocity_heuristic_reaches_goal_and_beats_random():
    seeds = [1, 2, 3]
    heuristic = ql.evaluate_heuristic(seeds)
    random_policy = ql.evaluate_random(seeds)
    assert heuristic["success_rate"] == 1.0
    assert heuristic["mean_return"] > random_policy["mean_return"]


def test_wilson_interval():
    lo, hi = ql.wilson_interval(100, 100)
    assert hi == pytest.approx(1.0) and 0.96 < lo < 0.97
    lo, hi = ql.wilson_interval(0, 200)
    assert lo == 0.0 and 0.0 < hi < 0.03
    lo, hi = ql.wilson_interval(50, 100)
    assert lo < 0.5 < hi


def test_legacy_bin_width_reproduces_precedence_error():
    np.testing.assert_allclose(ql.bin_widths(LOW, HIGH, N, legacy=True), [0.66, 0.0735])
    # Only a few cells are reachable with the legacy widths.
    pos = np.linspace(LOW[0], HIGH[0], 1000)
    vel = np.linspace(LOW[1], HIGH[1], 1000)
    legacy = ql.bin_widths(LOW, HIGH, N, legacy=True)
    assert {ql.discretize([p, 0.0], LOW, legacy, N)[0] for p in pos} == {0, 1, 2}
    assert {ql.discretize([0.0, v], LOW, legacy, N)[1] for v in vel} == {0, 1}


def test_ignore_exploration_matches_a_fully_greedy_run():
    forced = ql.Config(episodes=40, curve_eval_every=0, epsilon_start=1.0, ignore_exploration=True)
    greedy = ql.Config(episodes=40, curve_eval_every=0, epsilon_start=0.0, ignore_exploration=False)
    q1, _ = ql.train(forced, seed=1)
    q2, _ = ql.train(greedy, seed=1)
    np.testing.assert_array_equal(q1, q2)
