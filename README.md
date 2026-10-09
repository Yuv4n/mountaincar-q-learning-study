# MountainCar Q-learning

Tabular Q-learning on Gymnasium's MountainCar-v0. The car starts in a valley, the engine is too weak to climb the right-hand hill directly, and each step costs -1 until the flag is reached or the 200-step limit ends the episode. The agent has to learn to rock back and forth to build momentum.

The observation (position and velocity) is continuous, so it is binned into a 40 x 40 grid with one Q-value per action in each cell. This started as a 2021 script, and this version rewrites it so that it can be tested and measured.

## Results

Five agents were trained with identical settings and seeds 0 to 4. Each frozen greedy policy was then run for 200 episodes on environment seeds 10000 to 10199, which were not used for training or for choosing hyperparameters. The Q-table is not updated during evaluation. The random-policy baseline uses the same 200 environment seeds.

| training seed | goal reached | mean return |
|---|---|---|
| 0 | 100% | -127.8 |
| 1 | 100% | -128.6 |
| 2 | 100% | -135.0 |
| 3 | 96% | -120.5 |
| 4 | 100% | -119.1 |
| mean ± std over seeds | 99.2% ± 1.8% | -126.2 ± 6.5 |
| random policy | 0% | -200.0 |

The standard deviation is the sample standard deviation over the five training seeds. A return of -126 means the goal was reached after about 126 steps on average. Raw numbers, the configuration and the seed lists are in [results/final/results.json](results/final/results.json), and the five Q-tables are alongside it.

![Learning curve](results/final/learning_curve.png)

The left panel is measured during training, while the agent is still exploring with a decaying epsilon, so it mixes learning progress with the exploration schedule. The right panel is separate: every 500 episodes the current greedy policy is run for 20 episodes on seeds 20000 to 20019 with no exploration and no updates. Success in both panels starts to rise around episode 5,000 and the greedy policy is reliable on the checkpoint seeds from about episode 18,500. These evaluations share MountainCar's fixed start-state distribution with training, so they show that the policy handles unseen start positions of the same task, not other environments.

## What the code does

- `qlearning.py` holds discretisation, epsilon-greedy selection, the Q-update, training and evaluation as separate functions. Importing it does nothing.
- Bin indices are clipped to the table. Without clipping, an observation exactly at the upper limit would index one past the end.
- Gymnasium's `terminated` and `truncated` flags are handled separately. The 200-step time limit is not a real end of the task, so a truncated step still bootstraps from the next state. A step that reaches the flag uses the observed reward with no future term.
- Exploration is epsilon-greedy with epsilon falling linearly from 1 to 0 over the first half of training.
- Training is headless. `--render-every N` opens a window every N episodes.
- Hyperparameters were compared on separate validation seeds before the final run. The table and selection rule are in [docs/tuning.md](docs/tuning.md).
- `tests/` covers bin boundaries and clipping, action selection, the epsilon schedule, terminal versus truncated updates, and that evaluation leaves the table unchanged.

The original script and what was checked in it are described in [docs/notes.md](docs/notes.md).

## Setup and reproduction

Tested with Python 3.9.6 on macOS, with the pinned versions in `requirements.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Reproduce the reported experiment (about one minute with five CPU cores):

```sh
python experiment.py --config configs/final.json --out results/final
```

Train a single agent and compare it with the random baseline:

```sh
python train.py --seed 0 --out runs/seed0
python train.py --help
```

Training seed 0 from `train.py` with its default arguments produced a Q-table identical to the one saved for seed 0 in the experiment.

## Limits

This is one small, discrete-grid method on one environment, and the final comparison uses five training seeds. Hyperparameter tuning used three seeds and eight settings, so the chosen setting is not shown to be clearly better than the others. The one failure case in the table, seed 3 at 96%, was not investigated further. Rendering was only checked with a dummy video driver, not a visible window.
