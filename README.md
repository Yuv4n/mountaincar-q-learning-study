# MountainCar Q-learning

An early attempt at tabular Q-learning for Gym's MountainCar environment. The goal is to get the car up the hill. I mapped position and velocity into a Q-table and wrote the reward update and episode loop.

The script is configured for 25,000 episodes and a 20 × 20 × 3 table. Those are settings, not measured results. Training has not been verified: the bin-width calculation is wrong, and a later `argmax` overrides the exploratory action. There is no saved policy or separate evaluation.

## Running the original

`train.py` expects the old Gym reset/step interface and a NumPy version with `np.int`. The original dependency versions were not saved; `requirements.txt` only lists the imports. Current packages may be incompatible.

```sh
python3 train.py
```

It starts the full training loop immediately. I would fix binning and exploration first, then add seeded runs and evaluation against a random policy.

[Technical notes](docs/notes.md) · [Recorded dates](docs/history.md)
