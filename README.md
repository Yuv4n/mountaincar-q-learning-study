# MountainCar Q-learning

An early tabular reinforcement-learning study using Python, Gym and NumPy. The car must build enough momentum to reach the flag. The script represents position and velocity with a 20 × 20 grid and stores three action values per state.

`train.py` contains the state conversion and Q-value update inside a 25,000-episode loop. Those are configuration values. No measured training result is included.

## Run status

Training has not been reproduced. The script expects Gym's old observation-only `reset()` and four-value `step()`, plus NumPy's removed `np.int` alias. Original package versions were not recorded; `requirements.txt` is an import list, not a verified environment lock.

In a compatible legacy environment, run from this folder:

```sh
python3 train.py
```

It immediately starts training and periodically opens a render window. The bin-width expression is wrong. An unconditional `argmax` also replaces the random exploratory action, so the intended epsilon-greedy policy is not implemented correctly.

There are no saved policies or independent evaluation runs. The next step is to correct state conversion and exploration, then compare seeded policy returns with a random baseline.

[Actual training flow and technical notes](docs/notes.md) · [Source dates](docs/history.md)
