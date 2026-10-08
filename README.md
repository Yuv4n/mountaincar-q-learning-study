# MountainCar Q-learning study

An early tabular reinforcement-learning experiment for Gym's MountainCar environment. It implements a Q-table update but has defects that prevent a credible learning result.

## Implementation

`src/train.py` attempts to map position and velocity into 20 bins each, creates a 20 × 20 × 3 Q-table, and runs 25,000 episodes. The update uses a learning rate of 0.1 and discount of 0.95. Training occasionally renders and prints goal events.

## ML review

The problem is to drive the simulated car to the goal. Observations come from the simulator; there is no downloaded dataset, train/test preprocessing pipeline or supervised classification model.

The bin-width expression computes `high - low / bins` rather than `(high - low) / bins`. The code then overwrites an epsilon-greedy action with `argmax`, disabling its intended exploration. Upper-bound clipping is absent. Terminal goal handling sets the Q-value to zero; unsuccessful terminal transitions are not updated.

There is no seed control, frozen-policy evaluation, random-policy baseline, return curve, success-rate calculation or saved Q-table. Printed goal events occur during training. Treating them as held-out performance would mix training and evaluation. No dataset leakage was identified because there is no dataset split; independent evaluation is absent.

## Reproduction status

The script passed syntax compilation. Training was not run: system Python 3.9.6 had neither Gym nor NumPy installed. The backup contains no compatible package lockfile. `requirements.txt` lists imports only and is not a verified environment lock.

The script expects the old Gym interface: reset returns an observation, step returns four values, and rendering needs no configured render mode. It uses `np.int`, which is absent from recent NumPy versions. Installing current packages does not establish compatibility. Once a compatible legacy environment is recreated, run from the repository root:

```sh
python3 src/train.py
```

This starts the full training loop and rendering immediately. There is no short-run CLI. No successful training or reward claim is made.

## Existing material and sources

The supplied Q-table illustration was excluded from the public repository because its redistribution rights are unidentified. It was a learning aid, not an experiment result. The comments read as lesson notes; no tutor attribution is recorded. Keep this project labelled as a learning study until provenance is resolved.

## Proposed improvements

Fix binning and exploration first. Add seeds, a short-run option and explicit environment versions. Handle termination and time limits separately. Save the policy and evaluate it on independent seeds without learning. Report returns and success rates alongside a random baseline.

## Provenance

This repository was organised from a local coding backup on 7 October 2026. Original program bodies were preserved. File names and locations were changed for navigation. The source-to-destination inventory is in the portfolio root. No Git history was present in the source folder. Public-release exclusions are recorded in PUBLIC-EXCLUSIONS.md.

No project licence was found. Establish authorship and external-source permissions before distributing the code under a licence. Coursework and tutorial examples are labelled here.

## Public version

See [PUBLIC-EXCLUSIONS.md](PUBLIC-EXCLUSIONS.md). Some review results describe files excluded from this public version. Local historical copies remain in the organised portfolio.

## Recorded work dates

See [DATE-PROVENANCE.md](DATE-PROVENANCE.md) for dates in the original source and the limits of the imported history.
