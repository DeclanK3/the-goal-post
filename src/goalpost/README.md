# GoalPost package

The current extraction, chronological imputation and elapsed-clock simulator live in `goalpost.transitions`. See the repository root README for tensor layout, dependencies, commands, tests and known limitations.

`data` and `domain` retain the earlier ingestion scaffold. `kalman_vae` is a separate model scaffold and is not yet integrated with the new transition catalog. The former absorbing-terminal transition model and incompatible training/simulation examples have been retired.
