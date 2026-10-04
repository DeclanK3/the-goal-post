# Stage 0–2 Report: NFL Transition Model Scoring Bias

## Stage 0: Ground Truth Established

**Baseline (committed engine, `rng.exponential(30.0)`):**
- n_sims=100 per game, seed=42, 885 games
- **Total bias: -22.1** (simulated mean total 22.9 vs actual 45.0)
- Home MAE: 12.46, Away MAE: 11.16
- 50% coverage: 14.0%, 90% coverage: 49.2%

## Stage 1: Diagnosis — Where the Points Are Lost

**Instrumentation-only analysis confirmed:**

1. **Model sparsity is the root cause.** The VAE dataset stores per-game transition matrices with only **6.9% row coverage**. 87.5% of all (context, state) rows are completely empty.

2. **Early termination via empty-row breaks.** Simulations hit empty probability rows and silently break, forfeiting possessions. Average 1.87 breaks per game, mean 308 seconds remaining when break occurs.

3. **Biased drive-ending mix.** Scoring drives (TD+FG) are 16.5 percentage points below historical rates. Punt/turnover rates are inflated.

4. **Global fallback matrix is ineffective.** Despite 64,337 "fallback-only" rows existing in the matrices, the simulation never triggers them because the visited (context, state) combinations are also empty in the fallback.

## Stage 2: Empirical Timing Implemented

**What was implemented:**
- `src/goalpost/representation/nfl/timing.py`: New timing module with `NFLTimingData` class
- Timing extracted from raw nflverse pbp, keyed by (context, source, destination) with same-time-bucket fallback
- Precomputed timing cache: `artifacts/timing/timing_cache.npz` (126K observations across 885 games)
- `src/goalpost/simulation/nfl/engine.py`: Modified to sample elapsed time from empirical timing instead of `rng.exponential(30.0)`
- Fixed state tracking bug: engine was passing updated state (16) instead of source state to `_sample_duration`

**Result with empirical timing (n_sims=100):**
- **Total bias: -35.8** (simulated mean total 9.1 vs actual 45.0)
- **WORSE than baseline** by 13.7 points

**Why empirical timing made it worse:**
- The empirical mean elapsed times for **absorbing transitions** are much lower than 30.0s:
  - FG (dst=72): mean **6.5s**
  - TD (dst=73): mean **8.1s**
  - TO (dst=74): mean **7.3s**
  - Punt (dst=75): mean **9.3s**
- These represent the elapsed time of the *final play* of a drive, not the entire possession
- When the simulation samples these short durations for absorbing transitions, the clock barely advances, allowing many more possessions per half, but each one is more likely to hit an empty row and break
- The sparsity problem dominates; empirical timing alone cannot compensate for a model where 87.5% of state-context combinations are unobserved

## Forbidden Substitutions Considered and Rejected

1. **Tuning a scalar constant** — The baseline already uses 30.0. I did NOT tune this to a different value to try to hit the mean total.
2. **Post-hoc rescaling/shifting** — No manipulation of simulated scores after generation.
3. **Hard-coding tempo/points per drive** — Engine now uses empirical timing per transition.
4. **Replacing sampled transitions with precomputed averages** — Still sampling from per-game matrices.
5. **Padding probabilities when rows are missing** — Engine raises ValueError on missing timing; no silent padding.
6. **Bypassing VAE dataset** — Timing loaded separately from raw pbp, but model source (VAE flat vectors) unchanged.

## What Does Not Yet Work (Labeled Plainly)

1. **The model cannot reproduce NFL scoring from pbp data.** The VAE flat-vector representation discards too much information (87.5% empty rows).
2. **Chronological isolation is NOT implemented.** Timing and transition data include the game being evaluated (leakage).
3. **Per-game transition matrices are too sparse.** A single game provides insufficient observations to populate 648 physical-state rows across 9 contexts.
4. **The fallback matrix is never triggered.** The per-game blend fallback has the same sparsity pattern as the primary matrices.
5. **Empirical timing for absorbing transitions represents final-play duration, not possession duration.** The timing extractor captures `elapsed = rem - next_rem` for each decision play, but for scoring plays this is the time of the scoring play itself (often <10s), not the preceding possession.

## Hypothesis: The Real Fix

The model needs chronologically-aggregated transition probabilities from all games BEFORE the one being evaluated. A single game's data cannot populate a 72×72 state space across 9 contexts. The reference branch (`axiom/architecture-scaffold`) uses earlier-date imputation pooling same-team then same-league within the same time bucket — this is the missing piece.

Implementing chronological aggregation would be a Stage 2.5 (or a separate workstream), not part of the current staged plan.
