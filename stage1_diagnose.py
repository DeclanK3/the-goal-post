"""Stage 1: Diagnose point loss. Fast matrix-level + simulation diagnostics.

Reports simulated vs historical:
a. possessions/game, points/possession, seconds/possession
b. Visit-weighted mean elapsed vs flat mean (from matrix structure)
c. Drive-ending outcome mix
d. Empty-row break frequency, seconds remaining, split by context/team
e. Fraction of rows served by global fallback
f. Zero-duration closed loops (unverified — requires raw segment data)
"""
import json, sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import polars as pl

from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.representation.nfl.builder import (
    N_STATES, N_CONTEXTS, N_PHYSICAL_STATES,
    ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT
)
from goalpost.simulation.nfl.engine import NFLEngine, _context_index


def _load_models(dataset_dir: Path):
    df = pl.read_parquet(dataset_dir / "vae_dataset.parquet")
    home_cols = [c for c in df.columns
                   if c.startswith("home_") and c not in ("home_team", "home_score", "home_available")]
    away_cols = [c for c in df.columns
                   if c.startswith("away_") and c not in ("away_team", "away_score", "away_available")]

    models = {}
    game_info = {}
    for gid_tuple, group in df.group_by("game_id"):
        gid = gid_tuple[0] if isinstance(gid_tuple, tuple) else gid_tuple
        rows = sorted(group.to_dicts(), key=lambda r: r["context_index"])
        if len(rows) != 9:
            continue
        home_flat = np.concatenate([np.array([r[c] for c in home_cols], dtype=np.float32) for r in rows])
        away_flat = np.concatenate([np.array([r[c] for c in away_cols], dtype=np.float32) for r in rows])
        ht = rows[0]["home_team"]
        at = rows[0]["away_team"]
        game_info[gid] = {
            "home_team": ht, "away_team": at,
            "home_score": rows[0]["home_score"], "away_score": rows[0]["away_score"],
            "week": rows[0]["week"],
        }
        hm = NFLTransitionModel.from_flat(home_flat, ht, gid, at)
        am = NFLTransitionModel.from_flat(away_flat, at, gid, ht)
        blend = (hm._matrix + am._matrix) / 2.0
        hm._fallback_matrix = blend.astype(np.float32)
        am._fallback_matrix = blend.astype(np.float32)
        models[(ht, gid)] = hm
        models[(at, gid)] = am
    return models, game_info


def simulate_with_diagnostics(home_model, away_model, n_sims, seed):
    """Fast simulation with aggregated (not per-sim object) diagnostics."""
    rng = np.random.default_rng(seed)
    engine = NFLEngine()

    total_possessions = 0
    total_points = 0
    total_seconds = 0.0
    drive_outcomes = Counter()
    breaks = Counter()
    break_remaining = []
    fallbacks = 0
    lookups = 0
    total_steps = 0

    for sim_idx in range(n_sims):
        home = away = 0
        starter = int(rng.integers(2))
        for half in range(2):
            offense = starter if half == 0 else 1 - starter
            remaining = 1800.0
            state = 16
            possession_start = remaining
            while remaining > 0:
                lead = (home - away) if offense == 0 else (away - home)
                ctx = _context_index(remaining, lead)
                model = home_model if offense == 0 else away_model
                probs = model.get_probabilities(ctx, state)
                lookups += 1

                used_fb = False
                if not probs and hasattr(model, '_fallback_matrix') and model._fallback_matrix is not None:
                    fb = model._fallback_matrix[ctx, state]
                    nz = np.flatnonzero(fb)
                    if len(nz) > 0:
                        probs = {int(j): float(fb[j]) for j in nz}
                        used_fb = True

                if not probs:
                    breaks[(ctx, state)] += 1
                    break_remaining.append(remaining)
                    break

                states = list(probs.keys())
                weights = np.array(list(probs.values()), dtype=np.float64)
                weights /= weights.sum()
                next_state = rng.choice(states, p=weights)
                total_steps += 1

                if next_state >= N_PHYSICAL_STATES:
                    own_pts, opp_pts = model.get_point_values(ctx, state, next_state)
                    if offense == 0:
                        home += own_pts; away += opp_pts
                    else:
                        away += own_pts; home += opp_pts
                    elapsed = possession_start - remaining
                    total_possessions += 1
                    total_points += own_pts
                    total_seconds += elapsed
                    drive_outcomes[next_state] += 1
                    offense = 1 - offense
                    state = 16
                    possession_start = remaining
                else:
                    state = next_state
                remaining -= rng.exponential(30.0)

    return {
        "total_possessions": total_possessions,
        "total_points": total_points,
        "total_seconds": total_seconds,
        "drive_outcomes": drive_outcomes,
        "breaks": breaks,
        "break_remaining": break_remaining,
        "fallbacks": fallbacks,
        "lookups": lookups,
        "total_steps": total_steps,
    }


def analyze_raw_pbp():
    """Historical truth from VAE dataset (already extracted from pbp)."""
    dataset_dir = Path("/home/admin/repos/goal_post/artifacts/dataset")
    df = pl.read_parquet(dataset_dir / "vae_dataset.parquet")
    # Get one row per game (context_index == 0)
    first_ctx = df.filter(pl.col("context_index") == 0)
    return {
        "games": first_ctx["game_id"].n_unique(),
        "mean_total_score": float((first_ctx["home_score"] + first_ctx["away_score"]).mean()),
        "mean_home_score": float(first_ctx["home_score"].mean()),
        "mean_away_score": float(first_ctx["away_score"].mean()),
        "note": "From VAE dataset first-context rows (already extracted from nflverse pbp)",
    }


def main(n_sims=100, seed=42):
    dataset_dir = Path("/home/admin/repos/goal_post/artifacts/dataset")
    models, game_info = _load_models(dataset_dir)
    print(f"Loaded {len(game_info)} games", file=sys.stderr)

    # --- Simulation diagnostics on subset for speed ---
    sample_games = list(game_info.items())[:100]  # 100 games for diagnostics

    all_diagnostics = []
    for idx, (gid, info) in enumerate(sample_games):
        hm = models[(info["home_team"], gid)]
        am = models[(info["away_team"], gid)]
        diag = simulate_with_diagnostics(hm, am, n_sims=n_sims, seed=seed)
        all_diagnostics.append(diag)
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(sample_games)} done", file=sys.stderr)

    # Aggregate across sampled games
    total_poss = sum(d["total_possessions"] for d in all_diagnostics)
    total_pts = sum(d["total_points"] for d in all_diagnostics)
    total_secs = sum(d["total_seconds"] for d in all_diagnostics)
    total_sims = len(sample_games) * n_sims

    all_drive_outcomes = Counter()
    all_breaks = Counter()
    all_break_remaining = []
    total_fallbacks = sum(d["fallbacks"] for d in all_diagnostics)
    total_lookups = sum(d["lookups"] for d in all_diagnostics)
    total_steps = sum(d["total_steps"] for d in all_diagnostics)

    for d in all_diagnostics:
        all_drive_outcomes.update(d["drive_outcomes"])
        all_breaks.update(d["breaks"])
        all_break_remaining.extend(d["break_remaining"])

    # --- Matrix-level analysis: row coverage ---
    total_rows = 0
    observed_rows = 0
    fallback_rows = 0
    empty_rows = 0

    for gid, info in game_info.items():
        for team in [info["home_team"], info["away_team"]]:
            model = models[(team, gid)]
            for c in range(N_CONTEXTS):
                for s in range(N_PHYSICAL_STATES):
                    total_rows += 1
                    probs = model.get_probabilities(c, s)
                    if probs:
                        observed_rows += 1
                    elif hasattr(model, '_fallback_matrix') and model._fallback_matrix is not None:
                        fb = model._fallback_matrix[c, s]
                        if np.any(fb > 0):
                            fallback_rows += 1
                        else:
                            empty_rows += 1
                    else:
                        empty_rows += 1

    # --- Raw PBP truth ---
    historical = analyze_raw_pbp()

    report = {
        "n_sample_games": len(sample_games),
        "n_sims_per_game": n_sims,
        "seed": seed,
        "simulation_diagnostics": {
            "a_possessions": {
                "mean_per_game_simulated": total_poss / total_sims,
                "points_per_possession": total_pts / total_poss if total_poss else 0,
                "seconds_per_possession": total_secs / total_poss if total_poss else 0,
            },
            "b_timing": {
                "flat_mean_elapsed": 30.0,
                "note": "Engine uses constant 30.0s exponential. No empirical timing available.",
            },
            "c_drive_outcomes": {
                "TD": all_drive_outcomes.get(ABSORBING_TD, 0),
                "FG": all_drive_outcomes.get(ABSORBING_FG, 0),
                "TO": all_drive_outcomes.get(ABSORBING_TO, 0),
                "PUNT": all_drive_outcomes.get(ABSORBING_PUNT, 0),
                "total_drive_endings": sum(all_drive_outcomes.values()),
            },
            "d_empty_row_breaks": {
                "count": sum(all_breaks.values()),
                "rate_per_sim": sum(all_breaks.values()) / total_sims,
                "seconds_remaining_mean": float(np.mean(all_break_remaining)) if all_break_remaining else 0,
                "seconds_remaining_hist": {
                    "0-30s": sum(1 for r in all_break_remaining if 0 < r <= 30),
                    "30-120s": sum(1 for r in all_break_remaining if 30 < r <= 120),
                    ">120s": sum(1 for r in all_break_remaining if r > 120),
                },
            },
            "e_fallback_usage": {
                "total_lookups": total_lookups,
                "fallback_uses": total_fallbacks,
                "fallback_rate": total_fallbacks / total_lookups if total_lookups else 0,
            },
            "f_zero_loops": {
                "status": "unverified",
                "note": "Cannot detect from VAE flat vectors alone; need raw segment timing",
            },
        },
        "matrix_coverage": {
            "total_possible_rows": total_rows,
            "observed_rows": observed_rows,
            "fallback_only_rows": fallback_rows,
            "empty_rows": empty_rows,
            "observed_pct": observed_rows / total_rows * 100,
            "fallback_pct": fallback_rows / total_rows * 100,
            "empty_pct": empty_rows / total_rows * 100,
        },
        "historical_truth": historical,
    }

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    main(ns, sd)
