"""Stage 1: Instrumentation-only diagnosis.

Adds diagnostics to NFLEngine without changing behavior.
Collects per-simulation diagnostics and compares to historical truth from raw pbp.
"""
import json
import sys
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import polars as pl

from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.representation.nfl.builder import N_STATES, N_CONTEXTS, N_PHYSICAL_STATES
from goalpost.simulation.nfl.engine import NFLEngine, _context_index


class InstrumentedNFLEngine(NFLEngine):
    """Wraps NFLEngine with diagnostic collection, no behavior changes."""

    def __init__(self):
        super().__init__()
        self.diag = defaultdict(int)
        self.per_sim = []
        self.missing_row_log = []
        self.fallback_log = []

    def simulate(self, home_model, away_model, n_sims=10000, seed=0):
        rng = np.random.default_rng(seed)
        outcomes = []
        for sim_idx in range(n_sims):
            home_score, away_score, sim_diag = self._simulate_game_instrumented(
                home_model, away_model, rng, sim_idx
            )
            outcomes.append((home_score, away_score))
            self.per_sim.append(sim_diag)
        # Convert to GameOutcome-like tuples
        from goalpost.simulation.base.models import GameOutcome
        return [GameOutcome(home_score=h, away_score=a) for h, a in outcomes]

    def _simulate_game_instrumented(self, home_model, away_model, rng, sim_idx):
        home = away = 0
        starter = int(rng.integers(2))

        sim_diag = {
            "sim_idx": sim_idx,
            "possessions": 0,
            "points_from_drives": 0,
            "seconds_per_possession": [],
            "drive_outcomes": Counter(),
            "breaks": [],
            "fallbacks": 0,
            "primary_lookups": 0,
            "total_steps": 0,
            "half_steps": [0, 0],
        }

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
                used_fallback = False
                sim_diag["primary_lookups"] += 1

                if not probs and hasattr(model, '_fallback_matrix') and model._fallback_matrix is not None:
                    fb = model._fallback_matrix[ctx, state]
                    nz = np.flatnonzero(fb)
                    if len(nz) > 0:
                        probs = {int(j): float(fb[j]) for j in nz}
                        used_fallback = True
                        sim_diag["fallbacks"] += 1
                        self.fallback_log.append({
                            "sim_idx": sim_idx, "half": half + 1, "ctx": ctx, "state": state,
                            "team": model.team_id, "game_id": model.game_id,
                        })

                if not probs:
                    sim_diag["breaks"].append({
                        "sim_idx": sim_idx, "half": half + 1, "ctx": ctx, "state": state,
                        "remaining": remaining, "home": home, "away": away,
                        "team": model.team_id,
                    })
                    self.missing_row_log.append({
                        "sim_idx": sim_idx, "half": half + 1, "ctx": ctx, "state": state,
                        "remaining": remaining, "team": model.team_id, "game_id": model.game_id,
                    })
                    break

                states = list(probs.keys())
                weights = np.array(list(probs.values()), dtype=np.float64)
                weights /= weights.sum()
                next_state = rng.choice(states, p=weights)
                sim_diag["total_steps"] += 1
                sim_diag["half_steps"][half] += 1

                if next_state >= N_PHYSICAL_STATES:
                    own_pts, opp_pts = model.get_point_values(ctx, state, next_state)
                    if offense == 0:
                        home += own_pts; away += opp_pts
                    else:
                        away += own_pts; home += opp_pts

                    # Drive ended — record outcome and timing
                    elapsed = possession_start - remaining
                    sim_diag["possessions"] += 1
                    sim_diag["points_from_drives"] += own_pts
                    sim_diag["seconds_per_possession"].append(elapsed)
                    sim_diag["drive_outcomes"][next_state] += 1

                    offense = 1 - offense
                    state = 16
                    possession_start = remaining
                else:
                    state = next_state

                remaining -= rng.exponential(30.0)

            if remaining <= 0:
                remaining = 0

        return home, away, sim_diag


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


def analyze_raw_pbp():
    """Load raw nflverse pbp and compute historical truth for comparison."""
    cache_dir = Path.home() / ".cache" / "goalpost" / "nflverse"
    all_games = []
    for year in [2023, 2024, 2025, 2026]:
        path = cache_dir / f"nfl-{year}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        all_games.append(df)

    if not all_games:
        return None

    df = pl.concat(all_games)

    # Historical possessions: count unique posteam per game
    possessions = df.group_by("game_id").agg(
        pl.col("posteam").n_unique().alias("n_possessions"),
        pl.len().alias("n_plays"),
    )

    # Historical drive outcomes from scoring plays
    scoring = df.filter(
        (pl.col("posteam_score_post") - pl.col("posteam_score_pre")).is_not_null()
    )

    # Points per possession approximation
    # Total points / possessions
    return {
        "total_rows": len(df),
        "games": df["game_id"].n_unique(),
    }


def main(n_sims=100, seed=42):
    dataset_dir = Path("/home/admin/repos/goal_post/artifacts/dataset")
    models, game_info = _load_models(dataset_dir)
    print(f"Loaded {len(game_info)} games", file=sys.stderr)

    engine = InstrumentedNFLEngine()

    all_sim_diags = []
    game_results = []

    for idx, (gid, info) in enumerate(game_info.items()):
        hm = models[(info["home_team"], gid)]
        am = models[(info["away_team"], gid)]
        outcomes = engine.simulate(hm, am, n_sims=n_sims, seed=seed)

        hs = np.array([o.home_score for o in outcomes])
        aws = np.array([o.away_score for o in outcomes])

        game_results.append({
            "gid": gid,
            "actual_h": info["home_score"],
            "actual_a": info["away_score"],
            "sim_h": float(hs.mean()),
            "sim_a": float(aws.mean()),
        })

        # Collect per-sim diagnostics for this game
        # (engine.per_sim was populated during simulate)
        # But per_sim accumulates across all games — need to slice
        pass

    # Actually, the instrumented engine accumulates globally. Let's just use that.
    # But we need to know which sims belong to which game...
    # Easier: run one game at a time and collect.

    # Re-run per-game for diagnostics
    all_sim_diags = []
    all_missing_rows = []
    all_fallbacks = []

    for idx, (gid, info) in enumerate(game_info.items()):
        hm = models[(info["home_team"], gid)]
        am = models[(info["away_team"], gid)]

        game_engine = InstrumentedNFLEngine()
        outcomes = game_engine.simulate(hm, am, n_sims=n_sims, seed=seed)

        all_sim_diags.extend(game_engine.per_sim)
        all_missing_rows.extend(game_engine.missing_row_log)
        all_fallbacks.extend(game_engine.fallback_log)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(game_info)} games instrumented", file=sys.stderr)

    # --- Aggregate Diagnostics ---

    # a. possessions per game, points per possession, seconds per possession
    possessions_per_game = []
    points_per_possession = []
    seconds_per_possession = []
    for sd in all_sim_diags:
        possessions_per_game.append(sd["possessions"])
        if sd["possessions"] > 0:
            points_per_possession.append(sd["points_from_drives"] / sd["possessions"])
            seconds_per_possession.extend(sd["seconds_per_possession"])

    # c. drive-ending outcome mix
    outcome_mix = Counter()
    for sd in all_sim_diags:
        outcome_mix.update(sd["drive_outcomes"])

    # d. empty-row breaks
    break_ctx_counts = Counter()
    break_remaining = []
    for br in all_missing_rows:
        break_ctx_counts[(br["ctx"], br["state"])] += 1
        break_remaining.append(br["remaining"])

    # e. fallback fraction
    total_lookups = sum(sd["primary_lookups"] for sd in all_sim_diags)
    total_fallbacks = len(all_fallbacks)

    # b. Flat mean elapsed vs empirical — since we only have the constant, report what it is
    flat_mean_elapsed = 30.0  # The exponential parameter is the mean for scipy/numpy

    # Calibration
    ah = np.array([r["actual_h"] for r in game_results])
    aa = np.array([r["actual_a"] for r in game_results])
    sh = np.array([r["sim_h"] for r in game_results])
    sa = np.array([r["sim_a"] for r in game_results])

    report = {
        "n_games": len(game_info),
        "n_sims": n_sims,
        "seed": seed,
        "calibration": {
            "home_mae": float(np.mean(np.abs(sh - ah))),
            "away_mae": float(np.mean(np.abs(sa - aa))),
            "home_bias": float(np.mean(sh - ah)),
            "away_bias": float(np.mean(sa - aa)),
            "total_bias": float(np.mean((sh + sa) - (ah + aa))),
            "sim_mean_total": float(np.mean(sh + sa)),
            "actual_mean_total": float(np.mean(ah + aa)),
        },
        "diagnostics": {
            "a_possessions": {
                "mean_per_game": float(np.mean(possessions_per_game)) if possessions_per_game else 0,
                "median_per_game": float(np.median(possessions_per_game)) if possessions_per_game else 0,
                "points_per_possession_mean": float(np.mean(points_per_possession)) if points_per_possession else 0,
                "seconds_per_possession_mean": float(np.mean(seconds_per_possession)) if seconds_per_possession else 0,
            },
            "b_timing": {
                "flat_mean_elapsed": flat_mean_elapsed,
                "note": "Engine uses rng.exponential(30.0); no empirical timing collected",
            },
            "c_drive_outcomes": dict(outcome_mix),
            "d_empty_row_breaks": {
                "count": len(all_missing_rows),
                "rate_per_sim": len(all_missing_rows) / (len(game_info) * n_sims),
                "by_context_state": dict(break_ctx_counts),
                "seconds_remaining_mean": float(np.mean(break_remaining)) if break_remaining else 0,
                "seconds_remaining_hist": {
                    "0-30s": sum(1 for r in break_remaining if 0 < r <= 30),
                    "30-120s": sum(1 for r in break_remaining if 30 < r <= 120),
                    ">120s": sum(1 for r in break_remaining if r > 120),
                },
            },
            "e_fallback_fraction": {
                "total_lookups": total_lookups,
                "fallback_uses": total_fallbacks,
                "fallback_rate": total_fallbacks / total_lookups if total_lookups else 0,
            },
            "f_zero_duration_loops": {
                "note": "Cannot detect from engine alone; requires raw segment analysis",
                "status": "unverified",
            },
        },
    }

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    main(ns, sd)
