"""Reproducible calibration and diagnostics for NFL transition model.

Usage: PYTHONPATH=src python calibrate.py [n_sims] [seed]
"""
import json, sys, time
import numpy as np
import polars as pl
from pathlib import Path

from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.representation.nfl.timing import load_timing_for_game
from goalpost.simulation.nfl.engine import NFLEngine


def _load_models(dataset_dir: Path):
    df = pl.read_parquet(dataset_dir / "vae_dataset.parquet")
    home_cols = [c for c in df.columns
                   if c.startswith("home_") and c not in ("home_team", "home_score", "home_available")]
    away_cols = [c for c in df.columns
                   if c.startswith("away_") and c not in ("away_team", "away_score", "away_available")]

    # Pre-load timing once (singleton)
    timing = load_timing_for_game("dummy")

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
        if timing is not None:
            hm._timing = timing
            am._timing = timing
        models[(ht, gid)] = hm
        models[(at, gid)] = am
    return models, game_info


def _run(n_sims: int, seed: int):
    dataset_dir = Path("/home/admin/repos/goal_post/artifacts/dataset")
    models, game_info = _load_models(dataset_dir)
    engine = NFLEngine()

    results = []
    t0 = time.time()
    for idx, (gid, info) in enumerate(game_info.items()):
        hm = models[(info["home_team"], gid)]
        am = models[(info["away_team"], gid)]
        try:
            outcomes = engine.simulate(hm, am, n_sims=n_sims, seed=seed)
        except Exception as e:
            print(f"ERROR on game {gid}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            continue
        hs = np.array([o.home_score for o in outcomes])
        aws = np.array([o.away_score for o in outcomes])
        results.append({
            "gid": gid, "week": info["week"],
            "actual_h": info["home_score"], "actual_a": info["away_score"],
            "sim_h": float(hs.mean()), "sim_a": float(aws.mean()),
            "hs": hs, "aws": aws,
        })
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(game_info)} in {time.time()-t0:.1f}s", file=sys.stderr)

    ah = np.array([r["actual_h"] for r in results])
    aa = np.array([r["actual_a"] for r in results])
    at = ah + aa
    sh = np.array([r["sim_h"] for r in results])
    sa = np.array([r["sim_a"] for r in results])
    st = sh + sa

    # Coverage
    c50 = c90 = 0
    for r in results:
        tots = r["hs"] + r["aws"]
        if np.percentile(tots, 25) <= r["actual_h"] + r["actual_a"] <= np.percentile(tots, 75):
            c50 += 1
        if np.percentile(tots, 5) <= r["actual_h"] + r["actual_a"] <= np.percentile(tots, 95):
            c90 += 1

    report = {
        "n_games": len(results), "n_sims": n_sims, "seed": seed,
        "home_mae": float(np.mean(np.abs(sh - ah))),
        "away_mae": float(np.mean(np.abs(sa - aa))),
        "home_bias": float(np.mean(sh - ah)),
        "away_bias": float(np.mean(sa - aa)),
        "total_bias": float(np.mean(st - at)),
        "sim_mean_total": float(np.mean(st)),
        "actual_mean_total": float(np.mean(at)),
        "50pct_coverage": c50 / len(results) * 100,
        "90pct_coverage": c90 / len(results) * 100,
        "zero_zero": int(np.sum((sh == 0) & (sa == 0))),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    _run(ns, sd)
