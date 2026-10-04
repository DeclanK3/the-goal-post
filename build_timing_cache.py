"""Precompute timing data from raw nflverse pbp and cache as npz."""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl

from goalpost.representation.nfl.extractor import extract_game, NFLExtractedGame
from goalpost.representation.nfl.builder import (
    ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT
)


def _time_bucket(ctx: int) -> int:
    return ctx // 3


def _map_destination(event) -> int:
    """Same mapping logic as builder._map_destination."""
    if event.destination == -1:
        return -1
    if event.own_points == 3:
        return ABSORBING_FG
    if event.own_points in (6, 7, 8):
        return ABSORBING_TD
    if event.opponent_points > 0:
        return ABSORBING_TO
    if event.play_type == "punt":
        return ABSORBING_PUNT
    if event.switch:
        return ABSORBING_TO
    return event.destination


def build_timing_cache():
    cache_dir = Path.home() / ".cache" / "goalpost" / "nflverse"
    out_dir = Path("/home/admin/repos/goal_post/artifacts/timing")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE dataset to know which game_ids we need
    vae_df = pl.read_parquet("/home/admin/repos/goal_post/artifacts/dataset/vae_dataset.parquet")
    target_gids = set(vae_df["game_id"].unique().to_list())
    print(f"Target games: {len(target_gids)}")

    # Load all raw season data
    all_raw = {}
    for year in [2023, 2024, 2025, 2026]:
        p = cache_dir / f"nfl-{year}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            all_raw[year] = df
            print(f"Loaded {year}: {len(df)} rows")

    # Extract timing for each game using MAPPED destinations (same as probability matrix)
    primary: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    fallback: dict[tuple[int, int, int], list[float]] = defaultdict(list)

    processed = 0
    for gid in target_gids:
        year = int(gid.split("_")[0])
        if year not in all_raw:
            continue

        game_df = all_raw[year].filter(pl.col("game_id") == gid)
        if len(game_df) == 0:
            continue

        try:
            game = extract_game(game_df)
        except ValueError:
            continue

        if not isinstance(game, NFLExtractedGame):
            continue

        for e in game.events:
            if e.elapsed < 0:
                continue
            ctx = e.ctx
            src = e.source
            dst = _map_destination(e)
            if dst == -1:
                continue
            if src >= 72:  # Skip transitions from absorbing states
                continue
            elapsed = float(e.elapsed)
            primary[(ctx, src, dst)].append(elapsed)
            fallback[(_time_bucket(ctx), src, dst)].append(elapsed)

        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(target_gids)}")

    # Save as compressed npz
    primary_arrays = {}
    for key, times in primary.items():
        primary_arrays[f"p_{key[0]}_{key[1]}_{key[2]}"] = np.array(times, dtype=np.float32)

    fallback_arrays = {}
    for key, times in fallback.items():
        fallback_arrays[f"f_{key[0]}_{key[1]}_{key[2]}"] = np.array(times, dtype=np.float32)

    np.savez_compressed(out_dir / "timing_cache.npz", **primary_arrays, **fallback_arrays)
    print(f"Saved timing cache with {len(primary_arrays)} primary and {len(fallback_arrays)} fallback entries")

    # Also save a JSON manifest
    manifest = {
        "n_games": processed,
        "primary_entries": len(primary_arrays),
        "fallback_entries": len(fallback_arrays),
        "primary_total_observations": sum(len(v) for v in primary.values()),
        "fallback_total_observations": sum(len(v) for v in fallback.values()),
    }
    (out_dir / "timing_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build_timing_cache()
