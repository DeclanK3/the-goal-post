"""Empirical timing extraction and storage for NFL transitions.

Timing is keyed by (context, source_state, destination_state) like the
probability matrices, with a same-time-bucket pooled fallback.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl

from .extractor import extract_game, NFLExtractedGame


def _time_bucket(ctx: int) -> int:
    """Return time bucket 0 (>120s), 1 (30-120s), or 2 (<=30s) from context."""
    return ctx // 3


class NFLTimingData:
    """Stores elapsed-time observations keyed by (ctx, src, dst).

    Primary:    timing[ctx][src][dst] -> np.ndarray of floats
    Fallback:   timing_by_bucket[bucket][src][dst] -> np.ndarray of floats
    """

    def __init__(self):
        # Primary timing: dict[(ctx, src, dst)] -> list[float] of elapsed times
        self._primary: dict[tuple[int, int, int], list[float]] = {}
        # Fallback timing: dict[(bucket, src, dst)] -> list[float] of elapsed times
        self._fallback: dict[tuple[int, int, int], list[float]] = {}
        self._validated = False

    def add(self, ctx: int, src: int, dst: int, elapsed: float) -> None:
        """Append an elapsed-time observation to primary and fallback stores."""
        if not np.isfinite(elapsed) or elapsed < 0:
            return
        key = (ctx, src, dst)
        self._primary.setdefault(key, []).append(float(elapsed))
        bucket = _time_bucket(ctx)
        self._fallback.setdefault((bucket, src, dst), []).append(float(elapsed))

    def _set_primary(self, key: tuple[int, int, int], arr: np.ndarray) -> None:
        self._primary[key] = arr.tolist()

    def _set_fallback(self, key: tuple[int, int, int], arr: np.ndarray) -> None:
        self._fallback[key] = arr.tolist()

    def get(self, ctx: int, src: int, dst: int) -> list[float] | None:
        """Return primary timing array, or None if absent."""
        return self._primary.get((ctx, src, dst))

    def get_fallback(self, ctx: int, src: int, dst: int) -> list[float] | None:
        """Return same-time-bucket fallback timing array, or None."""
        return self._fallback.get((_time_bucket(ctx), src, dst))

    def sample_duration(self, ctx: int, src: int, dst: int, rng: np.random.Generator) -> float:
        """Sample an elapsed duration for (ctx, src, dst), using fallback if needed.

        Raises ValueError if no timing evidence exists at either level.
        """
        times = self.get(ctx, src, dst)
        if times is None or len(times) == 0:
            times = self.get_fallback(ctx, src, dst)
        if times is None or len(times) == 0:
            raise ValueError(
                f"No timing evidence for transition (ctx={ctx}, src={src}, dst={dst}) "
                f"at primary or same-time-bucket fallback level"
            )
        # Uniform sample from observed durations (same as reference)
        idx = int(rng.integers(len(times)))
        return float(times[idx])

    def validate_no_zero_loops(self, strict: bool = True) -> None:
        """Check for zero-duration closed loops that could trap simulation.

        A closed loop exists when a set of states can cycle with zero
        elapsed time and zero scoring, never reaching an absorbing state
        or positive-time transition.
        """
        # Build directed graph of zero-time, zero-score transitions
        # Nodes: (src, dst) physical-state pairs that can occur
        # We only check physical -> physical transitions (absorbing exits are safe)
        zero_edges: dict[int, set[int]] = defaultdict(set)
        for (ctx, src, dst), times in self._primary.items():
            if dst >= 72:  # absorbing states are safe exits
                continue
            if all(t == 0.0 for t in times):
                zero_edges[src].add(dst)

        # Find nodes reachable from themselves via zero-time paths
        trapped = set()
        for start in zero_edges:
            visited = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    if node == start:
                        trapped.add(start)
                    continue
                visited.add(node)
                for nxt in zero_edges.get(node, set()):
                    stack.append(nxt)

        if trapped:
            msg = f"Zero-duration closed loop detected in physical states: {sorted(trapped)}"
            if strict:
                raise ValueError(msg)
            else:
                print(f"WARNING: {msg}")

        self._validated = True


def build_timing_from_raw(game_id: str, raw_df: pl.DataFrame) -> NFLTimingData | None:
    """Build timing data for a single game from raw nflverse pbp.

    Returns None if the game cannot be extracted.
    """
    try:
        game = extract_game(raw_df)
    except ValueError:
        return None

    if not isinstance(game, NFLExtractedGame):
        return None

    timing = NFLTimingData()
    for e in game.events:
        timing.add(e.ctx, e.source, e.destination, e.elapsed)

    return timing


def load_timing_from_cache(cache_path: Path | None = None) -> dict:
    """Load precomputed timing cache as dict of arrays.

    Returns dict with 'primary' and 'fallback' numpy arrays keyed by tuples.
    """
    if cache_path is None:
        cache_path = Path("/home/admin/repos/goal_post/artifacts/timing/timing_cache.npz")
    if not cache_path.exists():
        return {}

    data = np.load(cache_path, allow_pickle=False)
    primary: dict[tuple[int, int, int], np.ndarray] = {}
    fallback: dict[tuple[int, int, int], np.ndarray] = {}

    for key in data.files:
        parts = key.split("_")
        if parts[0] == "p" and len(parts) == 4:
            primary[(int(parts[1]), int(parts[2]), int(parts[3]))] = data[key]
        elif parts[0] == "f" and len(parts) == 4:
            fallback[(int(parts[1]), int(parts[2]), int(parts[3]))] = data[key]

    return {"primary": primary, "fallback": fallback}


_TIMING_CACHE: dict | None = None
_TIMING_CACHE_VERSION: str = ""


def _get_cached_timing(version: str = "v1") -> dict:
    """Lazy-load the global timing cache with version invalidation."""
    global _TIMING_CACHE, _TIMING_CACHE_VERSION
    if _TIMING_CACHE is None or _TIMING_CACHE_VERSION != version:
        _TIMING_CACHE = load_timing_from_cache()
        _TIMING_CACHE_VERSION = version
    return _TIMING_CACHE


_TIMING_DATA_SINGLETON: NFLTimingData | None = None


def load_timing_for_game(game_id: str, cache_dir: Path | None = None) -> NFLTimingData | None:
    """Load timing for a specific game_id from the precomputed cache.

    The cache contains all games from the VAE dataset. Returns None only if
    the cache is missing.
    """
    global _TIMING_DATA_SINGLETON
    if _TIMING_DATA_SINGLETON is not None:
        return _TIMING_DATA_SINGLETON

    cached = _get_cached_timing()
    if not cached:
        return None

    timing = NFLTimingData()
    # Keep numpy arrays directly (much faster for indexing)
    for (ctx, src, dst), arr in cached["primary"].items():
        timing._set_primary((ctx, src, dst), arr)
    for (bucket, src, dst), arr in cached["fallback"].items():
        timing._set_fallback((bucket, src, dst), arr)

    _TIMING_DATA_SINGLETON = timing
    return timing
