"""NFL decision-segment extraction from nflverse pbp data."""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
import polars as pl

from ..base.extractor import GameExtractor as BaseExtractor
from ..base.models import ExtractedGame

DECISIONS = {"run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"}
N_PHYSICAL_STATES = 72


def _present(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def state_index(down, distance, yardline):
    if not all(_present(x) for x in (down, distance, yardline)):
        raise ValueError("missing decision state")
    if int(down) != down or not 1 <= down <= 4 or not 0 <= yardline <= 100 or distance < 0:
        raise ValueError("invalid decision state")
    d = 0 if distance <= 3 else 1 if distance <= 7 else 2
    y = int(np.searchsorted([5, 20, 40, 60, 80], yardline, side="left"))
    return (int(down) - 1) * 18 + d * 6 + y


def context_index(seconds, lead):
    time_bin = 0 if seconds > 120 else 1 if seconds > 30 else 2
    score_bin = 0 if lead < 0 else 1 if lead == 0 else 2
    return time_bin * 3 + score_bin


@dataclass(frozen=True)
class NFLEvent:
    game_id: str
    week: int
    half: int
    play_id: float
    next_play_id: float | None
    offense: str
    defense: str
    home_offense: bool
    source: int
    destination: int
    switch: bool
    own_points: int
    opponent_points: int
    elapsed: float
    remaining: float
    lead: int
    play_type: str
    raw_rows: int

    @property
    def key(self):
        return (self.source, self.destination, int(self.switch), self.own_points, self.opponent_points)

    @property
    def ctx(self):
        return context_index(self.remaining, self.lead)


@dataclass
class NFLExtractedGame:
    game_id: str
    week: int
    date: str
    home: str
    away: str
    events: list[NFLEvent]
    starts: list[dict]
    regulation: tuple[int, int]
    final: tuple[int, int]
    overtime: bool


def _is_decision(row):
    return (
        _present(row.get("down"))
        and row.get("play_type") in DECISIONS
        and row.get("extra_point_attempt", 0) != 1
        and row.get("two_point_attempt", 0) != 1
    )


def _replay(game):
    h = a = 0
    for opening in game.starts:
        h += opening["home_points"]
        a += opening["away_points"]
    for e in game.events:
        if e.home_offense:
            h += e.own_points
            a += e.opponent_points
        else:
            h += e.opponent_points
            a += e.own_points
    return int(h), int(a)


def extract_game(frame: pl.DataFrame) -> NFLExtractedGame:
    g = frame.sort("play_id")
    first = g.row(0, named=True)
    gid = str(first["game_id"])
    home = str(first["home_team"])
    away = str(first["away_team"])
    if home == away or not home or not away:
        raise ValueError("invalid team identity")

    rows = g.to_dicts()
    score_rows = []
    last = np.array([0.0, 0.0])
    for r in rows:
        if r.get("posteam") in (home, away) and all(_present(r.get(k)) for k in ("posteam_score_post", "defteam_score_post")):
            values = np.array([r["posteam_score_post"], r["defteam_score_post"]], float)
            last = values if r["posteam"] == home else values[::-1]
        score_rows.append(last.copy())
    scores = np.asarray(score_rows)
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores != scores.astype(int)):
        raise ValueError("invalid scoreboard")

    pre = np.vstack(([0, 0], scores[:-1]))
    final = (int(first["home_score"]), int(first["away_score"]))
    if tuple(scores[-1].astype(int)) != final:
        raise ValueError("final scoreboard mismatch")

    decisions = [i for i, r in enumerate(rows) if _is_decision(r)]
    for i, r in enumerate(rows):
        if _present(r.get("down")) and r.get("extra_point_attempt", 0) != 1 and r.get("two_point_attempt", 0) != 1 and i not in decisions and _present(r.get("play_type")):
            raise ValueError(f"unhandled down-bearing type {r['play_type']}")

    events = []
    starts = []
    previous_end = np.array([0, 0])
    for half in (1, 2):
        qtrs = [2 * half - 1, 2 * half]
        phase_idx = [i for i, r in enumerate(rows) if r.get("qtr") in qtrs]
        ix = [i for i in decisions if i in phase_idx]
        if not ix:
            raise ValueError("empty regulation half")
        end = scores[phase_idx[-1]].astype(int)
        r0 = rows[ix[0]]
        opening = pre[ix[0]].astype(int) - previous_end
        if np.any(opening < 0):
            raise ValueError("negative opening score")
        start_remaining = float(r0["half_seconds_remaining"])
        if not math.isfinite(start_remaining) or not 0 <= start_remaining <= 1800:
            raise ValueError("invalid initial clock")
        starts.append(
            dict(
                half=half,
                state=state_index(r0["down"], r0["ydstogo"], r0["yardline_100"]),
                home_offense=r0["posteam"] == home,
                remaining=start_remaining,
                home_points=int(opening[0]),
                away_points=int(opening[1]),
            )
        )
        for k, i in enumerate(ix):
            r = rows[i]
            j = ix[k + 1] if k + 1 < len(ix) else None
            nxt = rows[j] if j is not None else None
            reward = (pre[j] if j is not None else end) - pre[i]
            rem = float(r["half_seconds_remaining"])
            next_rem = float(nxt["half_seconds_remaining"]) if nxt is not None else 0.0
            if not all(math.isfinite(x) for x in (rem, next_rem)) or not 0 <= next_rem <= rem <= 1800:
                raise ValueError("nonmonotonic/missing half clock")
            if np.any(reward < 0):
                raise ValueError("negative segment score")
            home_off = r["posteam"] == home
            events.append(
                NFLEvent(
                    gid,
                    int(first["week"]),
                    half,
                    float(r["play_id"]),
                    float(nxt["play_id"]) if nxt is not None else None,
                    str(r["posteam"]),
                    away if home_off else home,
                    home_off,
                    state_index(r["down"], r["ydstogo"], r["yardline_100"]),
                    state_index(nxt["down"], nxt["ydstogo"], nxt["yardline_100"]) if nxt is not None else -1,
                    bool(nxt is not None and r["posteam"] != nxt["posteam"]),
                    int(reward[0 if home_off else 1]),
                    int(reward[1 if home_off else 0]),
                    rem - next_rem,
                    rem,
                    int(pre[i, 0] - pre[i, 1]) * (1 if home_off else -1),
                    str(r["play_type"]),
                    (j if j is not None else phase_idx[-1] + 1) - i,
                )
            )
        previous_end = end

    result = NFLExtractedGame(
        gid, int(first["week"]), str(first["game_date"]), home, away, events, starts,
        tuple(previous_end), final, bool(any(r.get("qtr", 0) > 4 for r in rows))
    )
    if _replay(result) != result.regulation:
        raise ValueError("regulation replay mismatch")
    if not result.overtime and result.regulation != result.final:
        raise ValueError("regulation/final mismatch")
    return result


class NFLGameExtractor(BaseExtractor):
    def extract(self, raw_df: pl.DataFrame, metadata: dict) -> ExtractedGame:
        nfl_game = extract_game(raw_df)
        return ExtractedGame(game_id=nfl_game.game_id, events=nfl_game.events)
