"""Offline fixtures only; no installed demo, data generator or network access."""

import json
import socket
from datetime import datetime, timezone

import pandas as pd
import pytest

from goalpost.data.nfl.espn_export import normalize
from goalpost.data.nfl.odds import BookEvent, Quote
from goalpost.simulation.nfl.contracts import NFLGame, ScoreDistribution


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.delenv("ODDS_API", raising=False)
    monkeypatch.delenv("GOALPOST_LOCAL_DB_ENABLED", raising=False)


@pytest.fixture
def now():
    return datetime(2026, 10, 13, 12, tzinfo=timezone.utc)


@pytest.fixture
def game():
    return NFLGame(
        "2026_06_PIT_CLE",
        "Cleveland Browns",
        "Pittsburgh Steelers",
        datetime(2026, 10, 15, 0, tzinfo=timezone.utc),
        ScoreDistribution((30, 37, 38, 44), (0.2, 0.2, 0.3, 0.3)),
        ScoreDistribution((-7, 0, 3, 7), (0.2, 0.1, 0.3, 0.4)),
        120,
        100,
        2,
    )


@pytest.fixture
def quotes(game, now):
    return [
        Quote("home", "h2h", game.home, None, -200, now, now, "live_response_context"),
        Quote("away", "h2h", game.away, None, 170, now, now, "live_response_context"),
    ]


@pytest.fixture
def event(game, quotes):
    return BookEvent("provider-id", game.home, game.away, game.kickoff, tuple(quotes))


@pytest.fixture
def export_path(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    event = {
        "id": "401000001",
        "date": "2026-10-15T00:00:00Z",
        "season": {"year": 2026, "type": 2},
        "week": {"number": 6},
        "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre", "completed": False}},
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"abbreviation": "CLE", "displayName": "Cleveland Browns"},
                    },
                    {
                        "homeAway": "away",
                        "team": {"abbreviation": "PIT", "displayName": "Pittsburgh Steelers"},
                    },
                ]
            }
        ],
    }
    schedule = {"events": [event]}
    normalize(schedule).to_csv(root / "schedule.csv", index=False)
    (root / "schedule-espn.json").write_text(json.dumps(schedule))
    result = {
        "game": {
            "game_id": "2026_06_PIT_CLE",
            "home": "CLE",
            "away": "PIT",
            "season": 2026,
            "kickoff_utc": event["date"],
        },
        "status": "simulated",
        "attempted": 120,
        "n": 100,
        "diagnostics": {"grace_completed_games": 2},
        "provenance": {
            "teams": {
                "CLE": {"games": [{"game_id": "prior"}]},
                "PIT": {"games": [{"game_id": "prior"}]},
            }
        },
    }
    (root / "report.json").write_text(
        json.dumps({"status": "complete", "regulation_only": True, "results": [result]})
    )
    folder = root / result["game"]["game_id"]
    folder.mkdir()
    pd.DataFrame(
        {"points": [30, 37, 38, 44], "count": [20, 20, 30, 30], "probability": [0.2, 0.2, 0.3, 0.3]}
    ).to_csv(folder / "total-pmf.csv", index=False)
    pd.DataFrame(
        {"points": [-7, 0, 3, 7], "count": [20, 10, 30, 40], "probability": [0.2, 0.1, 0.3, 0.4]}
    ).to_csv(folder / "home_minus_away-pmf.csv", index=False)
    return root
