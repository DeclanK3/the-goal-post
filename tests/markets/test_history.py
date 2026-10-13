import json
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from goalpost.markets import history as module
from goalpost.markets.history import NFLHistory


def schedule():
    return pd.DataFrame(
        [
            {
                "game_id": "prior",
                "season": 2026,
                "gameday": "2026-10-11",
                "game_type": "REG",
                "home_team": "CLE",
                "away_team": "PIT",
                "home_score": 20,
                "away_score": 10,
            }
        ]
    )


def test_uses_nflreadpy_loader(export_path, now, monkeypatch):
    report = json.loads((export_path / "report.json").read_text())
    loader = Mock(return_value=pl.from_pandas(schedule()))
    monkeypatch.setattr(module.nflreadpy, "load_schedules", loader)
    checked = NFLHistory().check(report, now)
    assert checked["status"] == "passed" and checked["completed_games"] == 1
    loader.assert_called_once_with(seasons=[2026])


def test_old_simulation_donors_block_even_with_current_schedule(export_path, now):
    report = json.loads((export_path / "report.json").read_text())
    report["results"][0]["provenance"]["teams"] = {}
    checked = NFLHistory().assess(schedule(), report["results"], now)
    assert checked["status"] == "blocked"
    assert {error["team"] for error in checked["errors"]} == {"CLE", "PIT"}


def test_unresolved_old_result_is_unknown_not_regraded(export_path, now):
    report = json.loads((export_path / "report.json").read_text())
    frame = schedule()
    frame.loc[0, "home_score"] = None
    checked = NFLHistory().assess(frame, report["results"], now)
    assert checked["errors"][0]["reason"] == "prior_date_result_unknown"


@pytest.mark.parametrize("instant", ["2026-10-12T00:05:00Z", "2026-10-12T02:00:00Z"])
def test_eastern_night_game_not_overdue_at_utc_midnight(instant):
    frame = schedule()
    frame.loc[0, "home_score"] = None
    frame.loc[0, "gametime"] = "20:15"
    checked = NFLHistory().assess(frame, [], pd.Timestamp(instant).to_pydatetime())
    assert checked["status"] == "passed" and checked["unresolved_not_due"] == ["prior"]


def test_no_upcoming_games_needs_no_history_fetch(now, monkeypatch):
    loader = Mock()
    monkeypatch.setattr(module.nflreadpy, "load_schedules", loader)
    assert NFLHistory().check({"results": []}, now)["status"] == "not_required"
    loader.assert_not_called()


def test_duplicate_schedule_rejected(export_path, now):
    report = json.loads((export_path / "report.json").read_text())
    with pytest.raises(ValueError, match="identity"):
        NFLHistory().assess(pd.concat([schedule(), schedule()]), report["results"], now)


def test_outdated_source_rejected(export_path, now):
    frame = schedule()
    frame.loc[0, "gameday"] = "2025-01-01"
    with pytest.raises(ValueError, match="recent"):
        NFLHistory().assess(frame, [], now)
