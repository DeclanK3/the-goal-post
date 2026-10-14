# Purpose: Check donor coverage without inferring results or fetching live schedules.
# Public API: pytest tests and local fixtures in this module.
# Consumers: The offline unit suite; temporary files never access production stores.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from goalpost.data.nfl import history as module
from goalpost.data.nfl.history import NFLHistory


# Purpose: Provide one completed synthetic donor game with stable team orientation.
# Callers: History assessment regressions.
def schedule() -> pd.DataFrame:
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


# Purpose: Verify uses nflreadpy loader.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_uses_nflreadpy_loader(
    export_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = json.loads((export_path / "report.json").read_text())
    loader = Mock(return_value=pl.from_pandas(schedule()))
    monkeypatch.setattr(module.nflreadpy, "load_schedules", loader)
    checked = NFLHistory().check(report, now)
    assert checked["status"] == "passed" and checked["completed_games"] == 1
    loader.assert_called_once_with(seasons=[2026])


# Purpose: Verify old simulation donors block even with current schedule.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_old_simulation_donors_block_even_with_current_schedule(
    export_path: Path, now: datetime
) -> None:
    report = json.loads((export_path / "report.json").read_text())
    report["results"][0]["provenance"]["teams"] = {}
    checked = NFLHistory().assess(schedule(), report["results"], now)
    assert checked["status"] == "blocked"
    assert {error["team"] for error in checked["errors"]} == {"CLE", "PIT"}


# Purpose: Verify unresolved old result is unknown not regraded.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_unresolved_old_result_is_unknown_not_regraded(export_path: Path, now: datetime) -> None:
    report = json.loads((export_path / "report.json").read_text())
    frame = schedule()
    frame.loc[0, "home_score"] = None
    checked = NFLHistory().assess(frame, report["results"], now)
    assert checked["errors"][0]["reason"] == "prior_date_result_unknown"


# Purpose: Verify eastern night game not overdue at utc midnight.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize("instant", ["2026-10-12T00:05:00Z", "2026-10-12T02:00:00Z"])
def test_eastern_night_game_not_overdue_at_utc_midnight(instant: str) -> None:
    frame = schedule()
    frame.loc[0, "home_score"] = None
    frame.loc[0, "gametime"] = "20:15"
    checked = NFLHistory().assess(frame, [], pd.Timestamp(instant).to_pydatetime())
    assert checked["status"] == "passed" and checked["unresolved_not_due"] == ["prior"]


# Purpose: Verify no upcoming games needs no history fetch.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_no_upcoming_games_needs_no_history_fetch(
    now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = Mock()
    monkeypatch.setattr(module.nflreadpy, "load_schedules", loader)
    assert NFLHistory().check({"results": []}, now)["status"] == "not_required"
    loader.assert_not_called()


# Purpose: Verify duplicate schedule rejected.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_duplicate_schedule_rejected(export_path: Path, now: datetime) -> None:
    report = json.loads((export_path / "report.json").read_text())
    with pytest.raises(ValueError, match="identity"):
        NFLHistory().assess(pd.concat([schedule(), schedule()]), report["results"], now)


# Purpose: Verify outdated source rejected.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_outdated_source_rejected(export_path: Path, now: datetime) -> None:
    frame = schedule()
    frame.loc[0, "gameday"] = "2025-01-01"
    with pytest.raises(ValueError, match="recent"):
        NFLHistory().assess(frame, [], now)
