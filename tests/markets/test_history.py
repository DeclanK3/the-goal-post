import pandas as pd
import pytest

from goalpost.markets import history as m


def inputs():
    schedule = pd.DataFrame(
        [
            dict(
                game_id="old",
                season=2026,
                gameday="2026-10-04",
                game_type="REG",
                home_team="A",
                away_team="B",
                home_score=20,
                away_score=10,
            ),
            dict(
                game_id="future",
                season=2026,
                gameday="2026-10-11",
                game_type="REG",
                home_team="A",
                away_team="B",
                home_score=None,
                away_score=None,
            ),
        ]
    )
    meta = pd.DataFrame(
        [
            dict(
                game_id="old",
                game_date="2026-10-04",
                home_team_id="A",
                away_team_id="B",
                matrix_status="parsed_regulation",
                matrix_issue=None,
            )
        ]
    )
    return schedule, meta


def test_complete_prior_games_and_future_missing_normal():
    s, d = inputs()
    r = m.assess(s, d, "2026-10-08T16:00:00Z")
    assert r["status"] == "passed" and r["completed_games"] == 1


@pytest.mark.parametrize("kind", ["missing", "rejected", "orientation"])
def test_incomplete_history_blocks_with_denominators(kind):
    s, d = inputs()
    if kind == "missing":
        d = d.iloc[0:0]
    elif kind == "rejected":
        d.loc[0, "matrix_status"] = "rejected"
    else:
        d.loc[0, "home_team_id"] = "B"
    r = m.assess(s, d, "2026-10-08T16:00:00Z")
    assert r["status"] == "blocked"
    assert {e["team"] for e in r["errors"]} == {"A", "B"}


def test_old_simulation_cannot_be_relabelled_by_new_dataset():
    s, d = inputs()
    sim = {
        "results": [
            {"game": {"game_id": "future", "home": "A", "away": "B"}, "provenance": {"teams": {}}}
        ]
    }
    r = m.assess(s, d, "2026-10-08T16:00:00Z", sim)
    assert r["status"] == "blocked" and len(r["errors"]) == 2
    assert all(e["reason"] == "simulation_omits_completed_history" for e in r["errors"])


def test_read_failure_leaves_blocker_report(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(m.urllib.request, "urlopen", fail)
    with pytest.raises(ValueError, match="freshness blocked"):
        m.check(tmp_path, tmp_path / "check")
    assert "offline" in (tmp_path / "check/report.json").read_text()


def test_unresolved_past_date_is_unknown_not_graded():
    s, d = inputs()
    s.loc[0, "home_score"] = None
    r = m.assess(s, d, "2026-10-08T16:00:00Z")
    assert r["errors"][0]["reason"] == "prior_date_schedule_result_unresolved"


@pytest.mark.parametrize("now", ["2026-10-12T00:05:00Z", "2026-10-12T02:00:00Z"])
def test_eastern_evening_game_is_not_overdue_at_utc_midnight(now):
    s, d = inputs()
    s.loc[1, "gametime"] = "20:15"
    r = m.assess(s, d, now)
    assert r["status"] == "passed"
    assert r["unresolved_not_due"] == [
        dict(
            game_id="future",
            kickoff_utc="2026-10-12T00:15:00+00:00",
            reason="not_prior_utc_kickoff_date",
        )
    ]


@pytest.mark.parametrize("gametime", ["20:15", "invalid", None])
def test_older_or_unknown_kickoff_still_blocks(gametime):
    s, d = inputs()
    s.loc[1, "gametime"] = gametime
    r = m.assess(s, d, "2026-10-13T02:00:00Z")
    assert r["status"] == "blocked"
    assert r["errors"][0]["reason"] == "prior_date_schedule_result_unresolved"
