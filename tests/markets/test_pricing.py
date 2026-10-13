import pandas as pd
import pytest

from goalpost.markets import comparison
from goalpost.markets import pricing as m


@pytest.mark.parametrize(
    "family,side,line,expected",
    [
        ("spreads", "H", -3, (0.3, 0.4, 0.3)),
        ("spreads", "A", 3, (0.3, 0.4, 0.3)),
        ("spreads", "H", 3.5, (1.0, 0.0, 0.0)),
        ("h2h", "H", None, (0.7, 0.0, 0.3)),
        ("totals", "Over", 3, (0.3, 0.4, 0.3)),
        ("totals", "Under", 3, (0.3, 0.4, 0.3)),
    ],
)
def test_orientation(family, side, line, expected):
    f = pd.DataFrame({"points": [-3, 3, 7], "probability": [0.3, 0.4, 0.3]})
    assert m.mass(f, family, side, line, "H", "A") == pytest.approx(expected)


def test_push_kelly_known_answer():
    r = m.sizing(0.5, 0.1, 0.4, 100)
    assert r["hypothetical_ev_pct"] == pytest.approx(10)
    assert r["full_kelly_pct"] == pytest.approx(100 / 9)
    assert r["quarter_kelly_pct"] == pytest.approx(25 / 9)
    assert m.sizing(0, 1, 0, -110)["full_kelly_pct"] == 0
    assert m.sizing(0.4, 0, 0.6, -110)["full_kelly_pct"] == 0


def test_binary_agrees_engine_formula():
    r = m.sizing(0.6, 0, 0.4, -110)
    assert r["quarter_kelly_pct"] == pytest.approx(
        ((0.6 * (100 / 110) - 0.4) / (100 / 110)) * 0.25 * 100
    )


def test_unknown_side_rejected():
    with pytest.raises(ValueError):
        m.mass(pd.DataFrame({"points": [1], "probability": [1]}), "spreads", "X", 1, "H", "A")


@pytest.mark.parametrize(
    "context,expected",
    [(None, "stale_quote"), ('{"home_team":"WRONG"}', "unpriced: Quote context mismatch")],
)
def test_compare_readonly_stale_quote_and_identity(tmp_path, context, expected):
    import json
    import sqlite3
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    kickoff = (now + timedelta(days=1)).isoformat()
    old = (now - timedelta(days=2)).isoformat()
    sim = tmp_path / "sim"
    sim.mkdir()
    gamefolder = sim / "g"
    gamefolder.mkdir()
    (sim / "report.json").write_text(
        json.dumps(
            {
                "regulation_only": True,
                "betting_admitted": False,
                "results": [
                    {
                        "game": {"game_id": "g", "home": "H", "away": "A", "kickoff_utc": kickoff},
                        "attempted": 100,
                        "n": 90,
                        "status": "simulated",
                        "diagnostics": {"grace_completed_games": 2},
                    }
                ],
            }
        )
    )
    (sim / "schedule-espn.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "competitions": [
                            {
                                "competitors": [
                                    {"team": {"abbreviation": "H", "displayName": "Home"}},
                                    {"team": {"abbreviation": "A", "displayName": "Away"}},
                                ]
                            }
                        ]
                    }
                ]
            }
        )
    )
    for name in ["total", "home_minus_away"]:
        pd.DataFrame({"points": [0, 7], "probability": [0.4, 0.6], "count": [36, 54]}).to_csv(
            gamefolder / (name + "-pmf.csv"), index=False
        )
    db = tmp_path / "quotes.sqlite"
    with sqlite3.connect(db) as c:
        c.executescript(
            "CREATE TABLE games(id TEXT,league TEXT,home_team TEXT,away_team TEXT,commence_time_utc TEXT,status TEXT); CREATE TABLE flags(key TEXT,value TEXT); CREATE TABLE markets(id INTEGER,game_id TEXT,market_type TEXT,participant TEXT,line REAL); CREATE TABLE odds_snapshots(id INTEGER,market_id INTEGER,book TEXT,source TEXT,odds_american INTEGER,taken_at TEXT,provider_updated_at TEXT,event_context_json TEXT);"
        )
        c.execute(
            "insert into games values(?,?,?,?,?,?)",
            ("e", "nfl", "Home", "Away", kickoff, "scheduled"),
        )
        c.execute("insert into markets values(1,'e','spreads','Home',-3.5)")
        c.execute(
            "insert into odds_snapshots values(1,1,?,?,?,?,?,?)",
            ("fanduel", None, -110, old, old, context),
        )
    with sqlite3.connect(db) as c:
        c.execute("insert into markets values(2,'e','spreads','Away',3.5)")
        c.execute(
            "insert into odds_snapshots values(2,2,?,?,?,?,?,?)",
            ("fanduel", None, -110, old, old, context),
        )
    before = db.read_bytes()
    comparison.compare(sim, tmp_path / "out", db)
    assert db.read_bytes() == before
    rows = pd.read_csv(tmp_path / "out/ev-kelly.csv")
    assert rows.iloc[0].quote_status == expected
    summary = json.loads((tmp_path / "out/report.json").read_text())
    assert summary["fresh_priced"] == 0
    if context is None:
        assert rows.iloc[0].p_win == pytest.approx(0.6)
        assert rows.iloc[0].incomplete == 10
        assert rows.iloc[0].identity_evidence == "current_database_identity_only"
        assert summary["actual_contract_ev_established"] is False


def test_latest_main_pair_drops_retired_lines_and_alternates():
    def q(side, line, at, kind="spreads"):
        return dict(participant=side, line=line, taken_at=at, market_type=kind)

    old = "2026-09-24T12:00:00Z"
    new = "2026-09-25T12:00:00Z"
    quotes = [
        q("H", -3.5, old),
        q("A", 3.5, old),
        q("H", -7, new),
        q("A", 7, new),
        q("H", -10, new, "alternate_spreads"),
    ]
    assert [r["line"] for r in comparison.latest_main_quotes(quotes, "H", "A")] == [-7, 7]
    assert comparison.latest_main_quotes(quotes[:-2], "H", "A") == []


@pytest.mark.parametrize("bad", ["mass", "counts", "nan", "duplicate", "negative_total"])
def test_corrupt_weekly_export_rejected(bad):
    f = pd.DataFrame({"points": [0, 7], "count": [40, 60], "probability": [0.4, 0.6]})
    if bad == "mass":
        f.loc[0, "probability"] = 0.5
    if bad == "counts":
        f.loc[0, "count"] = 41
    if bad == "nan":
        f.loc[0, "points"] = float("nan")
    if bad == "duplicate":
        f.loc[1, "points"] = 0
    if bad == "negative_total":
        f.loc[0, "points"] = -1
    with pytest.raises(ValueError):
        m.validate_pmfs({"total": f}, 100)


def test_rank_only_current_rows_no_fabricated_top_five():
    rows = [
        dict(
            game_id="g",
            market="spreads",
            participant="H",
            hypothetical_ev_pct=x,
            quote_status=status,
        )
        for x, status in [
            (80, "stale_quote"),
            (-2, "fresh_stored_quote"),
            (7, "fresh_stored_quote"),
        ]
    ]
    assert [r["hypothetical_ev_pct"] for r in m.ranked_current(rows)] == [7, -2]


def test_fresh_export_sorted_and_bound_to_input(tmp_path):
    import json
    import sqlite3
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    at = now.isoformat()
    kickoff = (now + timedelta(days=1)).isoformat()
    sim = tmp_path / "sim"
    (sim / "g").mkdir(parents=True)
    (sim / "report.json").write_text(
        json.dumps(
            {
                "regulation_only": True,
                "betting_admitted": False,
                "dataset_hashes": {"data": "test-hash"},
                "results": [
                    {
                        "game": {"game_id": "g", "home": "H", "away": "A", "kickoff_utc": kickoff},
                        "attempted": 100,
                        "n": 90,
                        "status": "simulated",
                    }
                ],
            }
        )
    )
    (sim / "schedule-espn.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "competitions": [
                            {
                                "competitors": [
                                    {"team": {"abbreviation": "H", "displayName": "Home"}},
                                    {"team": {"abbreviation": "A", "displayName": "Away"}},
                                ]
                            }
                        ]
                    }
                ]
            }
        )
    )
    for name in ["total", "home_minus_away"]:
        pd.DataFrame({"points": [0, 7], "count": [36, 54], "probability": [0.4, 0.6]}).to_csv(
            sim / "g" / (name + "-pmf.csv"), index=False
        )
    db = tmp_path / "db.sqlite"
    with sqlite3.connect(db) as c:
        c.executescript(
            "CREATE TABLE games(id TEXT,league TEXT,home_team TEXT,away_team TEXT,commence_time_utc TEXT,status TEXT); CREATE TABLE flags(key TEXT,value TEXT); CREATE TABLE markets(id INTEGER,game_id TEXT,market_type TEXT,participant TEXT,line REAL); CREATE TABLE odds_snapshots(id INTEGER,market_id INTEGER,book TEXT,source TEXT,odds_american INTEGER,taken_at TEXT,provider_updated_at TEXT,event_context_json TEXT);"
        )
        c.execute(
            "insert into games values(?,?,?,?,?,?)",
            ("e", "nfl", "Home", "Away", kickoff, "scheduled"),
        )
        for i, side, line in [(1, "Home", -3.5), (2, "Away", 3.5)]:
            c.execute("insert into markets values(?,?,?,?,?)", (i, "e", "spreads", side, line))
            c.execute(
                "insert into odds_snapshots values(?,?,?,?,?,?,?,?)",
                (i, i, "fanduel", None, -110, at, at, None),
            )
    before = db.read_bytes()
    comparison.compare(sim, tmp_path / "out", db, bankroll=1000)
    assert db.read_bytes() == before
    rows = json.load(open(tmp_path / "out/current-market-sizing.json"))
    r = rows[0]
    assert r["participant"] == "Home" and r["p_win"] == pytest.approx(0.6)
    assert r["p_win_all_attempts_lower"] == pytest.approx(0.54)
    assert r["p_win_all_attempts_upper"] == pytest.approx(0.64)
    assert r["hypothetical_quarter_kelly_dollars"] == pytest.approx(
        1000 * r["quarter_kelly_pct"] / 100
    )
    assert len(r["pmf_sha256"]) == 64 and r["actual_contract_ev"] is None
    report = json.load(open(tmp_path / "out/report.json"))
    assert report["simulation_dataset_hashes"] == {"data": "test-hash"}
    assert report["fresh_priced"] == 2
