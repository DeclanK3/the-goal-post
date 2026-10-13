"""Failure boundaries and end-to-end stored-quote comparison, all offline."""

import copy
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from goalpost.markets import bridge, comparison, pricing


@pytest.fixture
def export(tmp_path):
    path = tmp_path / "export"
    shutil.copytree(bridge.HERE / "example/export", path)
    return path


def change_report(export, edit):
    path = export / "report.json"
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r["results"][0]["game"].update(home="A", away="H"),
        lambda r: r["results"][0]["game"].update(kickoff_utc="2000-01-03T18:00:00Z"),
        lambda r: r["results"][0]["diagnostics"].update(grace_completed_games=101),
    ],
)
def test_invalid_schedule_binding_or_approximate_count(export, edit):
    change_report(export, edit)
    with pytest.raises(ValueError):
        bridge.validate_export(export)


def test_duplicate_schedule_identity_rejected(export):
    path = export / "schedule-espn.json"
    value = json.loads(path.read_text())
    value["events"].append(copy.deepcopy(value["events"][0]))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ambiguous"):
        bridge.validate_export(export)


def test_empty_slate_validates_without_simulation_claim(export):
    change_report(export, lambda r: r.update(results=[]))
    validation, _ = bridge.validate_export(export)
    assert validation["games"] == []


def test_blocked_simulation_remains_in_denominator(export):
    change_report(export, lambda r: r["results"][0].update(status="blocked", n=0, diagnostics={}))
    shutil.rmtree(export / "DEMO_AWAY_HOME")
    validation, _ = bridge.validate_export(export)
    assert validation["games"][0]["retained"] == 0
    assert validation["games"][0]["attempted"] == 120
    assert validation["games"][0]["status"] == "blocked"


def test_cli_failure_retained_and_existing_output_not_modified(export, tmp_path):
    change_report(export, lambda r: r.update(regulation_only=False))
    out = tmp_path / "failure"
    with pytest.raises(ValueError):
        bridge.main(["--export", str(export), "--validate-only", "--output", str(out)])
    failure = json.loads((out / "failure.json").read_text())
    assert failure["old_report_substituted"] is False
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError):
        bridge.main(["--demo", "--bankroll", "-1", "--output", str(out)])
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_read_failure_cannot_create_or_replace_database(export, tmp_path):
    database = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        comparison.compare(export, tmp_path / "out", database)
    assert not database.exists()
    assert not (tmp_path / "out/current-market-sizing.csv").exists()


def test_live_path_binds_dataset_and_calls_history_before_comparison(export, tmp_path, monkeypatch):
    # Use actual hashes and the actual read-only comparator; freeze only clock
    # and public schedule validation, never the pricing functions.
    data = tmp_path / "data"
    data.mkdir()
    names = ("transition_set.json", "nfl_games.parquet", "parsed_segments.parquet")
    for name in names:
        (data / name).write_text("fixture_" + name)
    change_report(
        export,
        lambda r: r.update(
            synthetic_demo=False,
            dataset_hashes={name: bridge.digest(data / name) for name in names},
        ),
    )
    db = tmp_path / "quotes.sqlite"
    bridge.demo_database(db)
    before = bridge.digest(db)
    seen = []

    def check(d, out, report):
        seen.append((d, report["dataset_hashes"]))
        return dict(status="passed", fixture=True)

    monkeypatch.setattr(bridge.history, "check", check)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(comparison, "datetime", Clock)
    result = bridge.run(export, tmp_path / "out", data=data, database=db)
    assert len(seen) == 1 and seen[0][0] == data
    assert bridge.digest(db) == before
    assert result["paid_requests"] == 0 and result["simulation_run"] is False
    report = json.loads((tmp_path / "out/comparison/report.json").read_text())
    assert report["fresh_priced"] == 6
    assert report["export_dataset_binding"] == "all_three_dataset_hashes_match"
    assert report["calibration_verified"] is False


@pytest.mark.parametrize(
    "kind",
    ["stale_provider", "missing_provider", "ambiguous_event", "already_started", "latest_unpaired"],
)
def test_quote_exclusions_no_old_line_fallback(export, tmp_path, kind):
    db = tmp_path / "quotes.sqlite"
    bridge.demo_database(db)
    with sqlite3.connect(db) as conn:
        if kind == "stale_provider":
            conn.execute("UPDATE odds_snapshots SET provider_updated_at='1999-12-31T12:00:00Z'")
        elif kind == "missing_provider":
            conn.execute("UPDATE odds_snapshots SET provider_updated_at=NULL")
        elif kind == "ambiguous_event":
            conn.execute(
                "INSERT INTO games SELECT 'duplicate',league,home_team,away_team,commence_time_utc,status FROM games"
            )
        elif kind == "already_started":
            conn.execute("UPDATE games SET status='in_progress'")
        else:
            conn.execute(
                "UPDATE odds_snapshots SET taken_at='2000-01-01T11:59:30Z' WHERE market_id IN (SELECT id FROM markets WHERE participant IN ('Example Home','Over'))"
            )
    before = bridge.digest(db)
    comparison.compare(
        export, tmp_path / "out", db, now=datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    )
    assert bridge.digest(db) == before
    assert json.loads((tmp_path / "out/current-market-sizing.json").read_text()) == []
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["games"] == 1
    if kind == "latest_unpaired":
        assert report["quote_exclusions"]["latest_main_pair_incomplete_or_invalid"] == 3
        assert report["quote_exclusions"]["superseded_main_line"] == 3
    elif kind == "stale_provider":
        assert report["quote_status_counts"] == {"stale_quote": 6}


def test_nonfinite_contract_is_rejected():
    frame = pd.DataFrame({"points": [7], "probability": [1]})
    with pytest.raises(ValueError):
        pricing.mass(frame, "spreads", "H", float("nan"), "H", "A")


def test_all_demo_rows_match_existing_handoff_arithmetic(tmp_path):
    result = bridge.run(None, tmp_path / "demo", demo=True)
    actual = json.loads(Path(result["table"]).with_suffix(".json").read_text())
    expected = json.loads((Path(__file__).parent / "fixtures/demo-expected.json").read_text())
    assert len(actual) == len(expected) == 6
    for a, e in zip(actual, expected):
        for key, value in e.items():
            assert a[key] == (pytest.approx(value) if isinstance(value, (int, float)) else value)
