import json

import pandas as pd
import pytest

from goalpost.markets.export import WeeklyExport


def edit_report(path, action):
    report = json.loads((path / "report.json").read_text())
    action(report)
    (path / "report.json").write_text(json.dumps(report))


def test_load_typed_game_without_admission_flag_or_dataset_hash(export_path):
    exported = WeeklyExport.load(export_path)
    assert exported.games[0].home == "Cleveland Browns"
    assert exported.games[0].retained == 100


@pytest.mark.parametrize(
    "action",
    [
        lambda report: report.update(regulation_only=False),
        lambda report: report["results"].append(report["results"][0]),
        lambda report: report["results"][0]["game"].update(game_id="../outside"),
        lambda report: report["results"][0]["game"].update(home="PIT", away="CLE"),
        lambda report: report["results"][0].update(attempted=99),
        lambda report: report["results"][0].update(n=101),
        lambda report: report["results"][0]["diagnostics"].update(grace_completed_games=101),
    ],
)
def test_invalid_export_blocks(export_path, action):
    edit_report(export_path, action)
    with pytest.raises((ValueError, KeyError)):
        WeeklyExport.load(export_path)


def test_incorrect_provider_event_binding_blocks(export_path):
    schedule = pd.read_csv(export_path / "schedule.csv", dtype={"espn_event_id": str})
    schedule.loc[0, "home_team"] = "PIT"
    schedule.to_csv(export_path / "schedule.csv", index=False)
    with pytest.raises(ValueError, match="identity"):
        WeeklyExport.load(export_path)


def test_provider_orientation_change_blocks(export_path):
    path = export_path / "schedule-espn.json"
    schedule = json.loads(path.read_text())
    for team in schedule["events"][0]["competitions"][0]["competitors"]:
        team["homeAway"] = "away" if team["homeAway"] == "home" else "home"
    path.write_text(json.dumps(schedule))
    with pytest.raises(ValueError, match="identity"):
        WeeklyExport.load(export_path)


@pytest.mark.parametrize(
    "change", ["duplicate", "negative_total", "probability", "fractional_count", "nonfinite"]
)
def test_invalid_csv_mass(export_path, change):
    path = export_path / "2026_06_PIT_CLE/total-pmf.csv"
    frame = pd.read_csv(path)
    if change == "duplicate":
        frame.loc[1, "points"] = 30
    if change == "negative_total":
        frame.loc[0, "points"] = -1
    if change == "probability":
        frame.loc[0, "probability"] = 0.9
    if change == "fractional_count":
        frame["count"] = frame["count"].astype(float)
        frame.loc[0, "count"] = 20.5
    if change == "nonfinite":
        frame.loc[0, "probability"] = float("inf")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        WeeklyExport.load(export_path)


def test_failed_simulation_preserves_denominators(export_path):
    edit_report(
        export_path,
        lambda report: report["results"][0].update(status="blocked", n=0, diagnostics={}),
    )
    exported = WeeklyExport.load(export_path)
    assert not exported.games and exported.unavailable[0]["attempted"] == 120


def test_symlink_pmf_escape(export_path, tmp_path):
    path = export_path / "2026_06_PIT_CLE/total-pmf.csv"
    outside = tmp_path / "outside.csv"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        WeeklyExport.load(export_path)


def test_season_cannot_bypass_history_binding(export_path):
    edit_report(export_path, lambda report: report["results"][0]["game"].update(season=2025))
    with pytest.raises(ValueError, match="seasons differ"):
        WeeklyExport.load(export_path)
