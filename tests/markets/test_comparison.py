from dataclasses import replace
from datetime import timedelta

import pytest

from goalpost.markets.comparison import ComparisonSettings, NFLComparison
from goalpost.markets.quotes import Quote, QuoteSnapshot


def run(game, event, now, **settings):
    return NFLComparison(ComparisonSettings(**settings)).run(
        [game], QuoteSnapshot((event,), "unit_test"), now
    )


def test_fresh_sorted_table_preserves_scope_and_missing_paths(game, event, now):
    result = run(game, event, now)
    home = result["current"][0]
    assert home["participant"] == game.home and home["hypothetical_ev_pct"] == pytest.approx(15)
    assert home["p_win_all_attempts_lower"] == pytest.approx(0.7 * 100 / 120)
    assert home["actual_contract_ev"] is None and home["actual_contract_kelly"] is None
    assert home["hypothetical_quarter_kelly_dollars"] is None
    assert result["matchups"][0]["incomplete"] == 20
    assert result["matchups"][0]["missing_main_families"] == ["spreads", "totals"]


def test_explicit_bankroll_only(game, event, now):
    home = run(game, event, now, bankroll=120)["current"][0]
    assert home["hypothetical_quarter_kelly_dollars"] == pytest.approx(10)


@pytest.mark.parametrize("field", ["captured_at", "updated_at"])
def test_stale_capture_or_provider_excludes_current_but_retains_arithmetic(game, event, now, field):
    event = replace(
        event,
        quotes=tuple(replace(q, **{field: now - timedelta(minutes=31)}) for q in event.quotes),
    )
    result = run(game, event, now)
    assert not result["current"] and len(result["rows"]) == 2
    assert all(row["quote_status"] == "stale_quote" for row in result["rows"])


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"updated_at": None}, "missing_or_invalid_quote_timestamp"),
        ({"context_matches": False}, "quote_context_mismatch"),
        ({"odds": 99}, "invalid_contract_or_price"),
    ],
)
def test_invalid_quotes_preserved_unpriced(game, event, now, changes, reason):
    event = replace(event, quotes=tuple(replace(q, **changes) for q in event.quotes))
    result = run(game, event, now)
    assert not result["current"] and len(result["rows"]) == 2
    assert all(row["reason"] == reason for row in result["rows"])


def test_future_quote_excluded(game, event, now):
    event = replace(
        event, quotes=tuple(replace(q, updated_at=now + timedelta(minutes=2)) for q in event.quotes)
    )
    assert {r["reason"] for r in run(game, event, now)["rows"]} == {"future_quote_timestamp"}


def test_incomplete_newest_pair_never_falls_back(game, event, now):
    newest = replace(event.quotes[0], reference="newest", captured_at=now + timedelta(seconds=1))
    event = replace(event, quotes=event.quotes + (newest,))
    result = run(game, event, now)
    assert not result["rows"]
    assert {r["disposition"] for r in result["quote_evidence"]} == {
        "superseded_main_line",
        "latest_main_pair_incomplete_or_invalid",
    }


def test_unorderable_capture_does_not_fall_back(game, event, now):
    event = replace(event, quotes=event.quotes + (replace(event.quotes[0], captured_at=None),))
    result = run(game, event, now)
    assert not result["rows"]
    assert all(
        r["disposition"] == "unorderable_capture_timestamp" for r in result["quote_evidence"]
    )


@pytest.mark.parametrize("family,lines", [("spreads", (-3, 3)), ("totals", (37, 37))])
def test_exact_push_lines(game, event, now, family, lines):
    names = (game.home, game.away) if family == "spreads" else ("Over", "Under")
    event = replace(
        event,
        quotes=tuple(
            Quote(str(i), family, name, line, -110, now, now, "live_response_context")
            for i, (name, line) in enumerate(zip(names, lines))
        ),
    )
    result = run(game, event, now)
    assert len(result["rows"]) == 2
    assert all(r["p_push_or_regulation_tie"] > 0 for r in result["rows"])


def test_alternates_and_nonopposing_lines_excluded(game, event, now):
    quotes = (
        Quote(
            "a",
            "alternate_spreads",
            game.home,
            -3,
            -110,
            now,
            now,
            "current_database_identity_only",
        ),
        Quote("b", "spreads", game.home, -3, -110, now, now, "current_database_identity_only"),
        Quote("c", "spreads", game.away, -3, -110, now, now, "current_database_identity_only"),
    )
    result = run(game, replace(event, quotes=quotes), now)
    assert not result["rows"] and len(result["quote_evidence"]) == 3


def test_duplicate_events_are_ambiguous(game, event, now):
    result = NFLComparison(ComparisonSettings()).run(
        [game], QuoteSnapshot((event, event), "unit_test"), now
    )
    assert (
        not result["current"]
        and result["matchups"][0]["quote_status"] == "missing_or_ambiguous_event"
    )


@pytest.mark.parametrize("change", ["reversed", "kickoff", "started"])
def test_ineligible_events(game, event, now, change):
    if change == "reversed":
        event = replace(event, home=event.away, away=event.home)
    if change == "kickoff":
        event = replace(event, kickoff=event.kickoff + timedelta(hours=1))
    if change == "started":
        event = replace(event, scheduled=False)
    assert not run(game, event, now)["current"]


@pytest.mark.parametrize(
    "settings",
    [
        {"book": "fanduel,other"},
        {"max_quote_age_seconds": 0},
        {"bankroll": float("nan")},
        {"kickoff_tolerance_seconds": -1},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        ComparisonSettings(**settings)
