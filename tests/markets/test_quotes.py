# Purpose: Verify source opt-ins, request accounting, redaction and read-only storage.
# evidence.
# Public API: pytest tests and local fixtures in this module.
# Consumers: The offline unit suite; temporary files never access production stores.

from __future__ import annotations

import io
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from goalpost.data.nfl import odds as module
from goalpost.data.nfl.odds import OddsAPISource, QuoteSource, QuoteSourceError, SQLiteQuoteSource


# Purpose: Build a minimal synthetic provider response with a two-sided main market.
# Callers: Quote adapter tests; contains no real event or account credentials.
def payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "event",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2030-10-15T00:00:00Z",
            "home_team": "Cleveland Browns",
            "away_team": "Pittsburgh Steelers",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "last_update": "2030-10-13T12:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Cleveland Browns", "price": -200},
                                {"name": "Pittsburgh Steelers", "price": 170},
                            ],
                        }
                    ],
                }
            ],
        }
    ]


# Purpose: Wrap synthetic JSON and headers in a mocked HTTP response stream.
# Callers: Quote-source tests that patch urlopen.
def response(body: Any, headers: dict[str, str]) -> io.BytesIO:
    stream = io.BytesIO(json.dumps(body).encode())
    stream.headers = headers
    return stream


# Purpose: Verify source is abstract.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_source_is_abstract() -> None:
    with pytest.raises(TypeError):
        QuoteSource()


# Purpose: Verify no key means no request.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_no_key_means_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = Mock()
    monkeypatch.setattr(module, "urlopen", fetch)
    with pytest.raises(QuoteSourceError, match="ODDS_API"):
        OddsAPISource().load()
    fetch.assert_not_called()


# Purpose: Verify live fetch uses env and reports known credits.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_live_fetch_uses_env_and_reports_known_credits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    fetch = Mock(
        return_value=response(
            payload(),
            {"x-requests-used": "90", "x-requests-remaining": "910", "x-requests-last": "3"},
        )
    )
    monkeypatch.setattr(module, "urlopen", fetch)
    source = OddsAPISource()
    snapshot = source.load()
    assert snapshot.requests == 1 and snapshot.credits_last_request == 3
    assert snapshot.credits_used == 90 and snapshot.credits_remaining == 910
    assert len(snapshot.events[0].quotes) == 2
    args = parse_qs(urlparse(fetch.call_args.args[0]).query)
    assert args["apiKey"] == ["unit-test-only"] and args["markets"] == ["h2h,spreads,totals"]
    assert args["bookmakers"] == ["fanduel"] and fetch.call_count == 1
    assert "unit-test-only" not in repr(source) + repr(snapshot)


# Purpose: Verify no credit headers means unknown not zero.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_no_credit_headers_means_unknown_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    monkeypatch.setattr(module, "urlopen", Mock(return_value=response(payload(), {})))
    snapshot = OddsAPISource().load()
    assert snapshot.credits_last_request is None and snapshot.credits_remaining is None


# Purpose: Verify failure redacts key and never retries.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("https://api.example/?apiKey=unit-test-only", 401, "unit-test-only", {}, None),
        URLError("secret URL apiKey=unit-test-only"),
        TimeoutError("unit-test-only"),
    ],
)
def test_failure_redacts_key_and_never_retries(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    fetch = Mock(side_effect=failure)
    monkeypatch.setattr(module, "urlopen", fetch)
    with pytest.raises(QuoteSourceError) as error:
        OddsAPISource().load()
    assert "unit-test-only" not in str(error.value) and fetch.call_count == 1


# Purpose: Verify bad response does not echo provider body.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_bad_response_does_not_echo_provider_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    monkeypatch.setattr(
        module, "urlopen", Mock(return_value=response({"error": "unit-test-only"}, {}))
    )
    with pytest.raises(QuoteSourceError) as error:
        OddsAPISource().load()
    assert "unit-test-only" not in str(error.value)


# Purpose: Verify missing book stays missing.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_missing_book_stays_missing(now: datetime) -> None:
    body = payload()
    body[0]["bookmakers"][0]["key"] = "anotherbook"
    assert not OddsAPISource().parse(body, now)[0].quotes


# Purpose: Verify market timestamp takes precedence.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_market_timestamp_takes_precedence(now: datetime) -> None:
    body = payload()
    body[0]["bookmakers"][0]["markets"][0]["last_update"] = "2026-10-12T00:00:00Z"
    quote = OddsAPISource().parse(body, now)[0].quotes[0]
    assert quote.updated_at.day == 12


# Purpose: Verify local database disabled by default.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_local_database_disabled_by_default(tmp_path: Path) -> None:
    path = tmp_path / "absent.db"
    with pytest.raises(QuoteSourceError, match="GOALPOST_LOCAL_DB_ENABLED"):
        SQLiteQuoteSource(path).load()
    assert not path.exists()


# Purpose: Verify local adapter never creates missing database.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_local_adapter_never_creates_missing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALPOST_LOCAL_DB_ENABLED", "true")
    path = tmp_path / "absent.db"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteQuoteSource(path).load()
    assert not path.exists()


# Purpose: Verify existing store readonly latest record and context.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_existing_store_readonly_latest_record_and_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALPOST_LOCAL_DB_ENABLED", "true")
    path = tmp_path / "quotes.db"
    with closing(sqlite3.connect(path)) as db:
        db.executescript("""
            CREATE TABLE games(id TEXT,league TEXT,home_team TEXT,away_team TEXT,
                               commence_time_utc TEXT,status TEXT);
            CREATE TABLE markets(id INTEGER,game_id TEXT,market_type TEXT,
                                 participant TEXT,line REAL);
            CREATE TABLE odds_snapshots(id INTEGER,market_id INTEGER,book TEXT,source TEXT,
                odds_american INTEGER,taken_at TEXT,provider_updated_at TEXT,
                event_context_json TEXT);
            INSERT INTO games VALUES('event','nfl','Cleveland Browns','Pittsburgh Steelers',
                                     '2026-10-15T00:00:00Z','scheduled');
            INSERT INTO markets VALUES(1,'event','h2h','Cleveland Browns',NULL);
            INSERT INTO odds_snapshots VALUES(1,1,'fanduel',NULL,-200,
                '2026-10-13T11:59:00Z','2026-10-13T11:59:00Z',NULL);
            INSERT INTO odds_snapshots VALUES(2,1,'fanduel',NULL,-210,
                '2026-10-13T12:00:00Z','2026-10-13T12:00:00Z','{"home_team":"wrong"}');
        """)
        db.commit()
    before = path.read_bytes()
    snapshot = SQLiteQuoteSource(path).load()
    assert snapshot.events[0].quotes[0].odds == -210
    assert not snapshot.events[0].quotes[0].context_matches
    assert snapshot.requests == 0 and path.read_bytes() == before


# Purpose: Verify duplicate api event ids rejected.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_duplicate_api_event_ids_rejected(now: datetime) -> None:
    body = payload()
    with pytest.raises(ValueError, match="Duplicate"):
        OddsAPISource().parse(body + body, now)
