import io
import json
import sqlite3
from contextlib import closing
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from goalpost.data.nfl import odds as module
from goalpost.data.nfl.odds import OddsAPISource, QuoteSource, QuoteSourceError, SQLiteQuoteSource


def payload():
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


def response(body, headers):
    stream = io.BytesIO(json.dumps(body).encode())
    stream.headers = headers
    return stream


def test_source_is_abstract():
    with pytest.raises(TypeError):
        QuoteSource()


def test_no_key_means_no_request(monkeypatch):
    fetch = Mock()
    monkeypatch.setattr(module, "urlopen", fetch)
    with pytest.raises(QuoteSourceError, match="ODDS_API"):
        OddsAPISource().load()
    fetch.assert_not_called()


def test_live_fetch_uses_env_and_reports_known_credits(monkeypatch):
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


def test_no_credit_headers_means_unknown_not_zero(monkeypatch):
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    monkeypatch.setattr(module, "urlopen", Mock(return_value=response(payload(), {})))
    snapshot = OddsAPISource().load()
    assert snapshot.credits_last_request is None and snapshot.credits_remaining is None


@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("https://api.example/?apiKey=unit-test-only", 401, "unit-test-only", {}, None),
        URLError("secret URL apiKey=unit-test-only"),
        TimeoutError("unit-test-only"),
    ],
)
def test_failure_redacts_key_and_never_retries(monkeypatch, failure):
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    fetch = Mock(side_effect=failure)
    monkeypatch.setattr(module, "urlopen", fetch)
    with pytest.raises(QuoteSourceError) as error:
        OddsAPISource().load()
    assert "unit-test-only" not in str(error.value) and fetch.call_count == 1


def test_bad_response_does_not_echo_provider_body(monkeypatch):
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    monkeypatch.setattr(
        module, "urlopen", Mock(return_value=response({"error": "unit-test-only"}, {}))
    )
    with pytest.raises(QuoteSourceError) as error:
        OddsAPISource().load()
    assert "unit-test-only" not in str(error.value)


def test_missing_book_stays_missing(now):
    body = payload()
    body[0]["bookmakers"][0]["key"] = "anotherbook"
    assert not OddsAPISource().parse(body, now)[0].quotes


def test_market_timestamp_takes_precedence(now):
    body = payload()
    body[0]["bookmakers"][0]["markets"][0]["last_update"] = "2026-10-12T00:00:00Z"
    quote = OddsAPISource().parse(body, now)[0].quotes[0]
    assert quote.updated_at.day == 12


def test_local_database_disabled_by_default(tmp_path):
    path = tmp_path / "absent.db"
    with pytest.raises(QuoteSourceError, match="GOALPOST_LOCAL_DB_ENABLED"):
        SQLiteQuoteSource(path).load()
    assert not path.exists()


def test_local_adapter_never_creates_missing_database(tmp_path, monkeypatch):
    monkeypatch.setenv("GOALPOST_LOCAL_DB_ENABLED", "true")
    path = tmp_path / "absent.db"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteQuoteSource(path).load()
    assert not path.exists()


def test_existing_store_readonly_latest_record_and_context(tmp_path, monkeypatch):
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


def test_duplicate_api_event_ids_rejected(now):
    body = payload()
    with pytest.raises(ValueError, match="Duplicate"):
        OddsAPISource().parse(body + body, now)
