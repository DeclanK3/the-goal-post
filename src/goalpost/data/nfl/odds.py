"""NFL live and read-only local odds adapters; requests are explicit."""

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from goalpost.data.base.odds import (
    BookEvent,
    Quote,
    QuoteSnapshot,
    QuoteSource,
    QuoteSourceError,
    optional_timestamp,
    timestamp,
    validate_book,
)

MAIN_MARKETS = ("h2h", "spreads", "totals")


@dataclass
class OddsAPISource(QuoteSource):
    """One NFL odds request per load, requiring ODDS_API; no retries or scheduler."""

    book: str = "fanduel"
    timeout_seconds: float = 30

    def load(self) -> QuoteSnapshot:
        validate_book(self.book)
        key = os.environ.get("ODDS_API", "").strip()
        if not key:
            raise QuoteSourceError("Live odds require the ODDS_API environment variable")
        query = urlencode(
            {
                "apiKey": key,
                "bookmakers": self.book,
                "markets": ",".join(MAIN_MARKETS),
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
        )
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?" + query
        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                raw = response.read(10 * 1024 * 1024 + 1)
                headers = response.headers
            captured = datetime.now(timezone.utc)
        except HTTPError as error:
            # HTTP/transport exceptions may embed the full URL, including apiKey.
            raise QuoteSourceError(f"Odds API HTTP {error.code}; not retried") from None
        except Exception:
            raise QuoteSourceError("Odds API request failed; charge unknown; not retried") from None
        if len(raw) > 10 * 1024 * 1024:
            raise QuoteSourceError("Odds API response exceeds size limit; not retried")
        try:
            payload = json.loads(raw)
            events = self.parse(payload, captured)
        except (ValueError, KeyError, TypeError, AttributeError):
            raise QuoteSourceError("Invalid Odds API response; not retried") from None

        def credit(name: str) -> int | None:
            value = headers.get(name)
            return int(value) if isinstance(value, str) and value.isdigit() else None

        return QuoteSnapshot(
            events=events,
            source="odds_api",
            requests=1,
            credits_used=credit("x-requests-used"),
            credits_remaining=credit("x-requests-remaining"),
            credits_last_request=credit("x-requests-last"),
        )

    def parse(self, payload: list, captured: datetime) -> tuple[BookEvent, ...]:
        if not isinstance(payload, list):
            raise ValueError("Expected a list of NFL events")
        ids = [event["id"] for event in payload]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate provider event ID")
        events = []
        for event in payload:
            if event["sport_key"] != "americanfootball_nfl":
                raise ValueError("Unexpected sport")
            quotes = []
            for book in event["bookmakers"]:
                if book["key"] != self.book:
                    continue
                for market in book["markets"]:
                    # Some feeds timestamp markets individually; missing timestamps stay unknown.
                    updated = optional_timestamp(market.get("last_update", book.get("last_update")))
                    for ordinal, outcome in enumerate(market["outcomes"]):
                        quotes.append(
                            Quote(
                                reference=f"{event['id']}:{market['key']}:{ordinal}",
                                family=market["key"],
                                participant=outcome["name"],
                                line=outcome.get("point"),
                                odds=outcome["price"],
                                captured_at=captured,
                                updated_at=updated,
                                identity_evidence="live_response_context",
                            )
                        )
            kickoff = timestamp(event["commence_time"])
            events.append(
                BookEvent(
                    event_id=event["id"],
                    home=event["home_team"],
                    away=event["away_team"],
                    kickoff=kickoff,
                    quotes=tuple(quotes),
                    scheduled=kickoff > captured,
                )
            )
        return tuple(events)


@dataclass
class SQLiteQuoteSource(QuoteSource):
    """Opt-in reads of an existing engine store; never creates tables or migrates it."""

    path: Path
    book: str = "fanduel"

    def load(self) -> QuoteSnapshot:
        if os.environ.get("GOALPOST_LOCAL_DB_ENABLED", "false").lower() != "true":
            raise QuoteSourceError("Local database access requires GOALPOST_LOCAL_DB_ENABLED=true")
        validate_book(self.book)
        events = []
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            for event in db.execute("SELECT * FROM games WHERE league='nfl'").fetchall():
                kickoff = timestamp(event["commence_time_utc"])
                records = db.execute(
                    """
                    WITH latest AS (
                        SELECT m.market_type, m.participant, m.line, s.id, s.odds_american,
                               s.taken_at, s.provider_updated_at, s.event_context_json,
                               ROW_NUMBER() OVER (PARTITION BY m.id ORDER BY s.id DESC) AS rank
                        FROM markets m JOIN odds_snapshots s ON s.market_id=m.id
                        WHERE m.game_id=? AND s.book=? AND s.source IS NULL
                    ) SELECT * FROM latest WHERE rank=1
                """,
                    (event["id"], self.book),
                ).fetchall()
                quotes = []
                for record in records:
                    evidence = "current_database_identity_only"
                    context_matches = True
                    if record["event_context_json"]:
                        evidence = "captured_quote_context"
                        try:
                            context = json.loads(record["event_context_json"])
                            context_matches = (
                                context["home_team"] == event["home_team"]
                                and context["away_team"] == event["away_team"]
                                and timestamp(context["commence_time_utc"]) == kickoff
                            )
                        except (ValueError, KeyError, TypeError, AttributeError):
                            context_matches = False
                    quotes.append(
                        Quote(
                            reference=str(record["id"]),
                            family=record["market_type"],
                            participant=record["participant"],
                            line=record["line"],
                            odds=record["odds_american"],
                            captured_at=optional_timestamp(record["taken_at"]),
                            updated_at=optional_timestamp(record["provider_updated_at"]),
                            identity_evidence=evidence,
                            context_matches=context_matches,
                        )
                    )
                events.append(
                    BookEvent(
                        event_id=event["id"],
                        home=event["home_team"],
                        away=event["away_team"],
                        kickoff=kickoff,
                        quotes=tuple(quotes),
                        scheduled=event["status"] == "scheduled",
                    )
                )
        return QuoteSnapshot(tuple(events), "sqlite")
