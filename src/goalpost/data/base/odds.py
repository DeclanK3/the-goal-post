# Purpose: Represent provider-neutral quote evidence without performing collection.
# Public API: Quote, BookEvent, QuoteSnapshot, QuoteSource, timestamp, optional_timestamp,
# validate_book.
# Consumers: NFL quote adapters and the market comparison/report pipeline.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# Purpose: Reject timezone-free event instants before identity or freshness comparisons.
# Callers: Quote adapters and WeeklyExport identity checks.
def timestamp(value: str) -> datetime:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return instant


# Purpose: Expose sanitized source failures without credential-bearing transport details.
# Consumers: Quote adapters and CLI/TUI failure reporting.
class QuoteSourceError(ValueError):
    pass


# Purpose: Preserve absent or malformed quote timestamps as unknown evidence.
# Callers: API and SQLite adapters; comparison clients exclude unknown timestamps.
def optional_timestamp(value: Any) -> datetime | None:
    try:
        return timestamp(value)
    except (ValueError, TypeError, AttributeError):
        return None


# Purpose: Retain one selection with capture/provider clocks and identity provenance.
# Consumers: Quote adapters construct records; NFLComparison selects and prices them.
@dataclass(frozen=True)
class Quote:
    reference: str
    family: str
    participant: str
    line: float | None
    odds: int
    captured_at: datetime | None
    updated_at: datetime | None
    identity_evidence: str
    context_matches: bool = True


# Purpose: Bind a provider event identity and schedule state to all returned selections.
# Consumers: Quote sources and NFLComparison event matching.
@dataclass(frozen=True)
class BookEvent:
    event_id: str
    home: str
    away: str
    kickoff: datetime
    quotes: tuple[Quote, ...]
    scheduled: bool = True


# Purpose: Carry events and observed request accounting without assuming missing credits are zero.
# Consumers: Quote sources produce snapshots; report jobs persist their evidence.
@dataclass(frozen=True)
class QuoteSnapshot:
    events: tuple[BookEvent, ...]
    source: str
    requests: int = 0
    credits_used: int | None = None
    credits_remaining: int | None = None
    credits_last_request: int | None = None


# Purpose: Require explicit snapshot loading so refreshes cannot occur implicitly.
# Consumers: OddsAPISource, SQLiteQuoteSource and MarketTableJob.
class QuoteSource(ABC):
    # Purpose: Return one evidence snapshot, retaining empty coverage and unknown clocks.
    # Callers: MarketTableJob through a concrete, explicitly selected source.
    @abstractmethod
    def load(self) -> QuoteSnapshot:
        pass


# Purpose: Require one safe bookmaker key instead of an implicit multi-book request.
# Callers: Quote sources and ComparisonSettings initialization.
def validate_book(book: str) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", book):
        raise ValueError("Specify one bookmaker key")
