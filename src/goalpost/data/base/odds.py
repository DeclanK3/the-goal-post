"""Sport-neutral quote snapshots and explicit quote-source interface."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


def timestamp(value: str) -> datetime:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return instant


class QuoteSourceError(ValueError):
    """A sanitized provider error that never contains a credential-bearing URL."""


def optional_timestamp(value) -> datetime | None:
    try:
        return timestamp(value)
    except (ValueError, TypeError, AttributeError):
        return None


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


@dataclass(frozen=True)
class BookEvent:
    event_id: str
    home: str
    away: str
    kickoff: datetime
    quotes: tuple[Quote, ...]
    scheduled: bool = True


@dataclass(frozen=True)
class QuoteSnapshot:
    events: tuple[BookEvent, ...]
    source: str
    requests: int = 0
    credits_used: int | None = None
    credits_remaining: int | None = None
    credits_last_request: int | None = None


class QuoteSource(ABC):
    """Acquire one comparison snapshot. Calling load again is an explicit refresh."""

    @abstractmethod
    def load(self) -> QuoteSnapshot:
        """Return events and exact quote timestamps, including missing coverage."""


def validate_book(book: str) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", book):
        raise ValueError("Specify one bookmaker key")
