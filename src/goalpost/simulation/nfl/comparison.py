"""Match upcoming NFL simulations to exact current main-market quote pairs."""

from dataclasses import asdict, dataclass
from datetime import datetime
from math import isclose, isfinite

from goalpost.data.base.odds import BookEvent, Quote, QuoteSnapshot, validate_book
from goalpost.data.nfl.odds import MAIN_MARKETS
from goalpost.simulation.base.pricing import AmericanOdds

from .contracts import NFLGame, NFLMoneyline, NFLSpread, NFLTotal


@dataclass(frozen=True)
class ComparisonSettings:
    book: str = "fanduel"
    max_quote_age_seconds: float = 1800
    kickoff_tolerance_seconds: float = 600
    bankroll: float | None = None

    def __post_init__(self):
        validate_book(self.book)
        if not isfinite(self.max_quote_age_seconds) or self.max_quote_age_seconds <= 0:
            raise ValueError("Quote age limit must be positive")
        if not isfinite(self.kickoff_tolerance_seconds) or self.kickoff_tolerance_seconds < 0:
            raise ValueError("Kickoff tolerance must be nonnegative")
        if self.bankroll is not None and (not isfinite(self.bankroll) or self.bankroll <= 0):
            raise ValueError("Bankroll must be positive")


@dataclass
class MarketRow:
    game_id: str
    home: str
    away: str
    kickoff: str
    provider_game_id: str
    book: str
    market: str
    participant: str
    line: float | None
    odds: int
    quote_reference: str
    quote_status: str = "unpriced"
    reason: str | None = None
    quote_taken_at: str | None = None
    provider_updated_at: str | None = None
    quote_age_minutes: float | None = None
    provider_age_minutes: float | None = None
    identity_evidence: str | None = None
    p_win: float | None = None
    p_push_or_regulation_tie: float | None = None
    p_loss: float | None = None
    hypothetical_ev_pct: float | None = None
    full_kelly_pct: float | None = None
    quarter_kelly_pct: float | None = None
    hypothetical_quarter_kelly_dollars: float | None = None
    p_win_all_attempts_lower: float | None = None
    p_win_all_attempts_upper: float | None = None
    actual_contract_ev: float | None = None
    actual_contract_kelly: float | None = None
    calculation_scope: str = "hypothetical_regulation_pushes_or_ties_void"


def complete_pair(quotes: list[Quote], game: NFLGame) -> bool:
    if len(quotes) != 2:
        return False
    family = quotes[0].family
    expected = {"Over", "Under"} if family == "totals" else {game.home, game.away}
    if {quote.participant for quote in quotes} != expected:
        return False
    lines = [quote.line for quote in quotes]
    if family == "h2h":
        return all(line is None for line in lines)
    if any(type(line) not in (int, float) or not isfinite(line) for line in lines):
        return False
    return lines[0] == lines[1] if family == "totals" else isclose(sum(lines), 0, abs_tol=1e-8)


class NFLComparison:
    def __init__(self, settings: ComparisonSettings):
        self.settings = settings

    def select(self, event: BookEvent, game: NFLGame) -> tuple[list[Quote], list[dict]]:
        """Never rescue an incomplete newest pair by taking an older line."""
        selected, evidence = [], []
        for quote in event.quotes:
            if quote.family not in MAIN_MARKETS:
                evidence.append(self.disposition(quote, "unsupported_or_alternate_market"))
        for family in MAIN_MARKETS:
            family_quotes = [quote for quote in event.quotes if quote.family == family]
            if not family_quotes:
                continue
            if any(quote.captured_at is None for quote in family_quotes):
                evidence.extend(
                    self.disposition(quote, "unorderable_capture_timestamp")
                    for quote in family_quotes
                )
                continue
            newest = max(quote.captured_at for quote in family_quotes)
            latest = [quote for quote in family_quotes if quote.captured_at == newest]
            valid = complete_pair(latest, game)
            for quote in family_quotes:
                if quote.captured_at != newest:
                    disposition = "superseded_main_line"
                elif not valid:
                    disposition = "latest_main_pair_incomplete_or_invalid"
                else:
                    disposition = "selected_latest_main_pair"
                    selected.append(quote)
                evidence.append(self.disposition(quote, disposition))
        return selected, evidence

    @staticmethod
    def disposition(quote: Quote, reason: str) -> dict:
        return {
            "quote_reference": quote.reference,
            "market": quote.family,
            "participant": quote.participant,
            "line": quote.line,
            "odds": quote.odds,
            "captured_at": quote.captured_at.isoformat() if quote.captured_at else None,
            "disposition": reason,
        }

    def price(self, game: NFLGame, event: BookEvent, quote: Quote, now: datetime) -> MarketRow:
        row = MarketRow(
            game_id=game.game_id,
            home=game.home,
            away=game.away,
            kickoff=game.kickoff.isoformat(),
            provider_game_id=event.event_id,
            book=self.settings.book,
            market=quote.family,
            participant=quote.participant,
            line=quote.line,
            odds=quote.odds,
            quote_reference=quote.reference,
            identity_evidence=quote.identity_evidence,
        )
        if quote.captured_at is None or quote.updated_at is None:
            row.reason = "missing_or_invalid_quote_timestamp"
            return row
        row.quote_taken_at = quote.captured_at.isoformat()
        row.provider_updated_at = quote.updated_at.isoformat()
        capture_age = (now - quote.captured_at).total_seconds()
        provider_age = (now - quote.updated_at).total_seconds()
        row.quote_age_minutes, row.provider_age_minutes = capture_age / 60, provider_age / 60
        if min(capture_age, provider_age) < -60:
            row.reason = "future_quote_timestamp"
            return row
        if max(quote.captured_at, quote.updated_at) >= min(game.kickoff, event.kickoff):
            row.reason = "quote_at_or_after_kickoff"
            return row
        if not quote.context_matches:
            row.reason = "quote_context_mismatch"
            return row
        try:
            if quote.family == "h2h":
                market = NFLMoneyline(quote.participant)
            elif quote.family == "spreads":
                market = NFLSpread(quote.participant, quote.line)
            elif quote.family == "totals":
                market = NFLTotal(quote.line, quote.participant)
            else:
                raise ValueError("Unsupported market")
            mass = game.mass(market)
            sizing = AmericanOdds(quote.odds).size(mass)
        except (ValueError, TypeError):
            row.reason = "invalid_contract_or_price"
            return row
        row.p_win, row.p_push_or_regulation_tie, row.p_loss = mass.win, mass.push, mass.loss
        row.hypothetical_ev_pct = sizing.hypothetical_ev_pct
        row.full_kelly_pct, row.quarter_kelly_pct = sizing.full_kelly_pct, sizing.quarter_kelly_pct
        completion_rate = game.retained / game.attempted
        row.p_win_all_attempts_lower = completion_rate * mass.win
        row.p_win_all_attempts_upper = completion_rate * mass.win + 1 - completion_rate
        if self.settings.bankroll is not None:
            row.hypothetical_quarter_kelly_dollars = (
                self.settings.bankroll * sizing.quarter_kelly_pct / 100
            )
        row.quote_status = (
            "fresh_quote"
            if max(capture_age, provider_age) <= self.settings.max_quote_age_seconds
            else "stale_quote"
        )
        return row

    def run(self, games: list[NFLGame], snapshot: QuoteSnapshot, now: datetime) -> dict:
        if now.tzinfo is None:
            raise ValueError("Comparison time must include a timezone")
        rows, matchups, evidence = [], [], []
        for game in games:
            matchup = {
                "game_id": game.game_id,
                "home": game.home,
                "away": game.away,
                "attempted": game.attempted,
                "retained": game.retained,
                "approximate": game.approximate,
                "incomplete": game.attempted - game.retained,
            }
            matches = [
                event
                for event in snapshot.events
                if (event.home, event.away) == (game.home, game.away)
                and abs((event.kickoff - game.kickoff).total_seconds())
                <= self.settings.kickoff_tolerance_seconds
            ]
            if len(matches) != 1:
                matchup["quote_status"] = "missing_or_ambiguous_event"
            elif not matches[0].scheduled or min(matches[0].kickoff, game.kickoff) <= now:
                matchup["quote_status"] = "not_pregame"
            else:
                event = matches[0]
                selected, dispositions = self.select(event, game)
                matchup["quote_status"] = "quotes_selected" if selected else "missing_quotes"
                matchup["missing_main_families"] = [
                    family
                    for family in MAIN_MARKETS
                    if not any(quote.family == family for quote in selected)
                ]
                evidence.extend({"game_id": game.game_id, **entry} for entry in dispositions)
                rows.extend(asdict(self.price(game, event, quote, now)) for quote in selected)
            matchups.append(matchup)
        current = sorted(
            (row for row in rows if row["quote_status"] == "fresh_quote"),
            key=lambda row: (
                -row["hypothetical_ev_pct"],
                row["game_id"],
                row["market"],
                row["participant"],
            ),
        )
        return {"rows": rows, "current": current, "matchups": matchups, "quote_evidence": evidence}
