"""Regulation PMF validation, exact contract probabilities and hypothetical sizing."""

import math
from datetime import datetime


def stamp(value):
    if not isinstance(value, str):
        raise ValueError("Missing or invalid timestamp")
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("Timezone missing")
    return d


def validate_pmfs(pmfs, included):
    """Validate exported masses against their common completed-path denominator."""
    if not isinstance(included, int) or included <= 0:
        raise ValueError("No included simulation paths")
    for name, frame in pmfs.items():
        if not {"points", "probability", "count"}.issubset(frame.columns) or frame.empty:
            raise ValueError("PMF columns or rows missing")
        x = frame.points
        p = frame.probability
        counts = frame["count"]
        if not all(math.isfinite(float(v)) for col in (x, p, counts) for v in col):
            raise ValueError("Nonfinite PMF")
        if x.duplicated().any() or not (x % 1 == 0).all() or (name == "total" and (x < 0).any()):
            raise ValueError("Invalid score support")
        if not (counts >= 0).all() or not (counts % 1 == 0).all() or int(counts.sum()) != included:
            raise ValueError("PMF count denominator mismatch")
        if not (p >= 0).all() or not math.isclose(float(p.sum()), 1, abs_tol=1e-8):
            raise ValueError("Invalid PMF probability mass")
        if any(abs(float(a) - float(b) / included) > 1e-8 for a, b in zip(p, counts)):
            raise ValueError("PMF probabilities disagree with counts")


def ranked_current(rows):
    # Rank contracts, not independent plays; no minimum count is forced.
    return sorted(
        (r for r in rows if r.get("quote_status") == "fresh_stored_quote"),
        key=lambda r: (-r["hypothetical_ev_pct"], r["game_id"], r["market"], r["participant"]),
    )


def mass(frame, family, participant, line, home, away):
    x = frame.points
    p = frame.probability
    if len(frame) == 0 or not p.ge(0).all() or not math.isclose(float(p.sum()), 1, abs_tol=1e-8):
        raise ValueError("Invalid PMF")
    if family in ("totals", "alternate_totals"):
        if participant not in ("Over", "Under") or line is None or not math.isfinite(float(line)):
            raise ValueError("Unknown total contract")
        difference = x - float(line) if participant == "Over" else float(line) - x
    elif family in ("spreads", "alternate_spreads"):
        if participant not in (home, away) or line is None or not math.isfinite(float(line)):
            raise ValueError("Unknown spread contract")
        difference = (x if participant == home else -x) + float(line)
    elif family == "h2h":
        if participant not in (home, away) or line is not None:
            raise ValueError("Unknown moneyline contract")
        difference = x if participant == home else -x
    else:
        raise ValueError("Unsupported contract")
    return tuple(float(p[mask].sum()) for mask in (difference > 0, difference == 0, difference < 0))


def sizing(win, push, loss, odds, fraction=0.25):
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (win, push, loss)) or not math.isclose(
        win + push + loss, 1, abs_tol=1e-8
    ):
        raise ValueError("Invalid probabilities")
    if not isinstance(odds, int) or abs(odds) < 100:
        raise ValueError("Invalid American odds")
    b = odds / 100 if odds > 0 else 100 / abs(odds)
    ev = b * win - loss
    full = max(0, min(1, ev / (b * (win + loss)))) if win + loss else 0
    return dict(
        decimal_odds=1 + b,
        hypothetical_ev_pct=100 * ev,
        full_kelly_pct=100 * full,
        quarter_kelly_pct=100 * full * fraction,
    )
