import numpy as np

from goalpost.representation.nfl.builder import (
    NFLTransitionBuilder, _map_destination, build_validity_mask,
    N_STATES, N_CONTEXTS,
)
from goalpost.representation.nfl.extractor import NFLExtractedGame, NFLEvent
from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.simulation.nfl.engine import NFLEngine


def _make_event(src, dst, switch, own_pts, opp_pts, remaining, lead, home_off=True, play_type="pass"):
    return NFLEvent(
        game_id="g1", week=1, half=1, play_id=1.0, next_play_id=None,
        offense="A", defense="B", home_offense=home_off,
        source=src, destination=dst, switch=switch,
        own_points=own_pts, opponent_points=opp_pts,
        elapsed=30.0, remaining=remaining, lead=lead, play_type=play_type, raw_rows=1,
    )


def test_validity_mask():
    mask = build_validity_mask()
    assert mask.shape == (N_STATES, N_STATES)
    assert mask[72, 72]
    assert not mask[72, 0]


def test_builder_produces_correct_shape():
    events = [
        _make_event(16, 17, False, 0, 0, 1200, 0),
        _make_event(17, 18, False, 0, 0, 1150, 0),
        _make_event(18, -1, True, 3, 0, 1100, 0),  # half-ending FG -> -1 -> skipped
        _make_event(16, 34, False, 7, 0, 600, 0),   # mid-half TD -> ABSORBING_TD
    ]
    game = NFLExtractedGame(
        game_id="g1", week=1, date="2024-01-01", home="A", away="B",
        events=events, starts=[], regulation=(0, 0), final=(0, 0), overtime=False,
    )
    builder = NFLTransitionBuilder()
    table = builder.build([game])
    assert table.keys
    for key, matrix in table.matrices.items():
        assert matrix.shape == (N_CONTEXTS, N_STATES, N_STATES)
        for c in range(N_CONTEXTS):
            for s in range(N_STATES):
                row = matrix[c, s]
                if row.sum() > 0:
                    assert abs(row.sum() - 1.0) < 1e-6
        for s in (72, 73, 74, 75):
            assert matrix[0, s, s] == 1.0
    # Mid-half TD observed only for home team (offense="A")
    home_matrix = table.matrices[("A", "g1")]
    assert home_matrix[1, 16, 73] > 0


def test_mid_half_scoring_simulates_non_zero():
    # Play 1: TD with PAT (7 pts for offense)
    # Play 2: FG (3 pts for offense)
    events = [
        _make_event(16, 34, False, 7, 0, 600, 0, home_off=True),
        _make_event(16, 17, False, 3, 0, 500, 0, home_off=False),
    ]
    game = NFLExtractedGame(
        game_id="g1", week=1, date="2024-01-01", home="A", away="B",
        events=events, starts=[], regulation=(0, 0), final=(0, 0), overtime=False,
    )
    builder = NFLTransitionBuilder()
    table = builder.build([game])

    # Load home matrix with point values into model
    home_model = NFLTransitionModel("A", "g1", "B")
    home_pts = table.point_values.get(("A", "g1"))
    home_model.load_matrix(table.matrices[("A", "g1")],
                           own_points=home_pts[0] if home_pts else None,
                           opp_points=home_pts[1] if home_pts else None)

    # Load away matrix with point values into model
    away_model = NFLTransitionModel("B", "g1", "A")
    away_pts = table.point_values.get(("B", "g1"))
    away_model.load_matrix(table.matrices[("B", "g1")],
                           own_points=away_pts[0] if away_pts else None,
                           opp_points=away_pts[1] if away_pts else None)

    engine = NFLEngine()
    outcomes = engine.simulate(home_model, away_model, n_sims=1000, seed=42)

    home_scores = np.array([o.home_score for o in outcomes])
    away_scores = np.array([o.away_score for o in outcomes])

    assert home_scores.mean() > 0, f"Expected home mean > 0, got {home_scores.mean()}"
    assert away_scores.mean() > 0, f"Expected away mean > 0, got {away_scores.mean()}"


def test_edge_case_mappings():
    from goalpost.representation.nfl.builder import (
        ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT,
    )
    from goalpost.representation.nfl.extractor import context_index

    cases = [
        ("fg", 3, 0, ABSORBING_FG, 3, 0),
        ("td_no_pat", 6, 0, ABSORBING_TD, 6, 0),
        ("td_pat", 7, 0, ABSORBING_TD, 7, 0),
        ("td_2pc", 8, 0, ABSORBING_TD, 8, 0),
        ("safety", 0, 2, ABSORBING_TO, 0, 2),
        ("def_td", 0, 7, ABSORBING_TO, 0, 7),
        ("punt", 0, 0, ABSORBING_PUNT, 0, 0),
        ("turnover", 0, 0, ABSORBING_TO, 0, 0),
    ]
    for label, own, opp, expected_dst, expected_own, expected_opp in cases:
        remaining = 600.0
        lead = 0
        if label == "punt":
            dst = 25
            play_type = "punt"
        elif label == "turnover":
            dst = 20
            play_type = "pass"
        else:
            dst = 34
            play_type = "pass"
        evt = _make_event(16, dst, label == "turnover", own, opp, remaining, lead, play_type=play_type)
        actual = _map_destination(evt)
        assert actual == expected_dst, f"{label}: expected {expected_dst}, got {actual}"

    # Build one event of each type and verify point values stored in model
    events = [
        _make_event(16, 34, False, 3, 0, 600, 0),   # FG
        _make_event(16, 34, False, 7, 0, 500, 0),   # TD
        _make_event(16, 34, False, 0, 2, 400, 0),   # safety
        _make_event(16, 34, True, 0, 0, 300, 0),   # turnover
        _make_event(16, 34, False, 0, 0, 200, 0, home_off=False),  # punt (away team)
    ]
    # Give each event a unique play_id so they don't collide
    events[0] = events[0].__class__(**{**events[0].__dict__, "play_id": 1.0})
    events[1] = events[1].__class__(**{**events[1].__dict__, "play_id": 2.0})
    events[2] = events[2].__class__(**{**events[2].__dict__, "play_id": 3.0})
    events[3] = events[3].__class__(**{**events[3].__dict__, "play_id": 4.0})
    events[4] = events[4].__class__(**{**events[4].__dict__, "play_id": 5.0})

    # Actually easier: rebuild events manually with play_id
    # Use distinct source states so transitions don't collide and average
    events = [
        _make_event(16, 34, False, 3, 0, 600, 0),           # FG from state 16
        _make_event(17, 34, False, 7, 0, 500, 0),           # TD from state 17
        _make_event(18, 34, False, 0, 2, 400, 0),           # safety from state 18
        _make_event(19, 34, True, 0, 0, 300, 0),            # turnover from state 19
    ]
    # Patch play_id so each event is distinct
    for i, e in enumerate(events):
        events[i] = NFLEvent(
            game_id=e.game_id, week=e.week, half=e.half,
            play_id=float(i + 1), next_play_id=e.next_play_id,
            offense=e.offense, defense=e.defense, home_offense=e.home_offense,
            source=e.source, destination=e.destination, switch=e.switch,
            own_points=e.own_points, opponent_points=e.opponent_points,
            elapsed=e.elapsed, remaining=e.remaining, lead=e.lead,
            play_type=e.play_type, raw_rows=e.raw_rows,
        )

    game = NFLExtractedGame(
        game_id="g1", week=1, date="2024-01-01", home="A", away="B",
        events=events, starts=[], regulation=(0, 0), final=(0, 0), overtime=False,
    )
    builder = NFLTransitionBuilder()
    table = builder.build([game])

    home_pts = table.point_values[("A", "g1")]
    home_own, home_opp = home_pts

    ctx = context_index(600.0, 0)  # ctx=1
    assert home_own[ctx, 16, ABSORBING_FG] == 3.0
    assert home_opp[ctx, 16, ABSORBING_FG] == 0.0

    ctx = context_index(500.0, 0)  # ctx=1
    assert home_own[ctx, 17, ABSORBING_TD] == 7.0
    assert home_opp[ctx, 17, ABSORBING_TD] == 0.0

    ctx = context_index(400.0, 0)  # ctx=1
    assert home_own[ctx, 18, ABSORBING_TO] == 0.0
    assert home_opp[ctx, 18, ABSORBING_TO] == 2.0

    ctx = context_index(300.0, 0)  # ctx=1
    assert home_own[ctx, 19, ABSORBING_TO] == 0.0
    assert home_opp[ctx, 19, ABSORBING_TO] == 0.0

    # Punt: separate event since play_type must be "punt"
    punt_event = _make_event(20, 25, False, 0, 0, 200, 0, play_type="punt")
    game2 = NFLExtractedGame(
        game_id="g2", week=1, date="2024-01-01", home="A", away="B",
        events=[punt_event], starts=[], regulation=(0, 0), final=(0, 0), overtime=False,
    )
    table2 = builder.build([game2])
    home_pts2 = table2.point_values[("A", "g2")]
    ctx = context_index(200.0, 0)  # ctx=1
    assert home_pts2[0][ctx, 20, ABSORBING_PUNT] == 0.0
    assert home_pts2[1][ctx, 20, ABSORBING_PUNT] == 0.0
