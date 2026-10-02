import numpy as np

from goalpost.representation.nfl.builder import NFLTransitionBuilder, build_validity_mask, N_STATES, N_CONTEXTS
from goalpost.representation.nfl.extractor import NFLExtractedGame, NFLEvent


def _make_event(src, dst, switch, own_pts, opp_pts, remaining, lead, home_off=True):
    return NFLEvent(
        game_id="g1", week=1, half=1, play_id=1.0, next_play_id=None,
        offense="A", defense="B", home_offense=home_off,
        source=src, destination=dst, switch=switch,
        own_points=own_pts, opponent_points=opp_pts,
        elapsed=30.0, remaining=remaining, lead=lead, play_type="pass", raw_rows=1,
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
        _make_event(18, 72, True, 3, 0, 1100, 0),
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
        # Observed rows sum to 1
        for c in range(N_CONTEXTS):
            for s in range(N_STATES):
                row = matrix[c, s]
                if row.sum() > 0:
                    assert abs(row.sum() - 1.0) < 1e-6
        # Absorbing rows self-loop
        for s in (72, 73, 74, 75):
            assert matrix[0, s, s] == 1.0
