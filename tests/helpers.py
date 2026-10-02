import numpy as np

from goalpost.data.base.source import DataSource
from goalpost.data.base.models import Game
from goalpost.representation.base.extractor import GameExtractor
from goalpost.representation.base.models import ExtractedGame, TransitionModel
from goalpost.simulation.base.engine import Simulator
from goalpost.simulation.base.models import GameOutcome


class ToyGameExtractor(GameExtractor):
    def extract(self, raw_df, metadata):
        return ExtractedGame(game_id=metadata.get("game_id", "toy"), events=[])


class ToyTransitionModel(TransitionModel):
    @property
    def state_space_size(self):
        return 3

    @property
    def context_count(self):
        return 2

    @property
    def validity_mask(self):
        m = np.zeros((3, 3), dtype=bool)
        m[0, 0] = m[0, 1] = m[1, 2] = m[2, 2] = True
        return m

    def extract_from_game(self, game):
        pass

    def get_probabilities(self, context, state):
        return {state: 1.0}

    def flatten(self):
        return np.array([1.0, 0.0])

    @classmethod
    def from_flat(cls, array, team_id, game_id, opponent_id):
        return cls(team_id, game_id, opponent_id)


class ToySimulator(Simulator):
    def simulate(self, transition_model, n_sims=10000, seed=0):
        rng = np.random.default_rng(seed)
        return [GameOutcome(home_score=int(rng.integers(10)), away_score=int(rng.integers(10))) for _ in range(n_sims)]
