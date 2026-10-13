"""NFL hierarchical imputation with 3-layer fallback."""
import numpy as np

from ..base.imputer import ImputationBuilder, ImputedDataset
from ..base.history import History
from .builder import NFLGameTable, N_STATES, N_CONTEXTS


class NFLImputedDataset(ImputedDataset):
    def __init__(self, matrices: dict):
        self.matrices = matrices


class NFLImputationBuilder(ImputationBuilder):
    def build(self, tables: NFLGameTable, history: History) -> NFLImputedDataset:
        imputed = {}
        for key, game_matrix in tables.matrices.items():
            team, game_id = key
            imputed[key] = self._impute(game_matrix, team, history)
        return NFLImputedDataset(imputed)

    def _impute(self, matrix: np.ndarray, team: str, history: History) -> np.ndarray:
        result = matrix.copy()
        team_entries = history.by_team(team)
        season_entries = {}
        game_entries = {}
        for e in team_entries:
            season_entries.setdefault(e.season, []).append(e)
            game_entries[e.game_id] = e

        for c in range(N_CONTEXTS):
            for s in range(N_STATES):
                if result[c, s].sum() > 0:
                    continue
                # Layer 3: specific game
                for gid, entry in game_entries.items():
                    if entry.matrix[c, s].sum() > 0:
                        result[c, s] = entry.matrix[c, s]
                        break
                else:
                    # Layer 2: season aggregate
                    for season, entries in season_entries.items():
                        agg = self._season_aggregate(entries, c, s)
                        if agg is not None and agg.sum() > 0:
                            result[c, s] = agg
                            break
                    else:
                        # Layer 1: global aggregate
                        global_mat = history.global_aggregate(c, s, N_CONTEXTS, N_STATES)
                        if global_mat is not None and global_mat.sum() > 0:
                            result[c, s] = global_mat
        return result

    def _season_aggregate(self, entries, context: int, state: int) -> np.ndarray | None:
        agg = np.zeros(N_STATES, dtype=np.float32)
        count = 0
        for entry in entries:
            m = entry.matrix
            if m[context, state].sum() > 0:
                agg += m[context, state]
                count += 1
        if count == 0:
            return None
        return agg / count
