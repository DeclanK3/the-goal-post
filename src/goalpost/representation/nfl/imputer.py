"""NFL hierarchical imputation with 3-layer fallback."""
import numpy as np

from ..base.imputer import ImputationBuilder, ImputedDataset, History
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
        # Collect historical matrices for this team
        team_history = [m for date, m in history.entries if m.get("team") == team]
        season_matrices = []
        game_matrices = []
        for h in team_history:
            if h.get("type") == "season":
                season_matrices.append(h["matrix"])
            elif h.get("type") == "game":
                game_matrices.append(h["matrix"])

        for c in range(N_CONTEXTS):
            for s in range(N_STATES):
                if result[c, s].sum() > 0:
                    continue
                # Layer 3: specific game
                for gm in game_matrices:
                    if gm[c, s].sum() > 0:
                        result[c, s] = gm[c, s]
                        break
                else:
                    # Layer 2: season aggregate
                    for sm in season_matrices:
                        if sm[c, s].sum() > 0:
                            result[c, s] = sm[c, s]
                            break
                    else:
                        # Layer 1: global aggregate
                        global_mat = self._global_aggregate(history, c, s)
                        if global_mat is not None and global_mat.sum() > 0:
                            result[c, s] = global_mat
        return result

    def _global_aggregate(self, history: History, context: int, state: int) -> np.ndarray | None:
        agg = np.zeros(N_STATES, dtype=np.float32)
        count = 0
        for date, entry in history.entries:
            if "matrix" in entry:
                m = entry["matrix"]
                if m.shape == (N_CONTEXTS, N_STATES, N_STATES) and m[context, state].sum() > 0:
                    agg += m[context, state]
                    count += 1
        if count == 0:
            return None
        return agg / count
