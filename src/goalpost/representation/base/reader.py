from pathlib import Path

import polars as pl
import numpy as np


class ParquetReader:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> pl.DataFrame:
        return pl.read_parquet(self.path)


def unpack_matrix(row: dict, shape: tuple) -> np.ndarray:
    flat = row["transition_probabilities_flat"]
    return np.array(flat, dtype=np.float32).reshape(shape)
