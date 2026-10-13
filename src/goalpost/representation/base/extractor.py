from abc import ABC, abstractmethod

import polars as pl

from .models import ExtractedGame


class GameExtractor(ABC):
    @abstractmethod
    def extract(self, raw_df: pl.DataFrame, metadata: dict) -> ExtractedGame:
        pass
