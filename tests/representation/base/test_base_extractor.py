import pytest
import polars as pl

from goalpost.representation.base.extractor import GameExtractor
from goalpost.representation.base.models import ExtractedGame


class ToyGameExtractor(GameExtractor):
    def extract(self, raw_df, metadata):
        return ExtractedGame(game_id=metadata.get("game_id", "toy"), events=[])


def test_game_extractor_is_abstract():
    with pytest.raises(TypeError):
        GameExtractor()


def test_toy_extractor():
    ext = ToyGameExtractor()
    df = pl.DataFrame({"a": [1, 2, 3]})
    game = ext.extract(df, {"game_id": "test"})
    assert game.game_id == "test"
