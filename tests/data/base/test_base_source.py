import pytest

from goalpost.data.base.source import DataSource


def test_data_source_is_abstract():
    with pytest.raises(TypeError):
        DataSource()
