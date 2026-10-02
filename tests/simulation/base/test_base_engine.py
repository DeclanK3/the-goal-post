import pytest

from goalpost.simulation.base.engine import Simulator


def test_simulator_is_abstract():
    with pytest.raises(TypeError):
        Simulator()
