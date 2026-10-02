import numpy as np
import pytest

from goalpost.representation.base.builder import TransitionBuilder


def test_transition_builder_is_abstract():
    with pytest.raises(TypeError):
        TransitionBuilder()
