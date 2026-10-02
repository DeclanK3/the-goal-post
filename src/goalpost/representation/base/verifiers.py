import numpy as np


def assert_stochastic(matrix: np.ndarray, atol: float = 1e-6):
    row_sums = matrix.sum(axis=-1)
    observed = row_sums > 0
    assert np.allclose(row_sums[observed], 1.0, atol=atol), "non-stochastic rows"


def assert_shape(matrix: np.ndarray, expected: tuple):
    assert matrix.shape == expected, f"shape mismatch: {matrix.shape} != {expected}"
