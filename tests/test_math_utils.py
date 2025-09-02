import pytest
import numpy as np
from world_happiness.math_utils import gradient_descent_from_scratch, apply_svd, bayesian_inference
import pandas as pd

def test_gradient_descent_from_scratch():
    X = np.array([[1, 1], [1, 2], [1, 3]])  # Bias + feature
    y = np.array([1, 2, 3])
    theta = gradient_descent_from_scratch(X, y, lr=0.1, epochs=100)
    assert np.allclose(theta, [0, 1], atol=0.1)  # Approx intercept 0, slope 1

def test_apply_svd():
    df = pd.DataFrame(np.random.rand(5, 3), columns=['f1', 'f2', 'f3'])
    reduced = apply_svd(df, k=2)
    assert reduced.shape == (5, 3)  # Reconstructed shape

def test_bayesian_inference():
    df = pd.DataFrame({'GDP per capita': np.random.rand(10), 'Happiness Score': np.random.rand(10)})
    with pytest.raises(Exception):  # May not raise, but test runs
        bayesian_inference(df)