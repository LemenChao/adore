import numpy as np
from sklearn.linear_model import LinearRegression
from adore.core import ADORE


def test_adore():
    X = np.random.rand(100, 5)
    y = X @ np.array([1.5, -2.0, 3.0, 0.0, 1.0]) + np.random.randn(100) * 0.1

    model = LinearRegression().fit(X, y)

    adore = ADORE(model, X)
    contributions, _, _, _, _, _, _, _ = adore.explain()

    assert contributions.shape == X.shape
