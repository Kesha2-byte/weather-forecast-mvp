import numpy as np
import pandas as pd
import pytest

from models import evaluate_baseline, mae, rmse


class TestMae:
    def test_known_values(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 4.0, 3.0])

        assert mae(actual, predicted) == pytest.approx(2.0 / 3.0)

    def test_perfect_prediction_gives_zero(self):
        actual = np.array([5.0, -3.0, 0.0])

        assert mae(actual, actual) == 0.0

    def test_sign_does_not_matter(self):
        actual = np.array([0.0, 0.0])
        over = np.array([2.0, 2.0])
        under = np.array([-2.0, -2.0])

        assert mae(actual, over) == mae(actual, under)


class TestRmse:
    def test_known_values(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 4.0, 3.0])

        assert rmse(actual, predicted) == pytest.approx(np.sqrt(4.0 / 3.0))

    def test_perfect_prediction_gives_zero(self):
        actual = np.array([5.0, -3.0, 0.0])

        assert rmse(actual, actual) == 0.0

    def test_penalises_large_errors_harder_than_mae(self):
        actual = np.array([0.0, 0.0, 0.0, 0.0])
        scattered = np.array([1.0, 1.0, 1.0, 1.0])
        concentrated = np.array([4.0, 0.0, 0.0, 0.0])

        assert mae(actual, scattered) == mae(actual, concentrated)
        assert rmse(actual, concentrated) > rmse(actual, scattered)


class TestEvaluateBaseline:
    def test_uses_baseline_column(self):
        test = pd.DataFrame({
            "target": [10.0, 12.0, 14.0],
            "baseline": [10.0, 10.0, 10.0],
        })

        result_mae, result_rmse = evaluate_baseline(test)

        assert result_mae == pytest.approx(2.0)
        assert result_rmse == pytest.approx(np.sqrt(20.0 / 3.0))
