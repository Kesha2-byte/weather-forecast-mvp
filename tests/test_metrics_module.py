import numpy as np
import pytest

import metrics as M


def test_mae_is_mean_absolute_difference():
    assert M.mae([0.0, 0.0, 0.0], [1.0, -3.0, 2.0]) == pytest.approx(2.0)


def test_rmse_penalises_large_errors_more_than_mae():
    y_true = [0.0, 0.0, 0.0, 0.0]
    spread = [3.0, -3.0, 3.0, -3.0]
    spike = [0.0, 0.0, 0.0, 12.0]
    assert M.mae(y_true, spread) == pytest.approx(M.mae(y_true, spike))
    assert M.rmse(y_true, spike) > M.rmse(y_true, spread)


def test_rmse_never_below_mae():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=500)
    y_pred = y_true + rng.normal(size=500)
    assert M.rmse(y_true, y_pred) >= M.mae(y_true, y_pred) - 1e-12


def test_bias_keeps_sign_while_mae_hides_it():
    y_true = [0.0, 0.0, 0.0]
    too_warm = [2.0, 2.0, 2.0]
    too_cold = [-2.0, -2.0, -2.0]
    assert M.bias(y_true, too_warm) == pytest.approx(2.0)
    assert M.bias(y_true, too_cold) == pytest.approx(-2.0)
    assert M.mae(y_true, too_warm) == pytest.approx(M.mae(y_true, too_cold))


def test_bias_cancels_out_for_symmetric_errors():
    y_true = [0.0, 0.0]
    y_pred = [5.0, -5.0]
    assert M.bias(y_true, y_pred) == pytest.approx(0.0)
    assert M.mae(y_true, y_pred) == pytest.approx(5.0)


def test_p90_and_worst_describe_the_tail():
    errors = list(range(101))
    y_true = [0.0] * 101
    assert M.p90(y_true, errors) == pytest.approx(90.0)
    assert M.worst(y_true, errors) == pytest.approx(100.0)


def test_rel_mae_equals_ratio_of_maes():
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [1.0, -1.0, 1.0, -1.0]
    y_base = [2.0, -2.0, 2.0, -2.0]
    assert M.rel_mae(y_true, y_pred, y_base) == pytest.approx(0.5)
    assert M.skill(y_true, y_pred, y_base) == pytest.approx(0.5)


def test_rel_mae_above_one_means_worse_than_baseline():
    y_true = [0.0, 0.0]
    y_pred = [3.0, 3.0]
    y_base = [1.0, 1.0]
    assert M.rel_mae(y_true, y_pred, y_base) > 1.0
    assert M.skill(y_true, y_pred, y_base) < 0.0


def test_naive_scale_uses_seasonal_difference():
    series = [0.0] * 24 + [5.0] * 24
    assert M.naive_scale(series, season=24) == pytest.approx(5.0)


def test_naive_scale_rejects_short_series():
    with pytest.raises(ValueError):
        M.naive_scale([1.0, 2.0, 3.0], season=24)


def test_mase_is_scale_free():
    y_true = [0.0, 0.0, 0.0]
    y_pred = [2.0, 2.0, 2.0]
    assert M.mase(y_true, y_pred, scale=4.0) == pytest.approx(0.5)
    celsius_to_half = [v * 0.5 for v in y_pred]
    assert M.mase(y_true, celsius_to_half, scale=2.0) == pytest.approx(0.5)


def test_mase_rejects_zero_scale():
    with pytest.raises(ValueError):
        M.mase([1.0], [1.0], scale=0.0)


def test_summary_contains_every_reported_metric():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.5, 2.5, 2.5, 4.5]
    y_base = [0.0, 0.0, 0.0, 0.0]
    out = M.summary(y_true, y_pred, y_base, scale=2.0)
    for key in ("mae", "rmse", "bias", "p90", "worst", "rel_mae", "rel_rmse", "skill_mae", "mase"):
        assert key in out
        assert np.isfinite(out[key])
