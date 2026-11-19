import numpy as np

HOURS_IN_DAY = 24


def _as_array(values):
    return np.asarray(values, dtype=float)


def mae(y_true, y_pred):
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred):
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    return float(np.mean(y_pred - y_true))


def p90(y_true, y_pred):
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    return float(np.percentile(np.abs(y_true - y_pred), 90))


def worst(y_true, y_pred):
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def rel_mae(y_true, y_pred, y_base):
    return mae(y_true, y_pred) / mae(y_true, y_base)


def skill(y_true, y_pred, y_base):
    return 1.0 - rel_mae(y_true, y_pred, y_base)


def naive_scale(train_series, season=HOURS_IN_DAY):
    values = _as_array(train_series)
    if values.size <= season:
        raise ValueError("train series shorter than one season")
    return float(np.mean(np.abs(values[season:] - values[:-season])))


def mase(y_true, y_pred, scale):
    if scale <= 0:
        raise ValueError("scale must be positive")
    return mae(y_true, y_pred) / scale


def summary(y_true, y_pred, y_base, scale=None):
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "p90": p90(y_true, y_pred),
        "worst": worst(y_true, y_pred),
        "base_mae": mae(y_true, y_base),
        "base_rmse": rmse(y_true, y_base),
        "rel_mae": rel_mae(y_true, y_pred, y_base),
        "rel_rmse": rmse(y_true, y_pred) / rmse(y_true, y_base),
        "skill_mae": skill(y_true, y_pred, y_base),
    }
    if scale is not None:
        out["mase"] = mase(y_true, y_pred, scale)
        out["base_mase"] = mase(y_true, y_base, scale)
    return out
