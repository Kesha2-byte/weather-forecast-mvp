import os
import warnings
from datetime import timedelta
from functools import lru_cache

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from features import (
    HOURS_IN_DAY,
    all_lags_up_to,
    baseline_lag,
    build_dataset,
    build_features,
    feature_names_for,
    lags_for,
    load_series,
)

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/weather")
MAX_HORIZON = 48
MODEL_NAME = "linear_regression"
MODEL_VERSION = "v2"

MODELS = {
    "seasonal_naive": "v1",
    "linear_regression": "v2",
    "holt_winters": "v2",
}

MODEL_TITLES = {
    "seasonal_naive": "Бейзлайн",
    "linear_regression": "Регрессия",
    "holt_winters": "Хольт-Уинтерс",
}

TRAIN_FRACTION = 0.8
_models = {}


def connect():
    return psycopg2.connect(DB_URL)


def filled_series(city_id, as_of=None):
    series = load_series(city_id)
    if as_of is not None:
        series = series[series["observed_at"] <= as_of]
    series = series.copy()
    series["temperature_c"] = series["temperature_c"].interpolate(limit_direction="both")
    return series.reset_index(drop=True)


@lru_cache(maxsize=None)
def get_features(city_id):
    return build_features(city_id, all_lags_up_to(MAX_HORIZON))


def get_model(city_id, horizon):
    key = (city_id, horizon)
    if key not in _models:
        dataset, names = build_dataset(city_id, horizon)
        model = LinearRegression()
        model.fit(dataset[names].to_numpy(), dataset["target"].to_numpy())
        _models[key] = model
    return _models[key]


@lru_cache(maxsize=None)
def holt_winters_params(city_id):
    values = filled_series(city_id)["temperature_c"].to_numpy(dtype=float)
    train = values[: int(len(values) * TRAIN_FRACTION)]
    fitted = ExponentialSmoothing(
        train,
        trend=None,
        seasonal="add",
        seasonal_periods=HOURS_IN_DAY,
        initialization_method="estimated",
    ).fit()
    return (
        float(fitted.params["smoothing_level"]),
        float(fitted.params["smoothing_seasonal"]),
    )


def predict_seasonal_naive(city_id, as_of, max_horizon):
    series = filled_series(city_id, as_of)
    values = series["temperature_c"].to_numpy(dtype=float)
    created_at = series["observed_at"].iloc[-1]
    predictions = [float(values[-1 - baseline_lag(h)]) for h in range(1, max_horizon + 1)]
    return created_at, predictions


def predict_linear_regression(city_id, as_of, max_horizon):
    features, _ = get_features(city_id)
    if as_of is None:
        row = features.iloc[-1]
    else:
        eligible = features[features["observed_at"] <= as_of]
        if eligible.empty:
            raise ValueError(f"Нет данных до {as_of}")
        row = eligible.iloc[-1]

    created_at = row["observed_at"]
    predictions = []
    for horizon in range(1, max_horizon + 1):
        model = get_model(city_id, horizon)
        wanted = feature_names_for(lags_for(horizon))
        inputs = row[wanted].to_numpy(dtype=float).reshape(1, -1)
        predictions.append(float(model.predict(inputs)[0]))
    return created_at, predictions


def predict_holt_winters(city_id, as_of, max_horizon):
    level, seasonal = holt_winters_params(city_id)
    series = filled_series(city_id, as_of)
    values = series["temperature_c"].to_numpy(dtype=float)
    created_at = series["observed_at"].iloc[-1]
    fitted = ExponentialSmoothing(
        values,
        trend=None,
        seasonal="add",
        seasonal_periods=HOURS_IN_DAY,
        initialization_method="estimated",
    ).fit(smoothing_level=level, smoothing_seasonal=seasonal, optimized=False)
    predictions = [float(v) for v in np.asarray(fitted.forecast(max_horizon), dtype=float)]
    return created_at, predictions


BUILDERS = {
    "seasonal_naive": predict_seasonal_naive,
    "linear_regression": predict_linear_regression,
    "holt_winters": predict_holt_winters,
}


def build_forecast(city_id, as_of=None, max_horizon=MAX_HORIZON, model=MODEL_NAME):
    if model not in BUILDERS:
        raise ValueError(f"Неизвестная модель: {model}")

    created_at, values = BUILDERS[model](city_id, as_of, max_horizon)

    predictions = []
    for horizon, value in enumerate(values, start=1):
        predictions.append({
            "horizon_hours": horizon,
            "target_time": created_at + timedelta(hours=horizon),
            "predicted_temperature_c": round(value, 2),
        })
    return created_at, predictions


def resolve_model_id(cursor, name=MODEL_NAME, version=None):
    if version is None:
        version = MODELS.get(name, MODEL_VERSION)
    cursor.execute(
        "SELECT model_id FROM models WHERE name = %s AND version = %s",
        (name, version),
    )
    row = cursor.fetchone()
    if row is None:
        raise LookupError(f"Model {name} {version} not found")
    return row[0]


INSERT_FORECASTS = """
INSERT INTO forecasts (
    city_id, model_id, created_at, target_time, horizon_hours, predicted_temperature_c
) VALUES %s
ON CONFLICT (city_id, model_id, created_at, target_time) DO NOTHING
"""


def save_forecast(city_id, created_at, predictions, model=MODEL_NAME):
    with connect() as conn:
        with conn.cursor() as cursor:
            model_id = resolve_model_id(cursor, model)
            rows = [
                (city_id, model_id, created_at, p["target_time"],
                 p["horizon_hours"], p["predicted_temperature_c"])
                for p in predictions
            ]
            execute_values(cursor, INSERT_FORECASTS, rows, page_size=100)
            stored = cursor.rowcount
    return stored


def run_and_store(city_id, as_of=None, max_horizon=MAX_HORIZON, model=MODEL_NAME):
    created_at, predictions = build_forecast(city_id, as_of, max_horizon, model)
    stored = save_forecast(city_id, created_at, predictions, model)
    return created_at, predictions, stored
