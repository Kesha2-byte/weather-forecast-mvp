import os
from datetime import timedelta
from functools import lru_cache

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sklearn.linear_model import LinearRegression

from features import build_dataset, build_features

np.seterr(all="ignore")

DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/weather")
MAX_HORIZON = 48
MODEL_NAME = "linear_regression"
MODEL_VERSION = "v1"

_models = {}


def connect():
    return psycopg2.connect(DB_URL)


@lru_cache(maxsize=None)
def get_features(city_id):
    return build_features(city_id)


def get_model(city_id, horizon):
    key = (city_id, horizon)
    if key not in _models:
        dataset, names = build_dataset(city_id, horizon)
        model = LinearRegression()
        model.fit(dataset[names].to_numpy(), dataset["target"].to_numpy())
        _models[key] = model
    return _models[key]


def resolve_model_id(cursor, name=MODEL_NAME, version=MODEL_VERSION):
    cursor.execute(
        "SELECT model_id FROM models WHERE name = %s AND version = %s",
        (name, version),
    )
    row = cursor.fetchone()
    if row is None:
        raise LookupError(f"Model {name} {version} not found")
    return row[0]


def build_forecast(city_id, as_of=None, max_horizon=MAX_HORIZON):
    features, names = get_features(city_id)

    if as_of is None:
        row = features.iloc[-1]
    else:
        eligible = features[features["observed_at"] <= as_of]
        if eligible.empty:
            raise ValueError(f"Нет данных до {as_of}")
        row = eligible.iloc[-1]

    created_at = row["observed_at"]
    inputs = row[names].to_numpy().reshape(1, -1)

    predictions = []
    for horizon in range(1, max_horizon + 1):
        model = get_model(city_id, horizon)
        value = float(model.predict(inputs)[0])
        predictions.append({
            "horizon_hours": horizon,
            "target_time": created_at + timedelta(hours=horizon),
            "predicted_temperature_c": round(value, 2),
        })

    return created_at, predictions


INSERT_FORECASTS = """
INSERT INTO forecasts (
    city_id, model_id, created_at, target_time, horizon_hours, predicted_temperature_c
) VALUES %s
ON CONFLICT (city_id, model_id, created_at, target_time) DO NOTHING
"""


def save_forecast(city_id, created_at, predictions):
    with connect() as conn:
        with conn.cursor() as cursor:
            model_id = resolve_model_id(cursor)
            rows = [
                (city_id, model_id, created_at, p["target_time"],
                 p["horizon_hours"], p["predicted_temperature_c"])
                for p in predictions
            ]
            execute_values(cursor, INSERT_FORECASTS, rows, page_size=100)
            stored = cursor.rowcount
    return stored


def run_and_store(city_id, as_of=None, max_horizon=MAX_HORIZON):
    created_at, predictions = build_forecast(city_id, as_of, max_horizon)
    stored = save_forecast(city_id, created_at, predictions)
    return created_at, predictions, stored
