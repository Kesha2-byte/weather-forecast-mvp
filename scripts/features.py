import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import psycopg2

DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/weather")

DEFAULT_LAGS = (0, 1, 24, 48)
HOURS_IN_DAY = 24
DAYS_IN_YEAR = 365.25
LOCAL_TZ = "Asia/Almaty"


@lru_cache(maxsize=None)
def load_series(city_id):
    sql = """
        SELECT observed_at, temperature_c
        FROM observations
        WHERE city_id = %s
        ORDER BY observed_at
    """
    conn = psycopg2.connect(DB_URL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql_query(sql, conn, params=(city_id,))
    finally:
        conn.close()

    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["temperature_c"] = df["temperature_c"].astype(float)
    return reindex_hourly(df)


def reindex_hourly(df):
    series = df.set_index("observed_at").sort_index()
    series.index = series.index.tz_convert("UTC")
    full = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
    series = series.reindex(full)
    series.index = series.index.tz_convert(LOCAL_TZ)
    series.index.name = "observed_at"
    return series.reset_index()


def baseline_lag(horizon):
    periods_back = -(-horizon // HOURS_IN_DAY)
    return periods_back * HOURS_IN_DAY - horizon


def lags_for(horizon, base=DEFAULT_LAGS):
    extra = baseline_lag(horizon)
    base = tuple(base)
    return base if extra in base else base + (extra,)


def all_lags_up_to(max_horizon, base=DEFAULT_LAGS):
    found = set(base)
    for horizon in range(1, max_horizon + 1):
        found.add(baseline_lag(horizon))
    return tuple(sorted(found))


def add_lag_features(df, lags=DEFAULT_LAGS):
    for lag in lags:
        df[f"lag_{lag}"] = df["temperature_c"].shift(lag)
    return df


def add_cyclical_features(df):
    hour = df["observed_at"].dt.hour
    day_of_year = df["observed_at"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / HOURS_IN_DAY)
    df["hour_cos"] = np.cos(2 * np.pi * hour / HOURS_IN_DAY)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / DAYS_IN_YEAR)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / DAYS_IN_YEAR)
    return df


def add_target(df, horizon):
    df["target"] = df["temperature_c"].shift(-horizon)
    return df


def add_baseline(df, horizon):
    df["baseline"] = df["temperature_c"].shift(baseline_lag(horizon))
    return df


def feature_names_for(lags=DEFAULT_LAGS):
    return [f"lag_{lag}" for lag in lags] + [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos"
    ]


def build_features(city_id, lags=DEFAULT_LAGS):
    df = load_series(city_id).copy()
    df = add_lag_features(df, lags)
    df = add_cyclical_features(df)

    names = feature_names_for(lags)
    columns = ["observed_at", "temperature_c"] + names

    return df[columns].dropna().reset_index(drop=True), names


def build_dataset(city_id, horizon, lags=None):
    if lags is None:
        lags = lags_for(horizon)
    df = load_series(city_id).copy()
    df = add_lag_features(df, lags)
    df = add_cyclical_features(df)
    df = add_target(df, horizon)
    df = add_baseline(df, horizon)

    names = feature_names_for(lags)
    columns = ["observed_at", "temperature_c"] + names + ["baseline", "target"]

    dataset = df[columns].dropna().reset_index(drop=True)
    return dataset, names


def chronological_split(dataset, test_size=0.2, gap_hours=None, horizon=None):
    if gap_hours is None:
        gap_hours = horizon if horizon is not None else HOURS_IN_DAY
    split_at = int(len(dataset) * (1 - test_size))

    train = dataset.iloc[: split_at - gap_hours]
    test = dataset.iloc[split_at:]
    return train, test


if __name__ == "__main__":
    HORIZON = 24

    for city_id, city_name in [(1, "Алматы"), (2, "Астана")]:
        dataset, names = build_dataset(city_id, HORIZON)
        train, test = chronological_split(dataset, horizon=HORIZON)
        features, _ = build_features(city_id)

        print(f"\n{city_name}")
        print(f"  Строк для обучения:  {len(dataset)}")
        print(f"  Строк для прогноза:  {len(features)}")
        print(f"  Последняя точка:     {features['observed_at'].max()}")
        print(f"  Обучающая выборка:   {len(train)}")
        print(f"  Тестовая выборка:    {len(test)}")
        print(f"  Лаги:                {lags_for(HORIZON)}")

        assert train["observed_at"].max() < test["observed_at"].min(), \
            "Выборки пересекаются по времени"
