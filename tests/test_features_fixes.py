import numpy as np
import pandas as pd
import pytest

from features import (
    DEFAULT_LAGS,
    HOURS_IN_DAY,
    LOCAL_TZ,
    add_baseline,
    all_lags_up_to,
    baseline_lag,
    chronological_split,
    lags_for,
    reindex_hourly,
)


def hourly(values, start="2025-06-01"):
    index = pd.date_range(start, periods=len(values), freq="h", tz=LOCAL_TZ)
    return pd.DataFrame({"observed_at": index, "temperature_c": np.asarray(values, dtype=float)})


def test_baseline_lag_matches_definition():
    expected = {1: 23, 6: 18, 12: 12, 23: 1, 24: 0, 25: 23, 30: 18, 47: 1, 48: 0}
    assert {h: baseline_lag(h) for h in expected} == expected


def test_baseline_lag_is_zero_only_on_whole_days():
    for horizon in range(1, 49):
        assert (baseline_lag(horizon) == 0) == (horizon % HOURS_IN_DAY == 0)


def test_baseline_lag_stays_inside_one_day():
    for horizon in range(1, 49):
        assert 0 <= baseline_lag(horizon) < HOURS_IN_DAY


def test_lags_for_covers_the_baseline():
    for horizon in range(1, 49):
        assert baseline_lag(horizon) in lags_for(horizon)


def test_lags_for_keeps_base_lags():
    for horizon in (1, 12, 24, 48):
        assert set(DEFAULT_LAGS).issubset(set(lags_for(horizon)))


def test_lags_for_adds_nothing_on_whole_days():
    assert lags_for(24) == tuple(DEFAULT_LAGS)
    assert lags_for(48) == tuple(DEFAULT_LAGS)


def test_lags_for_adds_exactly_one_lag_otherwise():
    for horizon in (1, 6, 12, 18, 30, 42):
        assert len(lags_for(horizon)) == len(DEFAULT_LAGS) + 1


def test_lags_for_has_no_duplicates():
    for horizon in range(1, 49):
        assert len(set(lags_for(horizon))) == len(lags_for(horizon))


def test_all_lags_up_to_is_the_union():
    union = all_lags_up_to(48)
    for horizon in range(1, 49):
        assert set(lags_for(horizon)).issubset(set(union))


def test_all_lags_up_to_is_sorted_and_unique():
    union = all_lags_up_to(48)
    assert list(union) == sorted(set(union))


def test_baseline_column_equals_its_lag():
    frame = hourly(np.arange(300))
    horizon = 12
    frame["lag"] = frame["temperature_c"].shift(baseline_lag(horizon))
    add_baseline(frame, horizon)
    assert np.allclose(frame["baseline"].dropna(), frame["lag"].dropna())


def test_baseline_points_at_the_same_clock_hour_as_the_target():
    frame = hourly(np.arange(400))
    for horizon in (1, 6, 12, 24, 36, 48):
        offset = baseline_lag(horizon)
        assert (horizon + offset) % HOURS_IN_DAY == 0


def test_reindex_inserts_the_missing_hour_of_the_timezone_change():
    times = pd.to_datetime(
        ["2024-02-29T15:00Z", "2024-02-29T16:00Z", "2024-02-29T18:00Z", "2024-02-29T19:00Z"],
        utc=True,
    ).tz_convert(LOCAL_TZ)
    raw = pd.DataFrame({"observed_at": times, "temperature_c": [-9.8, -9.1, -4.5, -8.3]})
    fixed = reindex_hourly(raw)
    assert len(fixed) == len(raw) + 1
    assert int(fixed["temperature_c"].isna().sum()) == 1


def test_reindex_makes_steps_uniform():
    times = pd.to_datetime(
        ["2024-02-29T15:00Z", "2024-02-29T16:00Z", "2024-02-29T18:00Z"], utc=True
    ).tz_convert(LOCAL_TZ)
    raw = pd.DataFrame({"observed_at": times, "temperature_c": [1.0, 2.0, 3.0]})
    gaps = reindex_hourly(raw)["observed_at"].diff().dropna().unique()
    assert len(gaps) == 1
    assert gaps[0] == pd.Timedelta(hours=1)


def test_reindex_keeps_a_complete_series_unchanged():
    frame = hourly(np.arange(100))
    assert len(reindex_hourly(frame)) == len(frame)
    assert not reindex_hourly(frame)["temperature_c"].isna().any()


def test_split_gap_defaults_to_the_horizon():
    frame = hourly(np.arange(1000))
    for horizon in (24, 48):
        train, test = chronological_split(frame, horizon=horizon)
        assert len(frame) - len(train) - len(test) == horizon


def test_split_leaves_no_training_target_inside_the_test_period():
    frame = hourly(np.arange(2000))
    for horizon in (6, 24, 48):
        train, test = chronological_split(frame, horizon=horizon)
        latest = train["observed_at"].max() + pd.Timedelta(hours=horizon)
        assert latest <= test["observed_at"].min()


def test_short_gap_would_leak_at_long_horizons():
    frame = hourly(np.arange(2000))
    train, test = chronological_split(frame, gap_hours=HOURS_IN_DAY)
    latest = train["observed_at"].max() + pd.Timedelta(hours=48)
    assert latest > test["observed_at"].min()


def test_split_keeps_order():
    frame = hourly(np.arange(1000))
    train, test = chronological_split(frame, horizon=24)
    assert train["observed_at"].max() < test["observed_at"].min()
