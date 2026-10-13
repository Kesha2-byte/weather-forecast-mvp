import numpy as np
import pandas as pd
import pytest

from features import (
    add_baseline,
    add_cyclical_features,
    add_lag_features,
    add_target,
    chronological_split,
)


def make_series(hours, values=None):
    index = pd.date_range("2024-01-01 00:00", periods=hours, freq="h", tz="Asia/Almaty")
    if values is None:
        values = np.arange(hours, dtype=float)
    return pd.DataFrame({"observed_at": index, "temperature_c": values})


def make_daily_cycle(days):
    hours = days * 24
    index = pd.date_range("2024-01-01 00:00", periods=hours, freq="h", tz="Asia/Almaty")
    cycle = 10 + 5 * np.sin(2 * np.pi * np.arange(hours) / 24)
    return pd.DataFrame({"observed_at": index, "temperature_c": cycle})


class TestLagFeatures:
    def test_lag_shifts_values_back(self):
        df = add_lag_features(make_series(50), lags=(1, 24))

        assert df.loc[24, "lag_1"] == 23.0
        assert df.loc[24, "lag_24"] == 0.0
        assert df.loc[49, "lag_24"] == 25.0

    def test_lag_zero_equals_current_value(self):
        df = add_lag_features(make_series(10), lags=(0,))

        assert (df["lag_0"] == df["temperature_c"]).all()

    def test_first_rows_have_no_history(self):
        df = add_lag_features(make_series(50), lags=(24,))

        assert df["lag_24"].isna().sum() == 24
        assert df.loc[23, "lag_24"] != df.loc[23, "lag_24"]


class TestTarget:
    def test_target_points_to_future(self):
        df = add_target(make_series(50), horizon=24)

        assert df.loc[0, "target"] == 24.0
        assert df.loc[10, "target"] == 34.0

    def test_last_rows_have_no_future(self):
        df = add_target(make_series(50), horizon=24)

        assert df["target"].isna().sum() == 24


class TestBaseline:
    @pytest.mark.parametrize("horizon,expected_offset", [
        (1, 23),
        (12, 12),
        (24, 0),
        (25, 23),
        (48, 0),
    ])
    def test_offset_matches_horizon(self, horizon, expected_offset):
        df = add_baseline(make_series(100), horizon)
        expected = make_series(100)["temperature_c"].shift(expected_offset)

        pd.testing.assert_series_equal(
            df["baseline"], expected, check_names=False
        )

    def test_exact_on_perfectly_periodic_series(self):
        df = make_daily_cycle(days=10)
        df = add_target(df, horizon=24)
        df = add_baseline(df, horizon=24)
        df = df.dropna()

        error = (df["target"] - df["baseline"]).abs().max()
        assert error < 1e-9


class TestCyclicalFeatures:
    def test_midnight_and_noon(self):
        df = add_cyclical_features(make_series(24))

        assert df.loc[0, "hour_sin"] == pytest.approx(0.0, abs=1e-12)
        assert df.loc[0, "hour_cos"] == pytest.approx(1.0, abs=1e-12)
        assert df.loc[12, "hour_sin"] == pytest.approx(0.0, abs=1e-12)
        assert df.loc[12, "hour_cos"] == pytest.approx(-1.0, abs=1e-12)

    def test_cycle_closes(self):
        df = add_cyclical_features(make_series(48))

        assert df.loc[0, "hour_sin"] == pytest.approx(df.loc[24, "hour_sin"])
        assert df.loc[0, "hour_cos"] == pytest.approx(df.loc[24, "hour_cos"])

    def test_values_stay_in_unit_circle(self):
        df = add_cyclical_features(make_series(24))
        radius = df["hour_sin"] ** 2 + df["hour_cos"] ** 2

        assert radius.sub(1.0).abs().max() < 1e-12


class TestChronologicalSplit:
    def test_train_strictly_precedes_test(self):
        dataset = make_series(1000)
        train, test = chronological_split(dataset)

        assert train["observed_at"].max() < test["observed_at"].min()

    def test_gap_is_applied(self):
        dataset = make_series(1000)
        train, test = chronological_split(dataset, test_size=0.2, gap_hours=24)

        assert len(train) == 800 - 24
        assert len(test) == 200

    def test_no_rows_are_duplicated(self):
        dataset = make_series(1000)
        train, test = chronological_split(dataset)
        combined = pd.concat([train, test])

        assert combined["observed_at"].duplicated().sum() == 0
