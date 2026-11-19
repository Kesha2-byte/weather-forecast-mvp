import warnings
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from audit import load_cities
from features import HOURS_IN_DAY, load_series

warnings.filterwarnings("ignore")
TRAIN_FRACTION = 0.8
MAX_HORIZON = 48
SEASON = HOURS_IN_DAY


def make(values, trend):
    return ExponentialSmoothing(values, trend=trend, seasonal="add",
                                seasonal_periods=SEASON,
                                initialization_method="estimated")


def main():
    for city_id, city_name in load_cities(None):
        series = load_series(city_id).sort_values("observed_at").reset_index(drop=True)
        values = series["temperature_c"].astype(float).to_numpy()
        total = len(values)
        train_end = int(total * TRAIN_FRACTION)

        base = make(values[:train_end], "add").fit()
        a = float(base.params["smoothing_level"])
        b = float(base.params["smoothing_trend"])
        g = float(base.params["smoothing_seasonal"])
        base_none = make(values[:train_end], None).fit()
        a2 = float(base_none.params["smoothing_level"])
        g2 = float(base_none.params["smoothing_seasonal"])

        print("")
        print("=" * 76)
        print("%s  beta на обучении = %.6f" % (city_name, b))
        print("=" * 76)
        print("начало      нач.тренд   снос за 48ч   разн. прогнозов на 48ч   факт")
        print("----------  ----------  ------------  -----------------------  --------")

        trends, diffs = [], []
        for origin in list(range(train_end, total - MAX_HORIZON, 311))[:20]:
            r1 = make(values[:origin + 1], "add").fit(
                smoothing_level=a, smoothing_trend=b, smoothing_seasonal=g, optimized=False)
            r2 = make(values[:origin + 1], None).fit(
                smoothing_level=a2, smoothing_seasonal=g2, optimized=False)
            bt = float(r1.params["initial_trend"])
            f1 = float(np.asarray(r1.forecast(MAX_HORIZON))[-1])
            f2 = float(np.asarray(r2.forecast(MAX_HORIZON))[-1])
            trends.append(bt)
            diffs.append(f1 - f2)
            print("%-10d  %-10.5f  %-12.3f  %-23.3f  %-8.2f"
                  % (origin, bt, bt * MAX_HORIZON, f1 - f2, values[origin + MAX_HORIZON]))

        t = np.array(trends)
        d = np.array(diffs)
        print("")
        print("начальный тренд: среднее %.5f, размах от %.5f до %.5f" % (t.mean(), t.min(), t.max()))
        print("ожидаемый снос за 48 ч:  %.3f град" % (t.mean() * MAX_HORIZON))
        print("фактическая разница прогнозов: среднее %.3f, размах от %.3f до %.3f"
              % (d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
