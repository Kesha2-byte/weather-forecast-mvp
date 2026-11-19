import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from audit import baseline_offset, build, load_cities
from features import HOURS_IN_DAY, load_series

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

TRAIN_FRACTION = 0.8
MAX_HORIZON = 48
SEASON = HOURS_IN_DAY
BASE_LAGS = (0, 1, 24, 48)
REPORT_HORIZONS = (1, 3, 6, 12, 18, 24, 30, 36, 42, 48)


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def estimate_params(train_values):
    model = ExponentialSmoothing(
        train_values,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASON,
        initialization_method="estimated",
    )
    result = model.fit()
    return (
        float(result.params["smoothing_level"]),
        float(result.params["smoothing_trend"]),
        float(result.params["smoothing_seasonal"]),
    ), result


def forecast_from(values, upto, params, steps):
    level, trend, seasonal = params
    model = ExponentialSmoothing(
        values[: upto + 1],
        trend="add",
        seasonal="add",
        seasonal_periods=SEASON,
        initialization_method="estimated",
    )
    result = model.fit(
        smoothing_level=level,
        smoothing_trend=trend,
        smoothing_seasonal=seasonal,
        optimized=False,
    )
    return np.asarray(result.forecast(steps), dtype=float)


def regression_predictions(series, horizon, train_end_time, origin_times):
    offset = baseline_offset(horizon)
    lags = BASE_LAGS if offset in BASE_LAGS else BASE_LAGS + (offset,)
    dataset, names = build(series, horizon, lags)
    cutoff = train_end_time - pd.Timedelta(hours=horizon)
    train = dataset[dataset["observed_at"] <= cutoff]
    if len(train) < 1000:
        raise SystemExit("обучающая выборка слишком мала на горизонте %d" % horizon)
    model = LinearRegression()
    model.fit(train[names].to_numpy(), train["target"].to_numpy())
    wanted = dataset[dataset["observed_at"].isin(origin_times)]
    predicted = model.predict(wanted[names].to_numpy())
    return pd.DataFrame(
        {
            "observed_at": wanted["observed_at"].to_numpy(),
            "horizon": horizon,
            "regression": predicted,
        }
    )


def collect(city_id, city_name, step):
    series = load_series(city_id).sort_values("observed_at").reset_index(drop=True)
    values = series["temperature_c"].astype(float).to_numpy()
    times = series["observed_at"].to_numpy()
    total = len(values)
    train_end = int(total * TRAIN_FRACTION)
    train_end_time = pd.Timestamp(times[train_end])

    print("%s: ряд %d часов, обучение до %s" % (city_name, total, train_end_time))

    params, full_fit = estimate_params(values[:train_end])
    print("  параметры сглаживания: alpha=%.4f beta=%.4f gamma=%.4f" % params)

    origins = list(range(train_end, total - MAX_HORIZON, step))
    print("  начал прогноза: %d, шаг %d ч" % (len(origins), step))

    records = []
    for number, origin in enumerate(origins, 1):
        predicted = forecast_from(values, origin, params, MAX_HORIZON)
        for horizon in range(1, MAX_HORIZON + 1):
            offset = baseline_offset(horizon)
            records.append(
                {
                    "observed_at": pd.Timestamp(times[origin]),
                    "horizon": horizon,
                    "actual": values[origin + horizon],
                    "holt_winters": predicted[horizon - 1],
                    "baseline": values[origin - offset],
                }
            )
        if number % 25 == 0:
            print("  обработано начал: %d из %d" % (number, len(origins)))

    frame = pd.DataFrame(records)
    origin_times = pd.Index(frame["observed_at"].unique())

    parts = []
    for horizon in range(1, MAX_HORIZON + 1):
        parts.append(regression_predictions(series, horizon, train_end_time, origin_times))
    regression = pd.concat(parts, ignore_index=True)

    frame = frame.merge(regression, on=["observed_at", "horizon"], how="inner")
    frame["city"] = city_name
    return frame, full_fit, series


def by_horizon(frame):
    rows = []
    for horizon, part in frame.groupby("horizon"):
        rows.append(
            {
                "horizon": int(horizon),
                "n": len(part),
                "mae_base": mae(part["actual"], part["baseline"]),
                "mae_reg": mae(part["actual"], part["regression"]),
                "mae_hw": mae(part["actual"], part["holt_winters"]),
                "rmse_base": rmse(part["actual"], part["baseline"]),
                "rmse_reg": rmse(part["actual"], part["regression"]),
                "rmse_hw": rmse(part["actual"], part["holt_winters"]),
            }
        )
    out = pd.DataFrame(rows)
    out["rel_reg"] = out["mae_reg"] / out["mae_base"]
    out["rel_hw"] = out["mae_hw"] / out["mae_base"]
    return out


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("%.3f" % value)
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return out


def screen_table(table):
    lines = [
        "гор.   n       MAE база  MAE регр  MAE Х-У   отн.регр  отн.Х-У   лучший",
        "-----  ------  --------  --------  --------  --------  --------  ----------",
    ]
    for _, r in table.iterrows():
        best = min(
            [("база", r["mae_base"]), ("регрессия", r["mae_reg"]), ("Хольт-Уинтерс", r["mae_hw"])],
            key=lambda pair: pair[1],
        )[0]
        lines.append(
            "%-5d  %-6d  %-8.3f  %-8.3f  %-8.3f  %-8.3f  %-8.3f  %s"
            % (r["horizon"], r["n"], r["mae_base"], r["mae_reg"], r["mae_hw"],
               r["rel_reg"], r["rel_hw"], best)
        )
    return lines


def make_plots(tables, fits, series_map, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    cities = list(tables.keys())

    fig, axes = plt.subplots(1, len(cities), figsize=(6.6 * len(cities), 4.4), squeeze=False)
    for ax, city in zip(axes[0], cities):
        t = tables[city]
        ax.plot(t["horizon"], t["mae_base"], "--", label="бейзлайн")
        ax.plot(t["horizon"], t["mae_reg"], "-", label="регрессия")
        ax.plot(t["horizon"], t["mae_hw"], "-", label="Хольт-Уинтерс")
        ax.set_title(city)
        ax.set_xlabel("Горизонт прогноза, часов")
        ax.set_ylabel("MAE, градусы")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hw-mae-by-horizon.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(cities), figsize=(6.6 * len(cities), 4.4), squeeze=False)
    for ax, city in zip(axes[0], cities):
        t = tables[city]
        ax.plot(t["horizon"], t["rel_reg"], "-", marker="o", ms=3, label="регрессия")
        ax.plot(t["horizon"], t["rel_hw"], "-", marker="s", ms=3, label="Хольт-Уинтерс")
        ax.axhline(1.0, color="#888888", linewidth=1)
        ax.set_title(city)
        ax.set_xlabel("Горизонт прогноза, часов")
        ax.set_ylabel("MAE метода / MAE бейзлайна")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hw-ratio-by-horizon.png"), dpi=140)
    plt.close(fig)

    city = cities[0]
    fit = fits[city]
    window = slice(-14 * 24, None)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(np.asarray(fit.level)[window])
    axes[0].set_ylabel("Уровень")
    axes[1].plot(np.asarray(fit.trend)[window])
    axes[1].set_ylabel("Тренд")
    axes[2].plot(np.asarray(fit.season)[window])
    axes[2].set_ylabel("Сезонность")
    axes[2].set_xlabel("Часы, последние две недели обучающей выборки")
    axes[0].set_title("%s: разложение ряда моделью Хольта-Уинтерса" % city)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hw-decomposition.png"), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", type=int, default=None)
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument("--out", default=os.path.join("..", "docs", "holt-winters.md"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    cities = load_cities(args.cities)
    if not cities:
        raise SystemExit("города не найдены")

    report = ["# Сравнение методов прогнозирования", "",
              "Отчёт сформирован скриптом scripts/holtwinters.py.", "",
              "Параметры сглаживания оценены только на обучающей части ряда. На каждом начале",
              "прогноза состояние модели пересобирается по наблюдениям до этого момента с уже",
              "зафиксированными параметрами, поэтому заглядывания в будущее нет.", ""]

    tables = {}
    fits = {}
    series_map = {}

    for city_id, city_name in cities:
        frame, fit, series = collect(city_id, city_name, args.step)
        table = by_horizon(frame)
        tables[city_name] = table
        fits[city_name] = fit
        series_map[city_name] = series

        print("")
        print("=" * 78)
        print("%s" % city_name)
        print("=" * 78)
        for line in screen_table(table[table["horizon"].isin(REPORT_HORIZONS)]):
            print(line)

        wins_reg = int((table["mae_reg"] < table["mae_base"]).sum())
        wins_hw = int((table["mae_hw"] < table["mae_base"]).sum())
        better = int((table["mae_hw"] < table["mae_reg"]).sum())
        print("")
        print("Регрессия лучше бейзлайна на %d горизонтах из %d" % (wins_reg, len(table)))
        print("Хольт-Уинтерс лучше бейзлайна на %d горизонтах из %d" % (wins_hw, len(table)))
        print("Хольт-Уинтерс лучше регрессии на %d горизонтах из %d" % (better, len(table)))

        report += ["## %s" % city_name, ""]
        report += md_table(
            ["Горизонт", "Пар", "MAE бейзлайн", "MAE регрессия", "MAE Хольт-Уинтерс",
             "RMSE бейзлайн", "RMSE регрессия", "RMSE Хольт-Уинтерс", "Отн. регрессия", "Отн. Х-У"],
            table[table["horizon"].isin(REPORT_HORIZONS)][
                ["horizon", "n", "mae_base", "mae_reg", "mae_hw",
                 "rmse_base", "rmse_reg", "rmse_hw", "rel_reg", "rel_hw"]
            ].to_numpy().tolist(),
        )
        report += ["",
                   "Регрессия лучше бейзлайна на %d горизонтах из %d." % (wins_reg, len(table)),
                   "Хольт-Уинтерс лучше бейзлайна на %d горизонтах из %d." % (wins_hw, len(table)),
                   "Хольт-Уинтерс лучше регрессии на %d горизонтах из %d." % (better, len(table)),
                   ""]

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print("")
    print("Отчёт записан: %s" % out_path)

    if args.plots:
        make_plots(tables, fits, series_map,
                   os.path.join(os.path.dirname(out_path) or ".", "figures"))
        print("Графики записаны в каталог figures")


if __name__ == "__main__":
    main()
