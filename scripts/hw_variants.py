import argparse
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from audit import baseline_offset, load_cities
from features import HOURS_IN_DAY, load_series

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

TRAIN_FRACTION = 0.8
MAX_HORIZON = 48
SEASON = HOURS_IN_DAY
REPORT = (1, 3, 6, 12, 18, 24, 30, 36, 42, 48)
VARIANTS = ("add", None)


def label_of(trend):
    return "с трендом" if trend else "без тренда"


def make(values, trend):
    return ExponentialSmoothing(
        values,
        trend=trend,
        seasonal="add",
        seasonal_periods=SEASON,
        initialization_method="estimated",
    )


def estimate(train_values, trend):
    result = make(train_values, trend).fit()
    params = {"smoothing_level": float(result.params["smoothing_level"]),
              "smoothing_seasonal": float(result.params["smoothing_seasonal"])}
    info = "alpha=%.4f gamma=%.4f" % (params["smoothing_level"], params["smoothing_seasonal"])
    if trend:
        params["smoothing_trend"] = float(result.params["smoothing_trend"])
        initial = result.params.get("initial_trend")
        initial = float(initial) if initial is not None else float("nan")
        info += " beta=%.4f начальный тренд=%.5f град/ч" % (params["smoothing_trend"], initial)
        info += " (снос за 48 ч: %+.2f град)" % (initial * MAX_HORIZON)
    return params, info


def forecast(values, upto, trend, params, steps):
    result = make(values[: upto + 1], trend).fit(optimized=False, **params)
    return np.asarray(result.forecast(steps), dtype=float)


def run_city(city_id, city_name, step):
    series = load_series(city_id).sort_values("observed_at").reset_index(drop=True)
    values = series["temperature_c"].astype(float).to_numpy()
    total = len(values)
    train_end = int(total * TRAIN_FRACTION)
    origins = list(range(train_end, total - MAX_HORIZON, step))
    hours = sorted(set(series["observed_at"].dt.hour.to_numpy()[origins].tolist()))

    print("")
    print("=" * 78)
    print("%s: %d начал прогноза, шаг %d ч" % (city_name, len(origins), step))
    print("Часы суток, попавшие в начала прогноза: %d разных" % len(hours))
    print("=" * 78)

    fitted = {}
    for trend in VARIANTS:
        params, info = estimate(values[:train_end], trend)
        fitted[trend] = params
        print("  %-11s %s" % (label_of(trend), info))

    records = []
    for number, origin in enumerate(origins, 1):
        predicted = {t: forecast(values, origin, t, fitted[t], MAX_HORIZON) for t in VARIANTS}
        for horizon in range(1, MAX_HORIZON + 1):
            records.append({
                "horizon": horizon,
                "actual": values[origin + horizon],
                "baseline": values[origin - baseline_offset(horizon)],
                "with_trend": predicted["add"][horizon - 1],
                "no_trend": predicted[None][horizon - 1],
            })
        if number % 50 == 0:
            print("  обработано начал: %d из %d" % (number, len(origins)))

    frame = pd.DataFrame(records)
    rows = []
    for horizon, part in frame.groupby("horizon"):
        rows.append({
            "horizon": int(horizon),
            "n": len(part),
            "base": float(np.mean(np.abs(part["actual"] - part["baseline"]))),
            "with_trend": float(np.mean(np.abs(part["actual"] - part["with_trend"]))),
            "no_trend": float(np.mean(np.abs(part["actual"] - part["no_trend"]))),
        })
    table = pd.DataFrame(rows)

    print("")
    print("гор.   n       MAE база  MAE с тр.  MAE без тр.  отн. с тр.  отн. без тр.")
    print("-----  ------  --------  ---------  -----------  ----------  ------------")
    for _, r in table[table["horizon"].isin(REPORT)].iterrows():
        print("%-5d  %-6d  %-8.3f  %-9.3f  %-11.3f  %-10.3f  %-12.3f"
              % (r["horizon"], r["n"], r["base"], r["with_trend"], r["no_trend"],
                 r["with_trend"] / r["base"], r["no_trend"] / r["base"]))

    print("")
    print("Лучше бейзлайна с трендом:  %d из %d" % (int((table["with_trend"] < table["base"]).sum()), len(table)))
    print("Лучше бейзлайна без тренда: %d из %d" % (int((table["no_trend"] < table["base"]).sum()), len(table)))
    print("Без тренда лучше, чем с трендом: %d из %d" % (int((table["no_trend"] < table["with_trend"]).sum()), len(table)))
    table["city"] = city_name
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", type=int, default=None)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--out", default=os.path.join("..", "docs", "hw-variants.md"))
    args = parser.parse_args()

    if args.step % HOURS_IN_DAY == 0:
        print("ВНИМАНИЕ: шаг кратен 24, все начала прогноза попадут на один час суток")

    tables = [run_city(cid, name, args.step) for cid, name in load_cities(args.cities)]
    combined = pd.concat(tables, ignore_index=True)

    lines = ["# Хольт-Уинтерс: с трендом и без тренда", "",
             "Шаг между началами прогноза %d ч." % args.step, ""]
    for city, part in combined.groupby("city", sort=False):
        lines += ["## %s" % city, "",
                  "| Горизонт | Пар | MAE бейзлайн | MAE с трендом | MAE без тренда |",
                  "| --- | --- | --- | --- | --- |"]
        for _, r in part[part["horizon"].isin(REPORT)].iterrows():
            lines.append("| %d | %d | %.3f | %.3f | %.3f |"
                         % (r["horizon"], r["n"], r["base"], r["with_trend"], r["no_trend"]))
        lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("")
    print("Отчёт записан: %s" % args.out)


if __name__ == "__main__":
    main()
