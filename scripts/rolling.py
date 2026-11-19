import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from features import build_dataset, lags_for

np.seterr(all="ignore")

DEFAULT_HORIZONS = (6, 12, 24, 48)
INITIAL_SHARE = 0.5
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

CITY_SQL = "SELECT * FROM cities"


def load_cities(city_ids=None):
    import psycopg2

    url = os.environ.get("DATABASE_URL", "postgresql://localhost/weather")
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(CITY_SQL)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    id_col = next((c for c in columns if c.lower().endswith("id")), columns[0])
    name_col = next((c for c in columns if "name" in c.lower()), columns[1])
    i_id, i_name = columns.index(id_col), columns.index(name_col)
    cities = [(int(r[i_id]), str(r[i_name])) for r in rows]
    if city_ids:
        cities = [c for c in cities if c[0] in set(city_ids)]
    return sorted(cities)


def months_of(times):
    present = sorted(set(pd.to_datetime(times).dt.month.tolist()))
    return " ".join(MONTHS[m - 1] for m in present)


def evaluate_fold(dataset, names, train_end, test_start, test_end, horizon):
    train = dataset.iloc[: max(0, train_end - horizon)]
    test = dataset.iloc[test_start:test_end]
    if len(train) < 2000 or len(test) < 200:
        return None
    model = LinearRegression()
    model.fit(train[names].to_numpy(), train["target"].to_numpy())
    predicted = model.predict(test[names].to_numpy())
    actual = test["target"].to_numpy()
    base = test["baseline"].to_numpy()
    mae_model = float(np.mean(np.abs(actual - predicted)))
    mae_base = float(np.mean(np.abs(actual - base)))
    rmse_model = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    rmse_base = float(np.sqrt(np.mean((actual - base) ** 2)))
    return {
        "n_train": len(train),
        "n_test": len(test),
        "from": test["observed_at"].iloc[0],
        "to": test["observed_at"].iloc[-1],
        "months": months_of(test["observed_at"]),
        "mae_base": mae_base,
        "mae_model": mae_model,
        "ratio_mae": mae_model / mae_base,
        "rmse_base": rmse_base,
        "rmse_model": rmse_model,
        "ratio_rmse": rmse_model / rmse_base,
    }


def run_city(city_id, city_name, horizons, folds):
    rows = []
    for horizon in horizons:
        dataset, names = build_dataset(city_id, horizon)
        total = len(dataset)
        start = int(total * INITIAL_SHARE)
        width = (total - start) // folds

        print("")
        print("=" * 100)
        print("%s, горизонт %d ч, лаги %s" % (city_name, horizon, lags_for(horizon)))
        print("=" * 100)
        print("фолд  период теста                            месяцы                "
              "обуч.   тест    MAE база  MAE мод.  отн.")
        print("----  --------------------------------------  --------------------  "
              "------  ------  --------  --------  -----")

        for index in range(folds):
            test_start = start + index * width
            test_end = test_start + width if index < folds - 1 else total
            result = evaluate_fold(dataset, names, test_start, test_start, test_end, horizon)
            if result is None:
                continue
            result.update({"city": city_name, "horizon": horizon, "fold": index + 1})
            rows.append(result)
            print("%-4d  %s .. %s  %-20s  %-6d  %-6d  %-8.3f  %-8.3f  %-5.3f"
                  % (index + 1,
                     str(result["from"])[:16], str(result["to"])[:16],
                     result["months"], result["n_train"], result["n_test"],
                     result["mae_base"], result["mae_model"], result["ratio_mae"]))

        part = [r for r in rows if r["horizon"] == horizon]
        ratios = [r["ratio_mae"] for r in part]
        if ratios:
            print("")
            print("  среднее отношение по фолдам: %.3f   размах: %.3f .. %.3f   "
                  "фолдов с выигрышем: %d из %d"
                  % (float(np.mean(ratios)), min(ratios), max(ratios),
                     sum(1 for v in ratios if v < 1.0), len(ratios)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--out", default=os.path.join("..", "docs", "rolling.md"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    frames = [run_city(cid, name, args.horizons, args.folds)
              for cid, name in load_cities(args.cities)]
    combined = pd.concat(frames, ignore_index=True)

    lines = ["# Оценка со скользящим началом", "",
             "Обучение всегда строго раньше проверки. Граница сдвигается вперёд, поэтому",
             "каждый фолд проверяет модель в своём отрезке года.", "",
             "Число фолдов: %d. Начальная доля обучения: %.0f %%." % (args.folds, INITIAL_SHARE * 100), ""]

    for (city, horizon), part in combined.groupby(["city", "horizon"], sort=False):
        lines += ["## %s, горизонт %d ч" % (city, horizon), "",
                  "| Фолд | Период проверки | Месяцы | MAE бейзлайн | MAE модель | Отношение MAE | Отношение RMSE |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in part.iterrows():
            lines.append("| %d | %s .. %s | %s | %.3f | %.3f | %.3f | %.3f |"
                         % (r["fold"], str(r["from"])[:10], str(r["to"])[:10], r["months"],
                            r["mae_base"], r["mae_model"], r["ratio_mae"], r["ratio_rmse"]))
        lines += ["",
                  "Среднее отношение MAE по фолдам: %.3f, размах %.3f .. %.3f."
                  % (part["ratio_mae"].mean(), part["ratio_mae"].min(), part["ratio_mae"].max()),
                  ""]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("")
    print("Отчёт записан: %s" % args.out)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cities = list(dict.fromkeys(combined["city"]))
        fig, axes = plt.subplots(1, len(cities), figsize=(6.6 * len(cities), 4.4), squeeze=False)
        for ax, city in zip(axes[0], cities):
            part = combined[combined["city"] == city]
            for horizon, sub in part.groupby("horizon"):
                ax.plot(sub["fold"], sub["ratio_mae"], marker="o", label="%d ч" % horizon)
            ax.axhline(1.0, color="#888888", linewidth=1)
            ax.set_title(city)
            ax.set_xlabel("Номер фолда (время идёт вперёд)")
            ax.set_ylabel("MAE модели / MAE бейзлайна")
            ax.legend()
            ax.grid(alpha=0.25)
        fig.tight_layout()
        folder = os.path.join(os.path.dirname(args.out) or ".", "figures")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "rolling-ratio-by-fold.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print("График записан: %s" % path)


if __name__ == "__main__":
    main()
