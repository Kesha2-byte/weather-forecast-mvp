import argparse
import os

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression

from features import DAYS_IN_YEAR, HOURS_IN_DAY, chronological_split, load_series

try:
    from features import DEFAULT_LAGS
except ImportError:
    DEFAULT_LAGS = (0, 1, 24, 48)

np.seterr(all="ignore")

DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/weather")
CITY_SQL = "SELECT * FROM cities"
HORIZONS = (1, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48)
TEST_SIZE = 0.2
GAP_HOURS = HOURS_IN_DAY
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def baseline_offset(horizon):
    return -(-horizon // HOURS_IN_DAY) * HOURS_IN_DAY - horizon


def load_cities(city_ids=None):
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(CITY_SQL)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    id_col = next((c for c in columns if c.lower().endswith("id")), columns[0])
    name_col = next((c for c in columns if "name" in c.lower()), columns[1])
    i_id = columns.index(id_col)
    i_name = columns.index(name_col)
    print("Таблица cities: столбцы %s, использую %s и %s" % (columns, id_col, name_col))
    cities = [(int(r[i_id]), str(r[i_name])) for r in rows]
    if city_ids:
        cities = [c for c in cities if c[0] in set(city_ids)]
    return sorted(cities)


def build(series, horizon, lags):
    df = series.loc[:, ["observed_at", "temperature_c"]].copy()
    df = df.sort_values("observed_at").reset_index(drop=True)
    temp = df["temperature_c"].astype(float)
    df["temperature_c"] = temp

    names = []
    for lag in lags:
        column = "lag_%d" % lag
        df[column] = temp.shift(lag)
        names.append(column)

    hour = df["observed_at"].dt.hour
    doy = df["observed_at"].dt.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * hour / HOURS_IN_DAY)
    df["cos_hour"] = np.cos(2 * np.pi * hour / HOURS_IN_DAY)
    df["sin_doy"] = np.sin(2 * np.pi * doy / DAYS_IN_YEAR)
    df["cos_doy"] = np.cos(2 * np.pi * doy / DAYS_IN_YEAR)
    names += ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]

    df["baseline"] = temp.shift(baseline_offset(horizon))
    df["target"] = temp.shift(-horizon)
    df["target_time"] = df["observed_at"].shift(-horizon)

    keep = ["observed_at", "target_time"] + names + ["baseline", "target"]
    return df.loc[:, keep].dropna().reset_index(drop=True), names


def measure(series, horizon, lags):
    dataset, names = build(series, horizon, lags)
    train, test = chronological_split(dataset, test_size=TEST_SIZE, gap_hours=GAP_HOURS)
    model = LinearRegression()
    model.fit(train[names].to_numpy(), train["target"].to_numpy())
    predicted = model.predict(test[names].to_numpy())
    actual = test["target"].to_numpy()
    base = test["baseline"].to_numpy()
    mae_model = float(np.mean(np.abs(actual - predicted)))
    mae_base = float(np.mean(np.abs(actual - base)))
    return mae_model / mae_base, mae_model, mae_base


def section(title, lines):
    out = ["", "=" * 78, title, "=" * 78, ""]
    out += lines
    return out


def table(headers, rows, widths):
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return [line, sep] + body


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", type=int, default=None)
    parser.add_argument("--out", default=os.path.join("..", "docs", "audit-report.md"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cities = load_cities(args.cities)
    if not cities:
        raise SystemExit("города не найдены")

    report = ["# Аудит методики измерения", "",
              "Отчёт сформирован скриптом scripts/audit.py на данных рабочей базы.", ""]
    screen = []

    base_lags = tuple(DEFAULT_LAGS)

    screen += section(
        "ДЕФЕКТ 1, ЧАСТЬ А. Покрывает ли набор признаков то значение, которым пользуется бейзлайн",
        table(
            ["горизонт", "лаг бейзлайна", "есть в наборе"],
            [[str(h), str(baseline_offset(h)), "ДА" if baseline_offset(h) in base_lags else "нет"]
             for h in HORIZONS],
            [10, 15, 14],
        ),
    )
    screen += ["", "Набор лагов: %s" % (", ".join(str(l) for l in base_lags)), ""]

    report += ["## Дефект 1. Покрытие бейзлайна набором признаков", "",
               "Бейзлайн на горизонте h использует значение, отстоящее на 24*ceil(h/24) - h часов назад.", ""]
    report += md_table(
        ["Горизонт", "Лаг бейзлайна", "Есть в наборе"],
        [[h, baseline_offset(h), "да" if baseline_offset(h) in base_lags else "нет"] for h in HORIZONS],
    )
    report += [""]

    plot_data = {}

    for city_id, city_name in cities:
        series = load_series(city_id)
        rows = []
        for h in HORIZONS:
            off = baseline_offset(h)
            extended = base_lags if off in base_lags else base_lags + (off,)
            without, mae_w, mae_b = measure(series, h, base_lags)
            with_lag, mae_x, _ = measure(series, h, extended)
            verdict = "без изменений" if off in base_lags else "%+.1f %%" % ((with_lag / without - 1) * 100)
            rows.append([h, off, round(mae_b, 3), round(mae_w, 3), round(without, 3),
                         round(mae_x, 3), round(with_lag, 3), verdict])
        plot_data[city_name] = rows

        screen += section(
            "ДЕФЕКТ 1, ЧАСТЬ Б. %s - замер с недостающим лагом и без него" % city_name,
            table(
                ["гор.", "лаг", "MAE база", "MAE без", "отн. без", "MAE с", "отн. с", "изменение"],
                [[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows],
                [5, 4, 9, 8, 9, 8, 8, 14],
            ),
        )

        report += ["## Дефект 1. Замер: %s" % city_name, ""]
        report += md_table(
            ["Горизонт", "Лаг бейзлайна", "MAE бейзлайна", "MAE модели без лага",
             "Отношение без", "MAE модели с лагом", "Отношение с", "Изменение"],
            rows,
        )
        report += [""]

    screen += section("ДЕФЕКТ 2. Зазор между обучением и тестом", [])
    report += ["## Дефект 2. Зазор между обучением и тестом", ""]

    gap_rows = []
    for city_id, city_name in cities:
        series = load_series(city_id)
        for h in (24, 48):
            dataset, _ = build(series, h, base_lags)
            train, test = chronological_split(dataset, test_size=TEST_SIZE, gap_hours=GAP_HOURS)
            last_train = train.iloc[-1]
            test_start = test.iloc[0]["observed_at"]
            leaked = int((train["target_time"] >= test_start).sum())
            gap_rows.append([
                city_name, h,
                str(last_train["observed_at"]),
                str(last_train["target_time"]),
                str(test_start),
                leaked,
                len(test),
            ])

    screen += table(
        ["город", "гор.", "посл. строка обучения", "её целевое время", "начало теста", "утекло", "всего"],
        gap_rows,
        [8, 5, 26, 26, 26, 7, 7],
    )
    screen += ["", "Зазор задан равным %d ч. Утечка возникает там, где горизонт превышает зазор." % GAP_HOURS, ""]

    report += ["Зазор задан равным %d часам. Строка обучения несёт ответ, отстоящий на величину горизонта," % GAP_HOURS,
               "поэтому при горизонте больше зазора целевые значения попадают в тестовый период.", ""]
    report += md_table(
        ["Город", "Горизонт", "Последняя строка обучения", "Её целевое время", "Начало теста",
         "Строк с утечкой", "Всего тестовых"],
        gap_rows,
    )
    report += [""]

    screen += section("ДЕФЕКТ 3. Сезонный состав тестовой части", [])
    report += ["## Дефект 3. Сезонный состав тестовой части", ""]

    for city_id, city_name in cities:
        series = load_series(city_id)
        dataset, _ = build(series, 24, base_lags)
        train, test = chronological_split(dataset, test_size=TEST_SIZE, gap_hours=GAP_HOURS)
        times = pd.to_datetime(test["target_time"])
        counts = times.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
        total = int(counts.sum())
        rows = [[MONTHS[m - 1], int(counts[m]), "%.1f %%" % (100.0 * counts[m] / total),
                 "#" * int(round(40.0 * counts[m] / max(1, counts.max())))] for m in range(1, 13)]
        screen += ["", "%s: тест с %s по %s, всего %d часов" % (
            city_name, times.min(), times.max(), total), ""]
        screen += table(["месяц", "часов", "доля", ""], rows, [6, 7, 8, 42])
        missing = [MONTHS[m - 1] for m in range(1, 13) if counts[m] == 0]
        if missing:
            screen += ["", "Месяцы, полностью отсутствующие в тесте: %s" % ", ".join(missing)]

        report += ["### %s" % city_name, "",
                   "Тестовая часть: с %s по %s, %d часов." % (times.min(), times.max(), total), ""]
        report += md_table(["Месяц", "Часов", "Доля"],
                           [[r[0], r[1], r[2]] for r in rows])
        if missing:
            report += ["", "Месяцы, полностью отсутствующие в тесте: %s." % ", ".join(missing)]
        report += [""]

    text = "\n".join(screen)
    print(text)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print("")
    print("Отчёт записан: %s" % out_path)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(plot_data), figsize=(6.5 * len(plot_data), 4.4), squeeze=False)
        for ax, (city_name, rows) in zip(axes[0], plot_data.items()):
            hs = [r[0] for r in rows]
            ax.plot(hs, [r[4] for r in rows], "--", marker="s", label="набор без лага бейзлайна")
            ax.plot(hs, [r[6] for r in rows], "-", marker="o", label="набор с лагом бейзлайна")
            ax.axhline(1.0, color="#888888", linewidth=1)
            ax.set_title(city_name)
            ax.set_xlabel("Горизонт прогноза, часов")
            ax.set_ylabel("MAE модели / MAE бейзлайна")
            ax.legend()
            ax.grid(alpha=0.25)
        fig.tight_layout()
        figure_dir = os.path.join(os.path.dirname(out_path) or ".", "figures")
        os.makedirs(figure_dir, exist_ok=True)
        path = os.path.join(figure_dir, "audit-lag-coverage.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print("График записан: %s" % path)


if __name__ == "__main__":
    main()
