import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from features import build_dataset, chronological_split
from models import mae, rmse

np.seterr(all="ignore")

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ALMATY = "#2a78d6"
ASTANA = "#eb6834"
NEUTRAL = "#898781"

CITIES = [(1, "Алматы", ALMATY), (2, "Астана", ASTANA)]
ALL_HORIZONS = list(range(1, 49))
KEY_HORIZONS = (1, 6, 12, 24, 48)
HYPOTHESIS_THRESHOLD = 0.85


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def quiet_legend(ax, **kwargs):
    legend = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return legend


def evaluate(city_id, horizon):
    dataset, feature_names = build_dataset(city_id, horizon)
    train, test = chronological_split(dataset)

    model = LinearRegression()
    model.fit(train[feature_names].to_numpy(), train["target"].to_numpy())

    actual = test["target"].to_numpy()
    predicted = model.predict(test[feature_names].to_numpy())
    base = test["baseline"].to_numpy()

    return {
        "baseline_mae": mae(actual, base),
        "baseline_rmse": rmse(actual, base),
        "model_mae": mae(actual, predicted),
        "model_rmse": rmse(actual, predicted),
        "test": test,
        "predicted": predicted,
    }


def collect(horizons):
    return {
        city_name: {h: evaluate(city_id, h) for h in horizons}
        for city_id, city_name, _ in CITIES
    }


def separated_label_positions(values, min_gap):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = list(values)
    for position in range(1, len(order)):
        lower, upper = order[position - 1], order[position]
        if adjusted[upper] - adjusted[lower] < min_gap:
            adjusted[upper] = adjusted[lower] + min_gap
    return adjusted


def plot_ratio_curve(results, horizons):
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    style_axes(ax)

    ax.axhline(1.0, color=NEUTRAL, linewidth=1.2, zorder=1)
    ax.axhline(HYPOTHESIS_THRESHOLD, color=NEUTRAL, linewidth=1.2,
               linestyle="--", zorder=1)
    ax.axvline(24.5, color=AXIS, linewidth=1.0, linestyle=":", zorder=1)

    ax.text(1.5, 1.015, "паритет с бейзлайном", color=INK_SECONDARY,
            fontsize=9, va="bottom", ha="left")
    ax.text(1.5, 0.865, "порог гипотезы 0.85", color=INK_SECONDARY,
            fontsize=9, va="bottom", ha="left")

    finals = []
    for _, city_name, color in CITIES:
        ratios = [results[city_name][h]["model_mae"] / results[city_name][h]["baseline_mae"]
                  for h in horizons]
        ax.plot(horizons, ratios, color=color, linewidth=2.0, zorder=3, label=city_name)
        ax.plot([24], [ratios[horizons.index(24)]], "o", color=color, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        finals.append(ratios[-1])

    for (_, city_name, color), y in zip(CITIES, separated_label_positions(finals, 0.055)):
        ax.text(49.2, y, city_name, color=color, fontsize=10, va="center", ha="left")

    ax.annotate(
        "бейзлайн переходит\nна позапрошлые сутки",
        xy=(24.5, 0.56), xytext=(29, 0.40),
        color=INK_SECONDARY, fontsize=9,
        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1.0),
    )

    ax.set_xlabel("Горизонт прогноза, часов", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("MAE модели / MAE бейзлайна", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Относительная точность модели по горизонту прогноза",
                 color=INK_PRIMARY, fontsize=13, pad=16, loc="left")
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xlim(0, 58)
    ax.set_ylim(0.12, 1.42)

    quiet_legend(ax, loc="lower right", ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "mae-ratio-by-horizon.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_absolute_errors(results, horizons):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE, sharey=True)

    for ax, (_, city_name, color) in zip(axes, CITIES):
        style_axes(ax)
        ax.axvline(24.5, color=AXIS, linewidth=1.0, linestyle=":", zorder=1)

        base = [results[city_name][h]["baseline_mae"] for h in horizons]
        model = [results[city_name][h]["model_mae"] for h in horizons]

        ax.plot(horizons, base, color=NEUTRAL, linewidth=2.0, zorder=3, label="Бейзлайн")
        ax.plot(horizons, model, color=color, linewidth=2.0, zorder=3, label="Модель")

        for value, text, text_color in zip(
            separated_label_positions([base[-1], model[-1]], 0.18),
            ("бейзлайн", "модель"),
            (INK_SECONDARY, color),
        ):
            ax.text(49.5, value, text, color=text_color, fontsize=9,
                    va="center", ha="left")

        ax.set_title(city_name, color=INK_PRIMARY, fontsize=12, pad=10, loc="left")
        ax.set_xlabel("Горизонт прогноза, часов", color=INK_SECONDARY, fontsize=10)
        ax.set_xticks([1, 12, 24, 36, 48])
        ax.set_xlim(0, 64)

    axes[0].set_ylabel("MAE, °C", color=INK_SECONDARY, fontsize=10)
    quiet_legend(axes[0], loc="lower right")

    fig.suptitle("Абсолютная ошибка прогноза температуры",
                 color=INK_PRIMARY, fontsize=13, x=0.012, ha="left", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "mae-absolute-by-horizon.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_forecast_vs_actual(city_id=1, horizon=24, days=7):
    result = evaluate(city_id, horizon)
    test = result["test"]
    predicted = result["predicted"]

    window = days * 24
    start = len(test) // 2
    sl = slice(start, start + window)

    times = test["observed_at"].iloc[sl].to_numpy()
    actual = test["target"].to_numpy()[sl]
    model_pred = predicted[sl]
    base_pred = test["baseline"].to_numpy()[sl]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    style_axes(ax)

    ax.plot(times, actual, color=INK_PRIMARY, linewidth=2.0, zorder=4, label="Факт")
    ax.plot(times, model_pred, color=ALMATY, linewidth=2.0, zorder=3,
            label="Прогноз модели")
    ax.plot(times, base_pred, color=NEUTRAL, linewidth=1.6, linestyle="--",
            zorder=2, label="Бейзлайн")

    quiet_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)

    ax.set_ylabel("Температура, °C", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Алматы, горизонт {horizon} ч: прогноз против факта",
                 color=INK_PRIMARY, fontsize=13, pad=34, loc="left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast-vs-actual.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Расчёт по всем горизонтам, это займёт около минуты...")
    results = collect(ALL_HORIZONS)

    plot_ratio_curve(results, ALL_HORIZONS)
    print("  docs/figures/mae-ratio-by-horizon.png")

    plot_absolute_errors(results, ALL_HORIZONS)
    print("  docs/figures/mae-absolute-by-horizon.png")

    plot_forecast_vs_actual()
    print("  docs/figures/forecast-vs-actual.png")

    print("\nКлючевые значения:")
    for _, city_name, _ in CITIES:
        print(f"\n{city_name}")
        for h in KEY_HORIZONS:
            r = results[city_name][h]
            ratio = r["model_mae"] / r["baseline_mae"]
            print(f"  {h:>2} ч   MAE {r['model_mae']:.3f} / {r['baseline_mae']:.3f}"
                  f"   ratio {ratio:.3f}")
