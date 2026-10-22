import argparse
from datetime import timedelta

from features import load_series
from forecast_service import run_and_store

CITIES = [(1, "Алматы"), (2, "Астана")]


def main():
    parser = argparse.ArgumentParser(
        description="Генерирует прогнозы в прошлом, чтобы метрики имели данные для сравнения"
    )
    parser.add_argument("--runs", type=int, default=30,
                        help="Количество моментов построения прогноза")
    parser.add_argument("--step-hours", type=int, default=24,
                        help="Шаг между моментами построения")
    args = parser.parse_args()

    for city_id, city_name in CITIES:
        series = load_series(city_id)
        latest = series["observed_at"].max()

        print(f"\n{city_name}")
        total = 0

        for index in range(args.runs, 0, -1):
            as_of = latest - timedelta(hours=index * args.step_hours)
            created_at, _, stored = run_and_store(city_id, as_of=as_of)
            total += stored
            print(f"  {created_at:%Y-%m-%d %H:%M}   сохранено {stored}")

        print(f"  Итого записей: {total}")


if __name__ == "__main__":
    main()
