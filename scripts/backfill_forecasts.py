import argparse
from datetime import timedelta

from features import load_series
from forecast_service import MODELS, MODEL_TITLES, run_and_store

CITIES = [(1, "Алматы"), (2, "Астана")]


def main():
    parser = argparse.ArgumentParser(
        description="Генерирует прогнозы в прошлом, чтобы метрики имели данные для сравнения"
    )
    parser.add_argument("--runs", type=int, default=30,
                        help="Количество моментов построения прогноза в прошлом; "
                             "дополнительно строится прогноз от последнего наблюдения")
    parser.add_argument("--step-hours", type=int, default=24,
                        help="Шаг между моментами построения")
    parser.add_argument("--models", nargs="*", default=list(MODELS),
                        help="Модели, для которых строить прогнозы")
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        raise SystemExit("Неизвестные модели: %s" % ", ".join(unknown))

    for city_id, city_name in CITIES:
        series = load_series(city_id)
        latest = series["observed_at"].max()

        for model in args.models:
            title = MODEL_TITLES.get(model, model)
            print(f"\n{city_name}, {title}")
            total = 0

            for index in range(args.runs, -1, -1):
                as_of = latest - timedelta(hours=index * args.step_hours)
                created_at, _, stored = run_and_store(city_id, as_of=as_of, model=model)
                total += stored

            print(f"  начал прогноза: {args.runs}, записей сохранено: {total}")


if __name__ == "__main__":
    main()
