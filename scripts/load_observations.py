"""
Proof of Concept загрузки погодных наблюдений в PostgreSQL.

Запуск:
    python load_observations.py --city Алматы --from 2026-09-01 --to 2026-09-30

Проверяет ФТ-1, ФТ-2 и ФТ-3: загрузка из внешнего источника, сохранение
в базу и защита от дублирования при повторном запуске того же периода.

Зависимости: psycopg2-binary
Строка подключения берётся из переменной окружения DATABASE_URL,
по умолчанию postgresql://localhost/weather
"""

import argparse
import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import psycopg2
from psycopg2.extras import execute_values

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/weather")

HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "pressure_msl",
    "precipitation",
]


def fetch_city(cursor, name):
    """Возвращает строку справочника городов по названию."""
    cursor.execute(
        "SELECT city_id, latitude, longitude, timezone FROM cities WHERE name = %s",
        (name,),
    )
    row = cursor.fetchone()
    if row is None:
        raise SystemExit(f"Город {name} не найден в справочнике cities")
    return row


def fetch_observations(lat, lon, tz, date_from, date_to):
    """Запрашивает часовой ряд у внешнего источника."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_from,
        "end_date": date_to,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": tz,
    }
    with urlopen(f"{ARCHIVE_URL}?{urlencode(params)}", timeout=60) as response:
        return json.loads(response.read().decode())


def to_rows(city_id, payload):
    """
    Ответ приходит объектом из параллельных массивов, а не списком записей.
    Сшиваем их по индексу в строки для вставки.
    """
    hourly = payload["hourly"]
    times = hourly["time"]
    columns = [hourly[field] for field in HOURLY_FIELDS]
    return [
        (city_id, times[i], *(column[i] for column in columns))
        for i in range(len(times))
    ]


# ON CONFLICT реализует ФТ-3: повторная загрузка того же периода
# обновляет существующую запись вместо создания дубликата.
UPSERT = """
INSERT INTO observations (
    city_id, observed_at, temperature_c, humidity_pct,
    pressure_surface_hpa, pressure_msl_hpa, precipitation_mm
) VALUES %s
ON CONFLICT (city_id, observed_at) DO UPDATE SET
    temperature_c        = EXCLUDED.temperature_c,
    humidity_pct         = EXCLUDED.humidity_pct,
    pressure_surface_hpa = EXCLUDED.pressure_surface_hpa,
    pressure_msl_hpa     = EXCLUDED.pressure_msl_hpa,
    precipitation_mm     = EXCLUDED.precipitation_mm
RETURNING (xmax = 0) AS inserted
"""


def store(cursor, rows):
    """
    Пакетная вставка. Системный столбец xmax равен нулю у строк,
    которые были вставлены, и ненулевой у обновлённых, что позволяет
    отличить новые записи от перезаписанных.
    """
    result = execute_values(cursor, UPSERT, rows, page_size=1000, fetch=True)
    inserted = sum(1 for (flag,) in result if flag)
    return inserted, len(result) - inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    args = parser.parse_args()

    with psycopg2.connect(DB_URL) as connection:
        with connection.cursor() as cursor:
            city_id, lat, lon, tz = fetch_city(cursor, args.city)
            payload = fetch_observations(lat, lon, tz, args.date_from, args.date_to)
            rows = to_rows(city_id, payload)
            inserted, updated = store(cursor, rows)

    print(f"Город:     {args.city} (city_id={city_id})")
    print(f"Период:    {args.date_from} .. {args.date_to}")
    print(f"Получено:  {len(rows)}")
    print(f"Вставлено: {inserted}")
    print(f"Обновлено: {updated}")
    if inserted == 0 and updated > 0:
        print("Защита от дублирования работает: новых записей нет.")


if __name__ == "__main__":
    main()
