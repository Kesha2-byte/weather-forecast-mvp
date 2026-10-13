"""
Проверка источника данных Open-Meteo.

Запуск:
    python check_data_source.py

Отвечает на вопросы:
  - какое доступно временное разрешение
  - как глубоко уходит архив
  - какие поля отдаются и как называются
  - какая задержка у архива

Ключ API не требуется.
"""

import json
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import urlopen

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CITY = {"name": "Алматы", "lat": 43.2567, "lon": 76.9286, "tz": "Asia/Almaty"}

HOURLY_VARS = "temperature_2m,relative_humidity_2m,surface_pressure,precipitation"


def fetch(url, params):
    full = f"{url}?{urlencode(params)}"
    with urlopen(full, timeout=30) as response:
        return json.loads(response.read().decode())


def check_hourly_resolution():
    """Часовое разрешение и набор полей."""
    print("1. ЧАСОВОЕ РАЗРЕШЕНИЕ И НАБОР ПОЛЕЙ")

    data = fetch(ARCHIVE_URL, {
        "latitude": CITY["lat"],
        "longitude": CITY["lon"],
        "start_date": "2026-08-01",
        "end_date": "2026-08-07",
        "hourly": HOURLY_VARS,
        "timezone": CITY["tz"],
    })

    hourly = data["hourly"]
    times = hourly["time"]

    print(f"Точек за 7 дней: {len(times)} (ожидается 168 при часовом шаге)")
    print(f"Первая метка:    {times[0]}")
    print(f"Вторая метка:    {times[1]}")
    print(f"Последняя:       {times[-1]}")
    print(f"Единицы: {json.dumps(data.get('hourly_units', {}), ensure_ascii=False)}")

    print("\nПоля и заполненность:")
    for field, values in hourly.items():
        if field == "time":
            continue
        missing = sum(1 for v in values if v is None)
        sample = next((v for v in values if v is not None), None)
        print(f"  {field:24} пропусков: {missing:4}   пример: {sample}")


def check_archive_depth():
    """Глубина архива."""
    print("\n2. ГЛУБИНА АРХИВА")

    for year in (1940, 1980, 2010):
        try:
            data = fetch(ARCHIVE_URL, {
                "latitude": CITY["lat"],
                "longitude": CITY["lon"],
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-01-02",
                "daily": "temperature_2m_mean",
                "timezone": CITY["tz"],
            })
            values = data["daily"]["temperature_2m_mean"]
            status = "есть данные" if any(v is not None for v in values) else "пусто"
            print(f"  {year}: {status}   {values}")
        except Exception as exc:
            print(f"  {year}: ошибка - {exc}")


def check_freshness():
    """Задержка публикации архива."""
    print("\n3. СВЕЖЕСТЬ ДАННЫХ")

    today = date.today()
    last_archive_day = None

    for days_back in range(1, 15):
        day = today - timedelta(days=days_back)
        data = fetch(ARCHIVE_URL, {
            "latitude": CITY["lat"],
            "longitude": CITY["lon"],
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "daily": "temperature_2m_mean",
            "timezone": CITY["tz"],
        })
        if any(v is not None for v in data["daily"]["temperature_2m_mean"]):
            last_archive_day = day
            break

    if last_archive_day:
        lag = (today - last_archive_day).days
        print(f"  Сегодня:                 {today}")
        print(f"  Последний день в архиве: {last_archive_day}")
        print(f"  Задержка архива:         {lag} дн.")
    else:
        print("  Архив не отдал данные за последние 14 дней")

    data = fetch(FORECAST_URL, {
        "latitude": CITY["lat"],
        "longitude": CITY["lon"],
        "hourly": "temperature_2m",
        "past_days": 7,
        "forecast_days": 3,
        "timezone": CITY["tz"],
    })
    times = data["hourly"]["time"]
    print(f"\n  Forecast API (past_days=7, forecast_days=3):")
    print(f"    диапазон: {times[0]} .. {times[-1]}")
    print(f"    точек:    {len(times)}")


if __name__ == "__main__":
    print(f"Источник: Open-Meteo")
    print(f"Город: {CITY['name']} ({CITY['lat']}, {CITY['lon']})\n")

    check_hourly_resolution()
    check_archive_depth()
    check_freshness()

    print("\nГотово. Результаты зафиксированы в docs/data-source.md")
