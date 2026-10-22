from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from forecast_service import MAX_HORIZON, MODEL_NAME, MODEL_VERSION, connect, run_and_store

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Weather Forecast MVP", version="1.0")


def fetch_all(sql, params=()):
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            columns = [c.name for c in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def ensure_city(city_id):
    rows = fetch_all("SELECT city_id FROM cities WHERE city_id = %s", (city_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"City {city_id} not found")


@app.get("/api/v1/cities")
def list_cities():
    return fetch_all("""
        SELECT city_id, name, latitude, longitude, timezone, elevation_m
        FROM cities ORDER BY city_id
    """)


@app.get("/api/v1/cities/{city_id}/observations")
def list_observations(
    city_id: int,
    date_from: datetime = Query(..., alias="from"),
    date_to: datetime = Query(..., alias="to"),
):
    ensure_city(city_id)
    if date_from > date_to:
        raise HTTPException(status_code=422, detail="Period start is after period end")

    rows = fetch_all("""
        SELECT observed_at, temperature_c, humidity_pct,
               pressure_surface_hpa, pressure_msl_hpa, precipitation_mm
        FROM observations
        WHERE city_id = %s AND observed_at BETWEEN %s AND %s
        ORDER BY observed_at
    """, (city_id, date_from, date_to))

    return {"city_id": city_id, "count": len(rows), "observations": rows}


@app.post("/api/v1/cities/{city_id}/forecasts", status_code=201)
def create_forecast(city_id: int, max_horizon_hours: int = MAX_HORIZON):
    ensure_city(city_id)
    if not 1 <= max_horizon_hours <= MAX_HORIZON:
        raise HTTPException(status_code=422, detail="Horizon must be between 1 and 48")

    created_at, predictions, stored = run_and_store(city_id, max_horizon=max_horizon_hours)

    return {
        "city_id": city_id,
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "created_at": created_at,
        "stored": stored,
        "forecasts": predictions,
    }


@app.get("/api/v1/cities/{city_id}/forecasts")
def get_forecast(city_id: int):
    ensure_city(city_id)

    rows = fetch_all("""
        SELECT f.created_at, f.target_time, f.horizon_hours, f.predicted_temperature_c
        FROM forecasts f
        WHERE f.city_id = %s
          AND f.created_at = (SELECT MAX(created_at) FROM forecasts WHERE city_id = %s)
        ORDER BY f.horizon_hours
    """, (city_id, city_id))

    if not rows:
        raise HTTPException(status_code=404, detail="No forecasts stored for this city")

    return {
        "city_id": city_id,
        "created_at": rows[0]["created_at"],
        "forecasts": rows,
    }


@app.get("/api/v1/metrics")
def get_metrics(city_id: int = None, horizon_hours: int = None):
    conditions = []
    params = []

    if city_id is not None:
        ensure_city(city_id)
        conditions.append("f.city_id = %s")
        params.append(city_id)
    if horizon_hours is not None:
        conditions.append("f.horizon_hours = %s")
        params.append(horizon_hours)

    where = ("AND " + " AND ".join(conditions)) if conditions else ""

    return fetch_all(f"""
        SELECT c.name AS city,
               m.name AS model,
               m.version,
               COUNT(*) AS matched_pairs,
               ROUND(AVG(ABS(o.temperature_c - f.predicted_temperature_c)), 3) AS mae,
               ROUND(SQRT(AVG(POWER(o.temperature_c - f.predicted_temperature_c, 2))), 3) AS rmse
        FROM forecasts f
        JOIN observations o
          ON o.city_id = f.city_id AND o.observed_at = f.target_time
        JOIN cities c ON c.city_id = f.city_id
        JOIN models m ON m.model_id = f.model_id
        WHERE TRUE {where}
        GROUP BY c.name, m.name, m.version
        ORDER BY c.name
    """, tuple(params))


@app.get("/api/v1/metrics/by-horizon")
def get_metrics_by_horizon(city_id: int):
    ensure_city(city_id)

    return fetch_all("""
        SELECT f.horizon_hours,
               COUNT(*) AS matched_pairs,
               ROUND(AVG(ABS(o.temperature_c - f.predicted_temperature_c)), 3) AS mae,
               ROUND(SQRT(AVG(POWER(o.temperature_c - f.predicted_temperature_c, 2))), 3) AS rmse
        FROM forecasts f
        JOIN observations o
          ON o.city_id = f.city_id AND o.observed_at = f.target_time
        WHERE f.city_id = %s
        GROUP BY f.horizon_hours
        ORDER BY f.horizon_hours
    """, (city_id,))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
