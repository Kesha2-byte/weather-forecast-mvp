-- Физическая схема базы данных
-- Система краткосрочного прогнозирования погоды
-- PostgreSQL 14+

DROP TABLE IF EXISTS forecasts;
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS cities;

-- ---------------------------------------------------------------
-- Города. Справочник, заполняется при инициализации системы.
-- ---------------------------------------------------------------
CREATE TABLE cities (
    city_id     SERIAL       PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    latitude    NUMERIC(8,5) NOT NULL,
    longitude   NUMERIC(8,5) NOT NULL,
    timezone    VARCHAR(64)  NOT NULL,
    elevation_m NUMERIC(6,1) NOT NULL,

    CONSTRAINT uq_cities_name   UNIQUE (name),
    CONSTRAINT ck_cities_lat    CHECK (latitude  BETWEEN  -90 AND  90),
    CONSTRAINT ck_cities_lon    CHECK (longitude BETWEEN -180 AND 180)
);

COMMENT ON COLUMN cities.elevation_m IS
    'Высота над уровнем моря. Необходима для интерпретации приземного давления: '
    'Алматы около 800 м, Астана около 350 м, разница достигает десятков гПа.';

-- ---------------------------------------------------------------
-- Наблюдения. Фактические измерения, накапливаются загрузчиком.
-- ---------------------------------------------------------------
CREATE TABLE observations (
    observation_id       BIGSERIAL   PRIMARY KEY,
    city_id              INTEGER     NOT NULL,
    observed_at          TIMESTAMPTZ NOT NULL,
    temperature_c        NUMERIC(5,2),
    humidity_pct         NUMERIC(5,2),
    pressure_surface_hpa NUMERIC(7,2),
    pressure_msl_hpa     NUMERIC(7,2),
    precipitation_mm     NUMERIC(6,2),

    CONSTRAINT fk_observations_city
        FOREIGN KEY (city_id) REFERENCES cities(city_id) ON DELETE CASCADE,

    -- ФТ-3: защита от дублирования при повторной загрузке периода
    CONSTRAINT uq_observations_city_time UNIQUE (city_id, observed_at),

    CONSTRAINT ck_observations_humidity
        CHECK (humidity_pct IS NULL OR humidity_pct BETWEEN 0 AND 100),
    CONSTRAINT ck_observations_precip
        CHECK (precipitation_mm IS NULL OR precipitation_mm >= 0)
);

COMMENT ON TABLE observations IS
    'Измеренные значения. NULL допустим: на границах архива источник возвращает пропуски.';

-- ---------------------------------------------------------------
-- Модели. Конфигурации, порождающие прогнозы.
-- ---------------------------------------------------------------
CREATE TABLE models (
    model_id   SERIAL      PRIMARY KEY,
    name       VARCHAR(64) NOT NULL,
    version    VARCHAR(32) NOT NULL,
    params     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    trained_at TIMESTAMPTZ,

    CONSTRAINT uq_models_name_version UNIQUE (name, version)
);

COMMENT ON COLUMN models.params IS
    'Параметры обучения в JSONB. Наборы параметров моделей не пересекаются: '
    'у бейзлайна период сезонности, у регрессии перечень лагов, '
    'у Хольта-Уинтерса три коэффициента сглаживания.';

-- ---------------------------------------------------------------
-- Прогнозы. Фиксируются в момент построения, не пересчитываются.
-- ---------------------------------------------------------------
CREATE TABLE forecasts (
    forecast_id             BIGSERIAL   PRIMARY KEY,
    city_id                 INTEGER     NOT NULL,
    model_id                INTEGER     NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL,
    target_time             TIMESTAMPTZ NOT NULL,
    horizon_hours           SMALLINT    NOT NULL,
    predicted_temperature_c NUMERIC(5,2) NOT NULL,

    CONSTRAINT fk_forecasts_city
        FOREIGN KEY (city_id) REFERENCES cities(city_id) ON DELETE CASCADE,
    CONSTRAINT fk_forecasts_model
        FOREIGN KEY (model_id) REFERENCES models(model_id),

    CONSTRAINT uq_forecasts_run
        UNIQUE (city_id, model_id, created_at, target_time),

    CONSTRAINT ck_forecasts_horizon
        CHECK (horizon_hours BETWEEN 1 AND 48),

    -- Прогноз всегда направлен в будущее относительно момента построения
    CONSTRAINT ck_forecasts_future
        CHECK (target_time > created_at)
);

COMMENT ON COLUMN forecasts.horizon_hours IS
    'Хранится явно, хотя выводится из разности target_time и created_at. '
    'Решение осознанное: горизонт является параметром запроса, а расчёт '
    'выполняется не строго в начале часа, из-за чего вычисленная разность '
    'давала бы дробные значения и разрушила группировку в аналитике.';

-- ---------------------------------------------------------------
-- Индексы
-- ---------------------------------------------------------------

-- Соединение прогнозов с фактами (центральный запрос аналитики)
-- и выборка истории наблюдений за период обслуживаются
-- индексом ограничения uq_observations_city_time.

-- Соединение со стороны прогнозов: ON f.city_id = o.city_id
--                                  AND f.target_time = o.observed_at
CREATE INDEX idx_forecasts_city_target
    ON forecasts (city_id, target_time);

-- Группировка метрик по модели и горизонту (ФТ-16, ФТ-17)
CREATE INDEX idx_forecasts_model_horizon
    ON forecasts (model_id, horizon_hours);

-- Выборка последнего прогноза по городу: ORDER BY created_at DESC
CREATE INDEX idx_forecasts_city_created
    ON forecasts (city_id, created_at DESC);

-- ---------------------------------------------------------------
-- Начальное наполнение справочников
-- ---------------------------------------------------------------
INSERT INTO cities (name, latitude, longitude, timezone, elevation_m) VALUES
    ('Алматы', 43.25670, 76.92860, 'Asia/Almaty', 800.0),
    ('Астана', 51.16940, 71.44910, 'Asia/Almaty', 350.0);

INSERT INTO models (name, version, params) VALUES
    ('seasonal_naive',    'v1', '{"season_period_hours": 24}'),
    ('linear_regression', 'v1', '{"lags": [1, 24, 48], "cyclical": ["hour", "doy"], "train_years": 3}'),
    ('holt_winters',      'v1', '{"season_period_hours": 24, "alpha": null, "beta": null, "gamma": null}');
