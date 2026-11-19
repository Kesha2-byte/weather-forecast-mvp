-- 0. Схема: проверка названий столбцов перед запуском остальных запросов

SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('cities', 'observations', 'forecasts', 'models')
ORDER BY table_name, ordinal_position;


-- 1. Скользящее среднее за сутки и отклонение текущего часа от него
-- Окно движется по времени внутри каждого города и берёт 24 последних часа

SELECT
    c.name AS city,
    o.observed_at,
    ROUND(o.temperature_c::numeric, 1) AS temperature_c,
    ROUND(AVG(o.temperature_c) OVER w::numeric, 2) AS ma_24h,
    ROUND((o.temperature_c - AVG(o.temperature_c) OVER w)::numeric, 2) AS deviation
FROM observations o
JOIN cities c ON c.id = o.city_id
WINDOW w AS (
    PARTITION BY o.city_id
    ORDER BY o.observed_at
    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
)
ORDER BY o.observed_at DESC, c.name
LIMIT 48;


-- 2. Суточная амплитуда и пять самых контрастных дней каждого города
-- RANK нумерует дни внутри города, внешний запрос отрезает первые пять

WITH daily AS (
    SELECT
        o.city_id,
        (o.observed_at AT TIME ZONE 'Asia/Almaty')::date AS day,
        MAX(o.temperature_c) - MIN(o.temperature_c) AS amplitude
    FROM observations o
    GROUP BY 1, 2
),
ranked AS (
    SELECT
        d.*,
        RANK() OVER (PARTITION BY d.city_id ORDER BY d.amplitude DESC) AS place
    FROM daily d
)
SELECT c.name AS city, r.day, ROUND(r.amplitude::numeric, 1) AS amplitude, r.place
FROM ranked r
JOIN cities c ON c.id = r.city_id
WHERE r.place <= 5
ORDER BY c.name, r.place;


-- 3. Средняя суточная амплитуда по месяцам и изменение к предыдущему месяцу
-- Этот запрос и есть числовое подтверждение того, что амплитуда зависит от сезона

WITH daily AS (
    SELECT
        o.city_id,
        (o.observed_at AT TIME ZONE 'Asia/Almaty')::date AS day,
        MAX(o.temperature_c) - MIN(o.temperature_c) AS amplitude
    FROM observations o
    GROUP BY 1, 2
),
monthly AS (
    SELECT
        city_id,
        DATE_TRUNC('month', day)::date AS month,
        AVG(amplitude) AS avg_amplitude
    FROM daily
    GROUP BY 1, 2
)
SELECT
    c.name AS city,
    m.month,
    ROUND(m.avg_amplitude::numeric, 2) AS avg_amplitude,
    ROUND((m.avg_amplitude - LAG(m.avg_amplitude) OVER (
        PARTITION BY m.city_id ORDER BY m.month))::numeric, 2) AS change_vs_prev,
    ROUND(AVG(m.avg_amplitude) OVER (
        PARTITION BY m.city_id ORDER BY m.month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2) AS ma_3m
FROM monthly m
JOIN cities c ON c.id = m.city_id
ORDER BY c.name, m.month;


-- 4. MAE сохранённых прогнозов по горизонтам, посчитанная прямо в базе
-- Контрольная сверка: значения должны совпасть с тем, что печатает experiment.py

SELECT
    c.name AS city,
    f.horizon_hours,
    COUNT(*) AS matched,
    ROUND(AVG(ABS(f.temperature_c - o.temperature_c))::numeric, 3) AS mae,
    ROUND(SQRT(AVG(POWER(f.temperature_c - o.temperature_c, 2)))::numeric, 3) AS rmse,
    ROUND(AVG(f.temperature_c - o.temperature_c)::numeric, 3) AS bias
FROM forecasts f
JOIN observations o
  ON o.city_id = f.city_id
 AND o.observed_at = f.target_time
JOIN cities c ON c.id = f.city_id
GROUP BY 1, 2
ORDER BY 1, 2;


-- 5. Ошибка по часам суток на горизонте 24 часа
-- NTILE делит часы на четыре группы по величине ошибки

WITH errors AS (
    SELECT
        f.city_id,
        EXTRACT(HOUR FROM f.target_time AT TIME ZONE 'Asia/Almaty')::int AS hour_of_day,
        ABS(f.temperature_c - o.temperature_c) AS abs_error
    FROM forecasts f
    JOIN observations o
      ON o.city_id = f.city_id
     AND o.observed_at = f.target_time
    WHERE f.horizon_hours = 24
),
per_hour AS (
    SELECT city_id, hour_of_day, COUNT(*) AS n, AVG(abs_error) AS mae
    FROM errors
    GROUP BY 1, 2
)
SELECT
    c.name AS city,
    p.hour_of_day,
    p.n,
    ROUND(p.mae::numeric, 3) AS mae,
    NTILE(4) OVER (PARTITION BY p.city_id ORDER BY p.mae) AS quartile
FROM per_hour p
JOIN cities c ON c.id = p.city_id
ORDER BY c.name, p.hour_of_day;


-- 6. Автокорреляция ошибки: если промахнулись час назад, промахнёмся ли сейчас

WITH errors AS (
    SELECT
        f.city_id,
        f.target_time,
        f.temperature_c - o.temperature_c AS error
    FROM forecasts f
    JOIN observations o
      ON o.city_id = f.city_id
     AND o.observed_at = f.target_time
    WHERE f.horizon_hours = 24
),
paired AS (
    SELECT
        city_id,
        error,
        LAG(error) OVER (PARTITION BY city_id ORDER BY target_time) AS prev_error
    FROM errors
)
SELECT
    c.name AS city,
    COUNT(*) AS pairs,
    ROUND(CORR(error, prev_error)::numeric, 3) AS autocorrelation
FROM paired p
JOIN cities c ON c.id = p.city_id
WHERE prev_error IS NOT NULL
GROUP BY 1
ORDER BY 1;


-- 7. Самые длинные серии часов подряд с растущей температурой
-- Разность двух нумераций даёт одинаковый номер группы для подряд идущих строк

WITH flagged AS (
    SELECT
        city_id,
        observed_at,
        temperature_c,
        temperature_c > LAG(temperature_c) OVER (
            PARTITION BY city_id ORDER BY observed_at) AS rising
    FROM observations
),
grouped AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (PARTITION BY city_id ORDER BY observed_at)
      - ROW_NUMBER() OVER (PARTITION BY city_id, rising ORDER BY observed_at) AS streak_id
    FROM flagged f
    WHERE rising IS NOT NULL
)
SELECT
    c.name AS city,
    MIN(g.observed_at) AS started_at,
    MAX(g.observed_at) AS ended_at,
    COUNT(*) AS hours,
    ROUND((MAX(g.temperature_c) - MIN(g.temperature_c))::numeric, 1) AS total_rise
FROM grouped g
JOIN cities c ON c.id = g.city_id
WHERE g.rising
GROUP BY c.name, g.streak_id
ORDER BY hours DESC
LIMIT 10;


-- 8. Месячная сводка по городам: средняя, минимум, максимум и доля часов ниже нуля

SELECT
    c.name AS city,
    DATE_TRUNC('month', o.observed_at AT TIME ZONE 'Asia/Almaty')::date AS month,
    COUNT(*) AS hours,
    ROUND(AVG(o.temperature_c)::numeric, 2) AS avg_t,
    ROUND(MIN(o.temperature_c)::numeric, 1) AS min_t,
    ROUND(MAX(o.temperature_c)::numeric, 1) AS max_t,
    ROUND((100.0 * COUNT(*) FILTER (WHERE o.temperature_c < 0) / COUNT(*))::numeric, 1) AS below_zero_pct
FROM observations o
JOIN cities c ON c.id = o.city_id
GROUP BY 1, 2
ORDER BY 1, 2;
