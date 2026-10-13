import numpy as np
from sklearn.linear_model import LinearRegression

from features import build_dataset, chronological_split

np.seterr(all="ignore")

RANDOM_SEED = 42
CITIES = [(1, "Алматы"), (2, "Астана")]
HORIZONS = (1, 6, 12, 24, 48)


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate_baseline(test):
    actual = test["target"].to_numpy()
    predicted = test["baseline"].to_numpy()
    return mae(actual, predicted), rmse(actual, predicted)


def evaluate_regression(train, test, feature_names):
    model = LinearRegression()
    model.fit(train[feature_names].to_numpy(), train["target"].to_numpy())

    actual = test["target"].to_numpy()
    predicted = model.predict(test[feature_names].to_numpy())
    return mae(actual, predicted), rmse(actual, predicted), model


def run(city_id, horizon):
    dataset, feature_names = build_dataset(city_id, horizon)
    train, test = chronological_split(dataset)

    base_mae, base_rmse = evaluate_baseline(test)
    reg_mae, reg_rmse, model = evaluate_regression(train, test, feature_names)

    return {
        "horizon": horizon,
        "n_train": len(train),
        "n_test": len(test),
        "baseline_mae": base_mae,
        "baseline_rmse": base_rmse,
        "model_mae": reg_mae,
        "model_rmse": reg_rmse,
        "mae_ratio": reg_mae / base_mae,
        "rmse_ratio": reg_rmse / base_rmse,
        "coefficients": dict(zip(feature_names, model.coef_)),
        "intercept": float(model.intercept_),
    }


if __name__ == "__main__":
    for city_id, city_name in CITIES:
        print(f"\n{'=' * 86}")
        print(city_name)
        print(f"{'=' * 86}")
        print(f"{'Горизонт':>9} {'Base MAE':>10} {'Model MAE':>10} {'MAE ratio':>10} "
              f"{'Base RMSE':>11} {'Model RMSE':>11} {'RMSE ratio':>11}")

        for horizon in HORIZONS:
            r = run(city_id, horizon)
            print(f"{r['horizon']:>7} ч "
                  f"{r['baseline_mae']:>10.3f} "
                  f"{r['model_mae']:>10.3f} "
                  f"{r['mae_ratio']:>10.3f} "
                  f"{r['baseline_rmse']:>11.3f} "
                  f"{r['model_rmse']:>11.3f} "
                  f"{r['rmse_ratio']:>11.3f}")

    print(f"\n{'=' * 86}")
    print("Коэффициенты модели, Алматы, горизонт 24 ч")
    print(f"{'=' * 86}")
    result = run(1, 24)
    for name, value in result["coefficients"].items():
        print(f"  {name:>10}: {value:>8.4f}")
    print(f"  {'intercept':>10}: {result['intercept']:>8.4f}")
