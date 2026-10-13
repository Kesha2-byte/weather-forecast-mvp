import numpy as np
from sklearn.linear_model import LinearRegression

from features import build_dataset, chronological_split

print("numpy:", np.__version__)
print("BLAS:", np.__config__.show(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name"))

dataset, feature_names = build_dataset(1, 24)
train, test = chronological_split(dataset)

X_train = train[feature_names].to_numpy()
y_train = train["target"].to_numpy()
X_test = test[feature_names].to_numpy()
y_test = test["target"].to_numpy()

print("\nX_train dtype:", X_train.dtype, " shape:", X_train.shape)
print("все значения конечны:", np.isfinite(X_train).all(), np.isfinite(X_test).all())
print("min:", X_train.min(), " max:", X_train.max())
print("y конечен:", np.isfinite(y_train).all(), np.isfinite(y_test).all())

model = LinearRegression()
model.fit(X_train, y_train)

print("\ncoef:", model.coef_)
print("intercept:", model.intercept_)
print("coef конечны:", np.isfinite(model.coef_).all())

pred_sklearn = model.predict(X_test)
pred_manual = X_test.dot(model.coef_) + model.intercept_

print("\nsklearn predict  min/max:", pred_sklearn.min(), pred_sklearn.max())
print("ручной расчёт    min/max:", pred_manual.min(), pred_manual.max())
print("максимальное расхождение:", np.abs(pred_sklearn - pred_manual).max())

print("\nMAE sklearn:", np.mean(np.abs(y_test - pred_sklearn)))
print("MAE ручной :", np.mean(np.abs(y_test - pred_manual)))
