import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/monthly_sales.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month").reset_index(drop=True)

# ---- Feature engineering ----
df["month_num"] = df["month"].dt.month
df["quarter"] = df["month"].dt.quarter
df["t"] = range(len(df))                                   # trend
df["is_holiday_season"] = df["month_num"].isin([11, 12]).astype(int)
df["lag_1"] = df["total_sales"].shift(1)
df["lag_2"] = df["total_sales"].shift(2)
df["lag_3"] = df["total_sales"].shift(3)
df["rolling_mean_3"] = df["total_sales"].shift(1).rolling(3).mean()
df = df.dropna().reset_index(drop=True)

features = ["month_num", "quarter", "t", "is_holiday_season",
            "lag_1", "lag_2", "lag_3", "rolling_mean_3"]
X, y = df[features], df["total_sales"]

# ---- Time-based split (last 6 months = test) ----
X_train, X_test = X[:-6], X[-6:]
y_train, y_test = y[:-6], y[-6:]

# ---- Models + tuning ----
tscv = TimeSeriesSplit(n_splits=3)
models = {
    "Linear": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"model__alpha": [0.1, 1, 10, 100]}),
    "Lasso": (Lasso(max_iter=20000), {"model__alpha": [1, 10, 100, 1000]}),
}

results, fitted = [], {}
for name, (est, grid) in models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
    gs = GridSearchCV(pipe, grid or {"model__fit_intercept": [True]},
                      cv=tscv, scoring="neg_mean_absolute_error")
    gs.fit(X_train, y_train)
    pred = gs.predict(X_test)
    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, pred)),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": round(r2_score(y_test, pred), 3),
        "MAPE_%": round(np.mean(np.abs((y_test - pred) / y_test)) * 100, 2),
        "Best_params": gs.best_params_,
    })
    fitted[name] = (gs.best_estimator_, pred)

res = pd.DataFrame(results)
print(res.to_string(index=False))

# ---- Save best model ----
best = res.sort_values("RMSE").iloc[0]["Model"]
print("\nBest model:", best)
joblib.dump({"model": fitted[best][0], "features": features}, "model.pkl")

# ---- Plot ----
plt.figure(figsize=(8, 4))
plt.plot(df["month"][-6:], y_test.values, marker="o", label="Actual")
plt.plot(df["month"][-6:], fitted[best][1], marker="x", label=f"Predicted ({best})")
plt.legend(); plt.title("Actual vs Predicted Monthly Sales")
plt.tight_layout(); plt.savefig("forecast.png")
print("Saved model.pkl and forecast.png")