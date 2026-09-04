"""Run the seasonal-naive baseline, lag regression, and gradient boosting.
Backtest with expanding window. Compute all four metrics."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("reservoir_data.csv", parse_dates=["date"]).set_index("date")

# ---- Feature engineering ----
d = df.copy()
for lag in [1, 2, 3, 7, 14]:
    d[f"C_lag{lag}"] = d["C_reservoir_mgL"].shift(lag)
d["rain_7d"]   = d["rainfall_mm"].rolling(7).mean()
d["Qin_7d"]    = d["Q_in_m3day"].rolling(7).mean()
d["Cin_7d"]    = d["C_in_mgL"].rolling(7).mean()
d["doy_sin"]   = np.sin(2*np.pi*d.index.dayofyear/365)
d["doy_cos"]   = np.cos(2*np.pi*d.index.dayofyear/365)
d = d.dropna()

target = "C_reservoir_mgL"
feature_cols = [c for c in d.columns if c not in [target, "rainfall_mm", "Q_in_m3day", "C_in_mgL"]]

# ---- Backtest: expanding window ----
# Start forecasting after day 400 (leaves ~330 test days)
start = 400
preds = {"seasonal_naive": [], "ridge": [], "gbm": []}
truths = []
dates_test = []

for i in range(start, len(d)):
    train = d.iloc[:i]
    test_row = d.iloc[i:i+1]
    y_train = train[target]
    y_true  = test_row[target].values[0]

    # 1) seasonal-naive: yesterday's value
    naive = train[target].iloc[-1]
    # 2) ridge on lags + rolling features + seasonal terms
    X_train = train[feature_cols].values
    X_test  = test_row[feature_cols].values
    sc = StandardScaler().fit(X_train)
    r = Ridge(alpha=1.0).fit(sc.transform(X_train), y_train)
    ridge_pred = r.predict(sc.transform(X_test))[0]
    # 3) gradient boosting
    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)[0]

    preds["seasonal_naive"].append(naive)
    preds["ridge"].append(ridge_pred)
    preds["gbm"].append(gbm_pred)
    truths.append(y_true)
    dates_test.append(test_row.index[0])

truths = np.array(truths)

def metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "MAE":  np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "WAPE": np.sum(np.abs(err)) / np.sum(np.abs(y_true)) * 100,
        "Bias": np.mean(err),
    }

print(f"{'Model':<18}{'MAE':>8}{'RMSE':>8}{'WAPE %':>10}{'Bias':>8}")
for name, p in preds.items():
    m = metrics(truths, np.array(p))
    print(f"{name:<18}{m['MAE']:>8.3f}{m['RMSE']:>8.3f}{m['WAPE']:>10.3f}{m['Bias']:>8.3f}")

# save predictions for chart
out = pd.DataFrame({"date": dates_test, "truth": truths, **{k: np.array(v) for k, v in preds.items()}})
out.to_csv("predictions.csv", index=False)
print("\nfirst 5 predictions:")
print(out.head().to_string(index=False))
