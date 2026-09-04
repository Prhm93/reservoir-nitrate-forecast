"""Generate synthetic reservoir nitrate data for the tutorial."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n_days = 730
dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
t = np.arange(n_days)

# Rainfall: seasonal, wet winters, occasional storms
season = 3 + 3 * np.cos(2 * np.pi * (t - 30) / 365)   # mm/day baseline
storms = rng.exponential(1.2, n_days) * (rng.random(n_days) < 0.15) * 15
rainfall = np.clip(season + rng.normal(0, 1.5, n_days) + storms, 0, None)

# Inflow: rainfall converted to flow, with a delay
Q_in = 100000 + 8000 * pd.Series(rainfall).rolling(3, min_periods=1).mean().values
Q_in += rng.normal(0, 5000, n_days)

# Upstream nitrate: rises after rain (fertiliser washoff), higher in autumn
autumn = 8 * np.exp(-((t % 365 - 280) ** 2) / (2 * 40**2))
C_in = 35 + autumn + 0.6 * pd.Series(rainfall).rolling(5, min_periods=1).mean().values
C_in += rng.normal(0, 2, n_days)
C_in = np.clip(C_in, 20, 65)

# Reservoir nitrate: mass-balance dynamics, target we forecast
V = 50_000_000  # m^3
Q_out = 200_000
C = np.zeros(n_days)
C[0] = 34.0
for i in range(1, n_days):
    dC = (Q_in[i] * C_in[i] - Q_out * C[i-1]) / V
    C[i] = C[i-1] + dC + rng.normal(0, 0.15)

df = pd.DataFrame({
    "date": dates,
    "rainfall_mm": rainfall.round(2),
    "Q_in_m3day": Q_in.round(0),
    "C_in_mgL": C_in.round(2),
    "C_reservoir_mgL": C.round(2),
})
df.to_csv("reservoir_data.csv", index=False)
print(df.head(7).to_string(index=False))
print(f"\n{len(df)} rows, mean C = {df.C_reservoir_mgL.mean():.2f} mg/L, max = {df.C_reservoir_mgL.max():.2f} mg/L")
