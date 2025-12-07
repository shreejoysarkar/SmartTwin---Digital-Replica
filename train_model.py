import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# Prefer annotated data file if available (contains anomaly labels and extra telemetry)
data_file = "data_with_anoms.csv" if os.path.exists("data_with_anoms.csv") else "data.csv"

# Expected columns from simulator / app: timestamp,rpm,vibration,temp,flow,voltage,pressure,anomaly
default_cols = ["timestamp", "rpm", "vibration", "temp", "flow", "voltage", "pressure", "anomaly"]

try:
    df = pd.read_csv(data_file, parse_dates=["timestamp"])
except ValueError:
    # Handle case where CSV is missing header — try to map columns into expected shape
    # If file is shorter, supply as many names as possible
    # Fallback column names for older CSVs (without voltage/pressure)
    fallback = ["timestamp", "rpm", "vibration", "temp", "flow", "fault"]
    # choose names length based on number of columns in the file
    sample = pd.read_csv(data_file, header=None, nrows=1)
    ncols = sample.shape[1]
    if ncols >= len(default_cols):
        names = default_cols
    elif ncols >= len(fallback):
        names = fallback
    else:
        # create generic names
        names = [f"c{i}" for i in range(ncols)]
    df = pd.read_csv(data_file, header=None, names=names, parse_dates=[0])

# Build feature set: include voltage and pressure when present
feature_cols = [c for c in ["rpm", "vibration", "temp", "flow", "voltage", "pressure"] if c in df.columns]

if len(feature_cols) == 0:
    raise RuntimeError(f"No valid feature columns found in {data_file}. Found: {list(df.columns)}")

# drop rows with missing values in selected features
X = df[feature_cols].dropna()

scaler = StandardScaler()
Xs = scaler.fit_transform(X.values)

# IsolationForest on multivariate telemetry (unsupervised anomaly detection)
model = IsolationForest(contamination=0.01, random_state=42, n_estimators=200)
model.fit(Xs)

joblib.dump({"scaler": scaler, "model": model}, "if_model.pkl")
print(f"Saved if_model.pkl (scaler+model) using features: {feature_cols}")
