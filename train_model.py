import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv("data.csv", parse_dates=["timestamp"])
except ValueError:
    # Handle case where CSV is missing header
    df = pd.read_csv("data.csv", header=None, names=["timestamp","rpm","vibration","temp","flow","fault"], parse_dates=["timestamp"])

# drop rows with missing values
X = df[["rpm","vibration","temp","flow"]].dropna()

scaler = StandardScaler()
Xs = scaler.fit_transform(X.values)

model = IsolationForest(contamination=0.01, random_state=42, n_estimators=200)
model.fit(Xs)

joblib.dump({"scaler": scaler, "model": model}, "if_model.pkl")
print("Saved if_model.pkl (scaler+model)")
