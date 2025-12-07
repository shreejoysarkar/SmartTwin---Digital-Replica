# detect_live.py
import pandas as pd
import joblib
import json
from datetime import datetime

MODEL_FILE = "if_model.pkl"
INPUT_DATA = "data.csv"
OUTPUT_DATA = "data_with_anoms.csv"
OUTPUT_ALERTS = "alerts.json"

def run_detection():
    try:
        raw = pd.read_csv(INPUT_DATA, parse_dates=["timestamp"])
        if raw.empty:
            print("No data yet.")
            return
    except FileNotFoundError:
        print("data.csv not found.")
        return

    clf = joblib.load(MODEL_FILE)
    scaler = clf["scaler"]
    model = clf["model"]

    df = raw.copy()
    X = df[["rpm","vibration","temp","flow"]].fillna(method="ffill")
    Xs = scaler.transform(X.values)
    preds = model.predict(Xs)   # -1 anomaly, 1 normal
    df["anomaly"] = (preds == -1).astype(int)

    # Save full annotated data
    df.to_csv(OUTPUT_DATA, index=False)

    # Extract last few anomalies
    anomalies = df[df["anomaly"] == 1].tail(20)
    alerts = anomalies.to_dict(orient="records")

    with open(OUTPUT_ALERTS, "w") as f:
        json.dump(alerts, f, default=str, indent=2)

    print(f"[{datetime.utcnow()}] Detection cycle complete. Anomalies: {len(alerts)}")

if __name__ == "__main__":
    run_detection()
