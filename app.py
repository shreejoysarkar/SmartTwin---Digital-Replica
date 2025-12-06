# app.py - Streamlit dashboard (Person A)
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="SmartTwin Pump Twin", layout="wide")
st.title("SmartTwin — Pump Digital Twin (Frontend)")
st.markdown("Simulated sensors + anomaly detection demo.")

DATAFILE = "data.csv"

# Left: visual area. Right: controls
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("Live charts (last 200 rows)")
    @st.cache_data(ttl=2)
    def read_data():
        try:
            df = pd.read_csv(DATAFILE, parse_dates=["timestamp"])
        except ValueError:
            # Handle case where CSV is missing header
            df = pd.read_csv(DATAFILE, header=None, names=["timestamp","rpm","vibration","temp","flow","fault"], parse_dates=["timestamp"])
        except:
            df = pd.DataFrame(columns=["timestamp","rpm","vibration","temp","flow","fault"])
        return df
    df = read_data()
    if df.shape[0] == 0:
        st.info("No data yet. Ask your teammate to run sensor_sim.py or run it yourself.")
    else:
        window = df.tail(200).set_index("timestamp")
        st.line_chart(window[["rpm"]], height=200)
        st.line_chart(window[["vibration"]], height=160)
        st.line_chart(window[["temp","flow"]], height=160)

with col2:
    st.subheader("Controls")
    if st.button("Inject vib spike"):
        row = {"timestamp": datetime.utcnow().isoformat(), "rpm":1500, "vibration":3.5, "temp":40, "flow":100, "fault":1}
        pd.DataFrame([row]).to_csv(DATAFILE, mode="a", header=False, index=False)
        st.rerun()
    if st.button("Inject low flow"):
        row = {"timestamp": datetime.utcnow().isoformat(), "rpm":1400, "vibration":0.7, "temp":36, "flow":60, "fault":2}
        pd.DataFrame([row]).to_csv(DATAFILE, mode="a", header=False, index=False)
        st.rerun()
    st.write("Export data:")
    try:
        with open(DATAFILE, "rb") as f:
            st.download_button("Download data.csv", data=f, file_name="data.csv")
    except:
        st.write("No data.csv yet.")

st.subheader("Anomaly Detection")
try:
    artifacts = joblib.load("if_model.pkl")
    if isinstance(artifacts, dict):
        scaler = artifacts.get("scaler")
        model = artifacts.get("model")
    else:
        model = artifacts
        scaler = None
except:
    model = None
    scaler = None

if model is None:
    st.info("No trained model found. After collecting ~200 rows of normal data, run `python train_model.py`.")
else:
    df = pd.read_csv(DATAFILE, parse_dates=["timestamp"])
    X = df[["rpm","vibration","temp","flow"]].ffill().values
    if scaler:
        Xs = scaler.transform(X)
    else:
        Xs = X
    preds = model.predict(Xs)
    df["anomaly"] = (preds == -1).astype(int)
    st.metric("Total anomalies", int(df["anomaly"].sum()))
    st.dataframe(df.tail(10)[["timestamp","rpm","vibration","temp","flow","anomaly"]].iloc[::-1])

st.subheader("Maintenance suggestion (rule)")
if df.shape[0] > 0:
    latest = df.tail(1).iloc[0]
    if latest.vibration > 2.0:
        st.warning("High vibration → suspect bearing failure. Recommend inspection and lubrication.")
    elif latest.temp > 60:
        st.warning("High temp → check cooling / lubrication.")
    elif latest.flow < 80:
        st.warning("Low flow → check cavitation / blockage.")
    else:
        st.success("No immediate action suggested.")
