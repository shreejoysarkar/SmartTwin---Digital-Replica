# app_polish.py
# Streamlit polished frontend for SmartTwin pump twin
# Drop this into your project folder and run: streamlit run app_polish.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(page_title="SmartTwin Pump Twin — Polished", layout="wide")
st.title("SmartTwin — Pump Digital Twin (Polished UI)")

DATAFILE = "data.csv"
MODELFILE = "if_model.pkl"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=2)
def read_data():
    if os.path.exists(DATAFILE):
        df = pd.read_csv(DATAFILE, parse_dates=["timestamp"])
    else:
        df = pd.DataFrame(columns=["timestamp","rpm","vibration","temp","flow","fault"])
    return df

def append_injection(rowdict):
    pd.DataFrame([rowdict]).to_csv(DATAFILE, mode="a", header=False, index=False)

def load_model():
    try:
        m = joblib.load(MODELFILE)
        return m
    except Exception:
        return None

# -----------------------------
# Layout: top controls
# -----------------------------
st.markdown("### Controls")
control_col1, control_col2, control_col3 = st.columns([1.5,1,1])

with control_col1:
    sim_mode = st.selectbox("Simulate fault mode (for demo)", ["None","Vibration spike","Low flow","Overheat"])
    severity = st.slider("Severity", 1, 10, 5)
    if st.button("Inject selected fault now"):
        # create a fault row according to selection + severity
        if sim_mode == "Vibration spike":
            vib = 0.6 + severity * 0.4 + np.random.normal(0,0.05)
            row = {"timestamp": datetime.utcnow().isoformat(), "rpm":1500, "vibration": round(float(vib),3), "temp":40, "flow":120, "fault":1}
        elif sim_mode == "Low flow":
            flow = max(10, 120 - severity*8 + np.random.normal(0,1))
            row = {"timestamp": datetime.utcnow().isoformat(), "rpm":1450, "vibration":0.6, "temp":36, "flow": round(float(flow),2), "fault":2}
        elif sim_mode == "Overheat":
            temp = 35 + severity*3 + np.random.normal(0,0.5)
            row = {"timestamp": datetime.utcnow().isoformat(), "rpm":1450, "vibration":0.6, "temp": round(float(temp),2), "flow":120, "fault":3}
        else:
            row = {"timestamp": datetime.utcnow().isoformat(), "rpm":1450, "vibration":0.6, "temp":35, "flow":120, "fault":0}
        append_injection(row)
        st.success("Injected: " + sim_mode)

with control_col2:
    st.write("Model")
    if st.button("(Re)load model"):
        m = load_model()
        if m is not None:
            st.success("Model loaded")
        else:
            st.warning("Model not found. Run train_model.py after collecting normal data.")

with control_col3:
    if st.button("Export data.csv"):
        if os.path.exists(DATAFILE):
            with open(DATAFILE, "rb") as f:
                st.download_button("Download data.csv", data=f, file_name="data.csv")
        else:
            st.info("No data.csv available yet.")

st.markdown("---")

# -----------------------------
# Read data and model
# -----------------------------
df = read_data()
model = load_model()

# If model exists compute anomalies
if model is not None and not df.empty:
    X = df[["rpm","vibration","temp","flow"]].fillna(method="ffill").values
    preds = model.predict(X)  # 1 normal, -1 anomaly
    df["anomaly"] = (preds == -1).astype(int)
else:
    if "anomaly" not in df.columns:
        df["anomaly"] = 0

# Quick metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Latest RPM", df["rpm"].iloc[-1] if not df.empty else "—")
with m2:
    st.metric("Latest Vibration", round(float(df["vibration"].iloc[-1]),3) if not df.empty else "—")
with m3:
    st.metric("Latest Temp (°C)", round(float(df["temp"].iloc[-1]),2) if not df.empty else "—")
with m4:
    st.metric("Total anomalies", int(df["anomaly"].sum()) if "anomaly" in df.columns else 0)

st.markdown("---")

# -----------------------------
# Main layout: left 3D + charts, right anomaly timeline & details
# -----------------------------
left, right = st.columns([2.2, 1])

# -----------------------------
# Left: 3D replica + plotly charts
# -----------------------------
with left:
    st.subheader("3D Replica — Pump + Motor")
    # Determine if latest is anomaly
    latest = df.tail(1)
    is_anomaly = False
    if not latest.empty and int(latest["anomaly"].iloc[0]) == 1:
        is_anomaly = True
    # also a simple heuristic rule
    if not latest.empty and float(latest["vibration"].iloc[0]) > 2.0:
        is_anomaly = True

    base_color = "red" if is_anomaly else "lightblue"
    motor_color = "darkred" if is_anomaly else "orange"

    # Build a simple 3D scene with plotly using primitives (cylinders approximated)
    # Motor cylinder (left)
    def create_cylinder(cx, cy, cz, radius, height, color, segments=30):
        theta = np.linspace(0, 2*np.pi, segments)
        x = cx + radius*np.cos(theta)
        y = cy + radius*np.sin(theta)
        z_top = cz + height/2
        z_bot = cz - height/2
        # top circle, bottom circle
        X = np.concatenate([x, x, [cx]])
        Y = np.concatenate([y, y, [cy]])
        Z = np.concatenate([np.full_like(x, z_top), np.full_like(x, z_bot), np.array([cz])])
        n = segments
        i = []
        j = []
        k = []
        center_idx = 2*n
        for s in range(n-1):
            i.append(s); j.append(s+1); k.append(center_idx)
        i.append(n-1); j.append(0); k.append(center_idx)
        # sides
        for s in range(n):
            i.append(s); j.append((s+1)%n); k.append(n + s)
            i.append((s+1)%n); j.append(((s+1)%n) + n); k.append(n + s)
        mesh = go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=color, opacity=0.9, name="cyl")
        return mesh

    motor = create_cylinder(-1.2, 0, 0, 0.45, 0.8, motor_color)
    # pump body box
    hx, hy, hz = 0.9, 0.6, 0.5
    cx, cy, cz = 0.6, 0.0, 0.0
    box_verts = [
        (cx-hx/2, cy-hy/2, cz-hz/2),
        (cx+hx/2, cy-hy/2, cz-hz/2),
        (cx+hx/2, cy+hy/2, cz-hz/2),
        (cx-hx/2, cy+hy/2, cz-hz/2),
        (cx-hx/2, cy-hy/2, cz+hz/2),
        (cx+hx/2, cy-hy/2, cz+hz/2),
        (cx+hx/2, cy+hy/2, cz+hz/2),
        (cx-hx/2, cy+hy/2, cz+hz/2),
    ]
    bx = [v[0] for v in box_verts]; by = [v[1] for v in box_verts]; bz = [v[2] for v in box_verts]
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(2,3,7),(2,7,6),(1,2,6),(1,6,5),(3,0,4),(3,4,7)]
    i_b, j_b, k_b = zip(*faces)
    pump_body = go.Mesh3d(x=bx, y=by, z=bz, i=i_b, j=j_b, k=k_b, color=base_color, opacity=0.7, name="pump")

    # impeller radial blades (approx)
    def impeller(cx=0, cy=0, cz=0, hub_r=0.12, tip_r=0.55, n_blades=8):
        verts = []
        faces = []
        for b in range(n_blades):
            a0 = 2*np.pi*b/n_blades
            a1 = a0 + 0.6*(2*np.pi/n_blades)
            v0 = (cx + hub_r*np.cos(a0), cy + hub_r*np.sin(a0), cz - 0.02)
            v1 = (cx + tip_r*np.cos((a0+a1)/2), cy + tip_r*np.sin((a0+a1)/2), cz - 0.02)
            v2 = (cx + hub_r*np.cos(a1), cy + hub_r*np.sin(a1), cz + 0.02)
            base = len(verts)
            verts.extend([v0, v1, v2])
            faces.append((base, base+1, base+2))
        vx = [v[0] for v in verts]; vy = [v[1] for v in verts]; vz = [v[2] for v in verts]
        i_f, j_f, k_f = zip(*faces)
        return go.Mesh3d(x=vx, y=vy, z=vz, i=i_f, j=j_f, k=k_f, color="gray", opacity=0.95, name="impeller")

    impeller_mesh = impeller()

    fig3d = go.Figure(data=[motor, pump_body, impeller_mesh])
    fig3d.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                        scene=dict(aspectmode='auto',
                                   camera=dict(eye=dict(x=2.5, y=-2.0, z=1.2))))
    st.plotly_chart(fig3d, use_container_width=True, height=420)

    st.subheader("Sensor charts")
    if df.shape[0] == 0:
        st.info("No data yet. Run sensor_sim.py (Person B or you).")
    else:
        window = df.tail(300).set_index("timestamp")

        # RPM plot with nicer axis and hover
        fig_rpm = go.Figure()
        fig_rpm.add_trace(go.Scatter(x=window.index, y=window["rpm"], mode="lines+markers", name="RPM"))
        fig_rpm.update_layout(height=220, margin=dict(l=10,r=10,t=20,b=20), yaxis_title="RPM")
        st.plotly_chart(fig_rpm, use_container_width=True)

        # Vibration plot
        fig_vib = go.Figure()
        vib_line = go.Scatter(x=window.index, y=window["vibration"], mode="lines+markers", name="Vibration")
        fig_vib.add_trace(vib_line)
        # add threshold band (example)
        fig_vib.add_hrect(y0=2.0, y1=5.0, fillcolor="red", opacity=0.15, layer="below", line_width=0)
        fig_vib.update_layout(height=180, margin=dict(l=10,r=10,t=20,b=20), yaxis_title="g")
        st.plotly_chart(fig_vib, use_container_width=True)

        # Temp and Flow shared chart with second y-axis
        fig_tf = go.Figure()
        fig_tf.add_trace(go.Scatter(x=window.index, y=window["temp"], mode="lines", name="Temp (°C)", yaxis="y1"))
        fig_tf.add_trace(go.Scatter(x=window.index, y=window["flow"], mode="lines", name="Flow (L/min)", yaxis="y2"))
        fig_tf.update_layout(height=220, margin=dict(l=10,r=10,t=20,b=20),
                             yaxis=dict(title="Temp (°C)"),
                             yaxis2=dict(title="Flow (L/min)", overlaying="y", side="right"))
        st.plotly_chart(fig_tf, use_container_width=True)

# -----------------------------
# Right: anomaly timeline + last anomaly stamp + details
# -----------------------------
with right:
    st.subheader("Anomaly timeline & details")
    if df.shape[0] == 0:
        st.info("No data yet.")
    else:
        # Mark anomaly points on a small timeline chart
        timeline_df = df.tail(300).set_index("timestamp")
        timeline_df = timeline_df.reset_index()
        timeline_df["marker"] = np.where(timeline_df["anomaly"]==1, 1, 0)

        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(x=timeline_df["timestamp"], y=timeline_df["vibration"], mode="lines", name="Vibration"))
        # add anomaly markers
        anom = timeline_df[timeline_df["anomaly"]==1]
        if not anom.empty:
            fig_tl.add_trace(go.Scatter(x=anom["timestamp"], y=anom["vibration"], mode="markers", marker=dict(size=10, color="red"), name="Anomaly"))
        fig_tl.update_layout(height=200, margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(fig_tl, use_container_width=True)

        # Last anomaly stamp
        if df["anomaly"].sum() > 0:
            last_anom = df[df["anomaly"]==1].tail(1).iloc[0]
            st.markdown("**Last anomaly detected**")
            st.write(f"- time: {last_anom['timestamp']}")
            st.write(f"- rpm: {last_anom['rpm']}, vib: {last_anom['vibration']}, temp: {last_anom['temp']}, flow: {last_anom['flow']}")
            st.warning("⚠️ Anomaly detected — recommended: inspection and follow-up.")
        else:
            st.success("No anomalies flagged by model.")

        # Show recent anomaly rows
        st.markdown("Recent rows (tail)")
        st.dataframe(df.tail(10)[["timestamp","rpm","vibration","temp","flow","anomaly"]].iloc[::-1], height=240)

# -----------------------------
# MQTT placeholder (Person B can wire this)
# -----------------------------
st.markdown("---")
st.subheader("MQTT placeholder (for Person B)")
st.caption("If you connect a real instrument via MQTT, Person B can publish JSON payloads with the fields: timestamp, rpm, vibration, temp, flow. Below is example placeholder code for the subscriber to run (not executed here).")

st.code('''\
# Example paho-mqtt subscriber (run separately on the machine that receives MQTT messages)
# pip install paho-mqtt
"""
import json
import paho.mqtt.client as mqtt
from datetime import datetime
import pandas as pd

BROKER = "mqtt://broker.example.com"  # replace
TOPIC = "plant/pump/telemetry"

def on_connect(client, userdata, flags, rc):
    print("Connected", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        # Expected: { "timestamp": "...", "rpm": 1450, "vibration": 0.6, "temp": 35, "flow": 120 }
        row = {
            "timestamp": payload.get("timestamp", datetime.utcnow().isoformat()),
            "rpm": payload.get("rpm"),
            "vibration": payload.get("vibration"),
            "temp": payload.get("temp"),
            "flow": payload.get("flow"),
            "fault": payload.get("fault", 0)
        }
        pd.DataFrame([row]).to_csv("data.csv", mode="a", header=False, index=False)
    except Exception as e:
        print("bad message", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.host", 1883, 60)
client.loop_forever()
"""
''', language="python")

st.caption("End of dashboard.")
