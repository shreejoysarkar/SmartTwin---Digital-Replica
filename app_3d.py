# app_3d.py
import streamlit as st
import pandas as pd
import json
import time
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from models import ProcessingUnit, Motor, Connector, export_models

st.set_page_config(layout="wide", page_title="SmartTwin 3D Twin")

# --- UI styling (small enhancements)
st.markdown("""
<style>
h1 {font-weight:700; color:#0b5cff;}
.subtitle {color:#666; font-size:16px; margin-top:-10px}
.card {background:#f8f9fb; padding:10px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.06);}
.card.warn {background: linear-gradient(90deg,#fff0f0,#ffecec);}
.small {font-size:13px; color:#444}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="subtitle">Interactive 3D motor pump visualization — live telemetry & anomaly highlights</div>', unsafe_allow_html=True)

DATA_ANNOT = "data_with_anoms.csv"
ALERTS = "alerts.json"

# Function to build PU models from current session_state
def build_pu_models_from_session():
    """Build ProcessingUnit models from the current session_state configuration."""
    pu_count = int(st.session_state.get('pu_count', 1))
    pu_models = []
    for i in range(pu_count):
        pmotors = []
        num_motors = int(st.session_state.get(f'pu_{i}_num_motors', 1))
        pu_poll = float(st.session_state.get(f'pu_{i}_poll', 1.0))
        for m in range(1, num_motors + 1):
            conns = int(st.session_state.get(f'pu_{i}_motor_{m}_conns', 0))
            connectors = [Connector(id=f'pu{i+1}_m{m}_c{c+1}') for c in range(conns)]
            pmotors.append(Motor(name=f'Motor {m}', connectors=connectors))
        pu_obj = ProcessingUnit(name=f'Processing Unit {i+1}', poll_interval=pu_poll, motors=pmotors)
        pu_models.append(pu_obj)
    return pu_models

st.title("SmartTwin — 3D Motor pump")

# Controls column
col_controls, col_vis = st.columns([1,3])

with col_controls:
    # split controls into two tabs: main controls and processing unit configuration
    tab_controls, tab_pu = st.tabs(["Controls", "Processing Units"])

    with tab_controls:
        if "running3d" not in st.session_state:
            st.session_state.running3d = False
        if st.button("Start 3D Live"):
            st.session_state.running3d = True
        if st.button("Stop 3D Live"):
            st.session_state.running3d = False
        poll = st.number_input("Poll interval (s)", min_value=1, max_value=5, value=1, key='global_poll')
        st.markdown("---")
        st.write("Latest alerts")
        try:
            alerts = json.load(open(ALERTS))
        except Exception:
            alerts = []
        if alerts:
            # show newest alert prominently with red label, then up to 4 recent warnings
            recent = alerts[::-1][:5]
            for idx, a in enumerate(recent):
                msg = f"{a.get('timestamp','?')} — vib:{a.get('vibration')} temp:{a.get('temp')} volt:{a.get('voltage')} pres:{a.get('pressure')}"
                if idx == 0:
                    st.error(f"ALERT — {msg}")
                else:
                    st.warning(msg)
        else:
            st.info("No recent alerts")

    with tab_pu:
        # number of processing units
        pu_count = st.number_input("Number of processing units", min_value=1, max_value=8, value=st.session_state.get('pu_count', 1), key='pu_count')
        st.markdown("---")
        # for each processing unit, create a section with fields
        total_motors = 0
        for i in range(int(pu_count)):
            st.header(f"Processing Unit No. {i+1}")
            num_motors = st.number_input(f"Number of motors (PU {i+1})", min_value=1, max_value=8, value=st.session_state.get(f'pu_{i}_num_motors', 1), key=f'pu_{i}_num_motors')
            pu_poll = st.number_input(f"Poll interval (s) (PU {i+1})", min_value=0.1, max_value=60.0, value=float(st.session_state.get(f'pu_{i}_poll', 1.0)), step=0.1, key=f'pu_{i}_poll')
            with st.expander("Connectors per motor", expanded=False):
                for m in range(1, int(num_motors)+1):
                    st.number_input(f"Motor {m} connectors (PU {i+1})", min_value=0, max_value=8, value=st.session_state.get(f'pu_{i}_motor_{m}_conns', 0), key=f'pu_{i}_motor_{m}_conns')
            total_motors += int(num_motors)
        # store computed totals in session_state for use elsewhere
        st.session_state['pu_total_motors'] = total_motors
        
        # Build and display models
        pu_models = build_pu_models_from_session()
        st.session_state['pu_models'] = [p.to_dict() for p in pu_models]
        
        with st.expander("View / Export Models", expanded=False):
            for p in pu_models:
                st.subheader(p.name)
                st.json(p.to_dict())
            if st.button("Export models to JSON", key='export_models'):
                out_path = export_models(pu_models, path="models.json")
                st.success(f"Exported {len(pu_models)} models to {out_path}")

with col_vis:
    placeholder = st.empty()

# derive counts from processing unit configuration
pump_count = st.session_state.get('pu_total_motors', 1)
motor_count = st.session_state.get('pu_count', 1)

# helper: read latest annotated data
def read_latest():
    try:
        df = pd.read_csv(DATA_ANNOT, parse_dates=["timestamp"])
        if df.empty:
            return None
        return df.tail(1).iloc[0].to_dict()
    except Exception:
        return None

# create simple pump geometry (points in 3D)
def create_geometry(angle_rad=0.0, shaft_len=1.0, shaft_radius=0.05):
    # motor block (a rectangular cluster)
    motor_x = np.linspace(-1.0, -0.2, 4)
    motor_y = np.linspace(-0.5, 0.5, 4)
    motor_z = np.linspace(-0.3, 0.3, 2)
    mx, my, mz = np.meshgrid(motor_x, motor_y, motor_z)
    motor_pts = np.vstack([mx.ravel(), my.ravel(), mz.ravel()]).T

    # pump body (a cylinder-ish cluster at right)
    theta = np.linspace(0, 2*np.pi, 20)
    r = 0.35
    px = 0.8 + 0.5*np.cos(theta)
    py = 0.0 + r*np.sin(theta)
    pz = 0.0 + 0.2*np.sin(3*theta)  # a bit bumpy
    pump_pts = np.vstack([px, py, pz]).T

    # shaft points (a line rotated by angle)
    # base shaft points before rotation
    shaft_x = np.linspace(-0.2, 0.8, 10)
    shaft_y = np.zeros_like(shaft_x)
    shaft_z = np.zeros_like(shaft_x)

    # rotate shaft cross-section (simulate rotation by rotating small offsets)
    # We'll create 8 points around the shaft for visual "rotation" effect
    cross_r = 0.06
    cross_theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    shaft_ring_pts = []
    for xi in shaft_x:
        # ring around the shaft center, rotated by angle_rad
        xs = xi + cross_r * np.cos(cross_theta + angle_rad)
        ys = 0.1 * np.sin(cross_theta + angle_rad)  # slight y offsets
        zs = 0.02 * np.cos(cross_theta + angle_rad)
        pts = np.vstack([xs, ys, zs]).T
        shaft_ring_pts.append(pts)
    shaft_ring = np.vstack(shaft_ring_pts)

    return motor_pts, pump_pts, shaft_ring


def make_cylinder_mesh(center_x=0.0, center_y=0.0, center_z=0.0, radius=0.3, length=0.6, n_around=24):
    # helper to create a simple cylinder mesh (returns x,y,z,i,j,k)
    theta = np.linspace(0, 2*np.pi, n_around)
    z_top = center_z + length/2.0
    z_bot = center_z - length/2.0
    xs_top = center_x + radius * np.cos(theta)
    ys_top = center_y + radius * np.sin(theta)
    zs_top = np.full_like(theta, z_top)
    xs_bot = center_x + radius * np.cos(theta)
    ys_bot = center_y + radius * np.sin(theta)
    zs_bot = np.full_like(theta, z_bot)
    xs = np.concatenate([xs_top, xs_bot])
    ys = np.concatenate([ys_top, ys_bot])
    zs = np.concatenate([zs_top, zs_bot])
    # faces (triangles) between top and bottom ring
    i = []
    j = []
    k = []
    N = len(theta)
    for t in range(N-1):
        top_a = t
        top_b = t+1
        bot_a = t + N
        bot_b = t+1 + N
        i.append(top_a); j.append(top_b); k.append(bot_a)
        i.append(top_b); j.append(bot_b); k.append(bot_a)
    return xs, ys, zs, i, j, k


def make_box_mesh(center_x=0.0, center_y=0.0, center_z=0.0, dx=0.6, dy=0.5, dz=0.4):
    # create 8 corners of a box and return triangular faces for Mesh3d
    hx = dx / 2.0
    hy = dy / 2.0
    hz = dz / 2.0
    verts = np.array([
        [center_x-hx, center_y-hy, center_z-hz],
        [center_x+hx, center_y-hy, center_z-hz],
        [center_x+hx, center_y+hy, center_z-hz],
        [center_x-hx, center_y+hy, center_z-hz],
        [center_x-hx, center_y-hy, center_z+hz],
        [center_x+hx, center_y-hy, center_z+hz],
        [center_x+hx, center_y+hy, center_z+hz],
        [center_x-hx, center_y+hy, center_z+hz],
    ])
    xs = verts[:,0]
    ys = verts[:,1]
    zs = verts[:,2]
    # define 12 triangles (two per face)
    faces = [
        (0,1,2),(0,2,3),
        (4,5,6),(4,6,7),
        (0,1,5),(0,5,4),
        (1,2,6),(1,6,5),
        (2,3,7),(2,7,6),
        (3,0,4),(3,4,7)
    ]
    i = [f[0] for f in faces]
    j = [f[1] for f in faces]
    k = [f[2] for f in faces]
    return xs, ys, zs, i, j, k

# build Plotly figure for a single Processing Unit (per-PU rendering)
def build_figure_for_pu(pu_model, rpm, vibration, temp, flow, voltage, pressure, anomaly):
    """
    Render a single ProcessingUnit model with all its motors and individual connectors.
    pu_model: dict with keys 'name', 'motors' (list of motor dicts with 'name' and 'connectors' list).
    """
    now = time.time()
    revs_per_sec = (rpm or 0.0) / 60.0
    angle = 2 * np.pi * revs_per_sec * (now % 1.0)
    motor_pts, pump_pts, shaft_ring = create_geometry(angle_rad=angle)
    shaft_size = np.clip(1.5 + (rpm or 0.0) / 600.0, 1.5, 8.0)

    fig = go.Figure()

    # Position motors vertically (one per input, representing the processing unit internals)
    motors = pu_model.get('motors', [])
    num_motors = len(motors)
    
    motor_spacing = 0.9
    motor_centers_y = list(np.linspace(-motor_spacing*(max(1, num_motors)-1)/2,
                                       motor_spacing*(max(1, num_motors)-1)/2,
                                       max(1, num_motors)))
    motor_positions = []  # store (mx, my) for connector drawing
    motor_labels = []
    
    for idx, my in enumerate(motor_centers_y):
        mx = -0.6
        bx, by, bz, bi, bj, bk = make_box_mesh(center_x=mx, center_y=my, center_z=0.0, dx=0.6, dy=0.5, dz=0.4)
        color = 'orangered' if anomaly else 'lightsteelblue'
        fig.add_trace(go.Mesh3d(x=bx, y=by, z=bz, i=bi, j=bj, k=bk,
                                color=color, opacity=1.0, flatshading=True, name=f'Motor {idx+1}'))
        motor_positions.append((mx, my))
        motor_labels.append((mx, my, 0.45, f'Motor {idx+1}'))

    # Draw individual connectors from each motor to corresponding right-side outlet
    # Each motor has its own set of connectors (from pu_model.motors[idx].connectors)
    outlet_x = 0.8
    for motor_idx, (mx, my) in enumerate(motor_positions):
        connectors = motors[motor_idx].get('connectors', []) if motor_idx < len(motors) else []
        num_conns = len(connectors)
        if num_conns == 0:
            num_conns = 1  # default to 1 if not specified
        
        # Draw parallel connector lines for this motor
        conn_color = 'red' if anomaly else 'dodgerblue'
        z_offsets = np.linspace(-0.08, 0.08, num_conns)
        
        for ci, zoff in enumerate(z_offsets):
            xs = [mx + 0.4, outlet_x - 0.4]
            ys = [my, my]  # connectors stay at motor's Y level
            zs = [zoff, zoff]
            lw = 8 if num_conns == 1 else max(3, int(10 / num_conns))
            
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                       line=dict(width=lw, color=conn_color, dash='solid'),
                                       name=(f'Connector {motor_idx+1}' if ci == 0 else None),
                                       opacity=0.98, showlegend=(ci == 0)))
            # Endpoint markers
            fig.add_trace(go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode='markers',
                                       marker=dict(size=6, color=conn_color), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[xs[1]], y=[ys[1]], z=[zs[1]], mode='markers',
                                       marker=dict(size=6, color=conn_color), showlegend=False))

    # Draw a single outlet/pump body on the right (represents the PU output)
    pxs, pys, pzs, pi, pj, pk = make_cylinder_mesh(center_x=outlet_x, center_y=0.0, center_z=0.0,
                                                    radius=0.45, length=0.5, n_around=36)
    outlet_color = 'tomato' if anomaly else 'seagreen'
    fig.add_trace(go.Mesh3d(x=pxs, y=pys, z=pzs, i=pi, j=pj, k=pk,
                            color=outlet_color, opacity=0.95, flatshading=True, name='Outlet'))
    outlet_label = (outlet_x, 0.0, 0.45, 'Outlet')

    # labels: place readable text labels near each main component
    try:
        label_x = []
        label_y = []
        label_z = []
        label_text = []
        # add motors labels
        for (lx, ly, lz, txt) in motor_labels:
            label_x.append(lx)
            label_y.append(ly)
            label_z.append(lz)
            label_text.append(txt)
        # add outlet label
        label_x.append(outlet_label[0])
        label_y.append(outlet_label[1])
        label_z.append(outlet_label[2])
        label_text.append(outlet_label[3])
        # add connector label near shaft center
        label_x.append(0.3)
        label_y.append(0.0)
        label_z.append(0.12)
        label_text.append('Connector')

        colors = []
        for t in label_text:
            if t.startswith('Motor'):
                colors.append('steelblue' if not anomaly else 'orangered')
            elif t.startswith('Outlet'):
                colors.append('darkgreen' if not anomaly else 'tomato')
            else:
                colors.append('black')

        fig.add_trace(go.Scatter3d(x=label_x, y=label_y, z=label_z,
                                   mode='text', text=label_text,
                                   textfont=dict(size=14, color=colors),
                                   showlegend=False))
    except Exception:
        pass

    # shaft ring (shows rotation) with size varying by rpm
    fig.add_trace(go.Scatter3d(
        x=shaft_ring[:,0], y=shaft_ring[:,1], z=shaft_ring[:,2],
        mode='markers', marker=dict(size=shaft_size, color='black' if not anomaly else 'red', opacity=0.95),
        name='Shaft (rotating)'
    ))

    # add a subtle central shaft line
    fig.add_trace(go.Scatter3d(
        x=[-0.2, 0.8], y=[0,0], z=[0,0],
        mode='lines', line=dict(width=6, color='rgba(80,80,80,0.6)'), name='Shaft center'
    ))

    # add small cone blades around the shaft to indicate rotation (visual cue)
    try:
        # pick a subset of ring points to place cones
        blade_idx = np.linspace(0, shaft_ring.shape[0]-1, 8, dtype=int)
        bx = shaft_ring[blade_idx,0]
        by = shaft_ring[blade_idx,1]
        bz = shaft_ring[blade_idx,2]
        # tangential vectors for cones to convey rotation direction
        bx_vec = -np.sin(np.linspace(0, 2*np.pi, len(bx)) + angle)
        by_vec =  np.cos(np.linspace(0, 2*np.pi, len(by)) + angle)
        bz_vec = np.zeros_like(bx_vec)
        fig.add_trace(go.Cone(x=bx, y=by, z=bz, u=bx_vec, v=by_vec, w=bz_vec,
                              sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False,
                              colorscale='Hot', name='Blades', opacity=0.9))
    except Exception:
        # if Cone not available or fails, silently skip — visual enhancement only
        pass

    # annotation: numeric overlays
    title = (f"{pu_model.get('name', 'Processing Unit')} | "
             f"RPM: {rpm:.1f} | Vib: {vibration:.3f} | Temp: {temp:.2f}°C | Flow: {flow:.2f} | "
             f"Volt: {voltage:.2f}V | Pres: {pressure:.2f} | "
             f"{'ANOMALY' if anomaly else 'Normal'}")
    # nicer plot styling
    # nicer plot styling and camera for a more attractive 3D view
    camera = dict(eye=dict(x=1.6, y=1.2, z=0.6))
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='auto',
        bgcolor='rgba(245,245,250,0.0)',
        camera=camera
    ),
    margin=dict(l=0,r=0,t=48,b=0),
    paper_bgcolor='white',
    plot_bgcolor='white',
    title=dict(text=title, x=0.5, font=dict(size=14))
    )
    # subtle annotation for extra telemetry
    fig.add_annotation(dict(text=f"Volt:{voltage:.1f}V • Pres:{pressure:.2f}",
                            x=0.5, y=0.94, xref='paper', yref='paper', showarrow=False))
    return fig


# small helper to render metrics in a compact card style
def render_metrics(rpm, vibration, temp, flow, anomaly):
    # show primary metrics in a 4-column row
    cols = st.columns(4)
    bg_class = "card warn" if anomaly else "card"
    with cols[0]:
        st.markdown(f'<div class="{bg_class}"><div class="small"><strong>RPM</strong></div><div style="font-size:18px">{rpm:.1f}</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="{bg_class}"><div class="small"><strong>Vibration</strong></div><div style="font-size:18px">{vibration:.3f}</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="{bg_class}"><div class="small"><strong>Temp (°C)</strong></div><div style="font-size:18px">{temp:.1f}</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="{bg_class}"><div class="small"><strong>Flow</strong></div><div style="font-size:18px">{flow:.1f}</div></div>', unsafe_allow_html=True)

def render_extended_metrics(voltage, pressure, anomaly):
    # show extended metrics in a 2-column row
    cols = st.columns(2)
    bg_class = "card warn" if anomaly else "card"
    with cols[0]:
        st.markdown(f'<div class="{bg_class}"><div class="small"><strong>Voltage (V)</strong></div><div style="font-size:18px">{voltage:.2f}</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="{bg_class}"><div class="small"><strong>Pressure</strong></div><div style="font-size:18px">{pressure:.2f}</div></div>', unsafe_allow_html=True)

# initial render
with placeholder.container():
    st.write("Waiting for data... (start simulator & detector)")

# polling loop (drives updates)
if st.session_state.running3d:
    # do one iteration and request rerun to keep UI responsive
    latest = read_latest()
    if latest is None:
        with placeholder.container():
            st.info("No annotated data available yet. Ensure detect loop is running.")
    else:
        rpm = float(latest.get("rpm", 0.0))
        vibration = float(latest.get("vibration", 0.0))
        temp = float(latest.get("temp", 0.0))
        flow = float(latest.get("flow", 0.0))
        # new parameters
        voltage = float(latest.get("voltage", 0.0))
        pressure = float(latest.get("pressure", 0.0))
        anomaly = int(latest.get("anomaly", 0))
        
        # Rebuild PU models from current session state (ensures they're always up-to-date)
        pu_models_objs = build_pu_models_from_session()
        pu_models = [p.to_dict() for p in pu_models_objs]
        st.session_state['pu_models'] = pu_models
        
        # Render all processing units
        with placeholder.container():
            if len(pu_models) == 0:
                st.info("No processing unit models configured. Please set up PUs in the 'Processing Units' tab.")
            else:
                # Render each PU with its own heading and figure
                for pu_idx, pu_model in enumerate(pu_models):
                    st.subheader(f"{pu_model.get('name', f'Processing Unit {pu_idx+1}')}")
                    fig = build_figure_for_pu(pu_model, rpm, vibration, temp, flow, voltage, pressure, anomaly==1)
                    st.plotly_chart(fig, width='stretch')
                
                # Show metrics once below all PUs
                render_metrics(rpm, vibration, temp, flow, anomaly==1)
                render_extended_metrics(voltage, pressure, anomaly==1)
    time.sleep(poll)
    st.rerun()
else:
    # not running: show latest static snapshot if exists
    latest = read_latest()
    if latest is not None:
        rpm = float(latest.get("rpm", 0.0))
        vibration = float(latest.get("vibration", 0.0))
        temp = float(latest.get("temp", 0.0))
        flow = float(latest.get("flow", 0.0))
        # new telemetry fields
        voltage = float(latest.get("voltage", 0.0))
        pressure = float(latest.get("pressure", 0.0))
        anomaly = int(latest.get("anomaly", 0))
        
        # Rebuild PU models from current session state (ensures they're always up-to-date)
        pu_models_objs = build_pu_models_from_session()
        pu_models = [p.to_dict() for p in pu_models_objs]
        st.session_state['pu_models'] = pu_models
        
        with placeholder.container():
            if len(pu_models) == 0:
                st.info("No processing unit models configured. Please set up PUs in the 'Processing Units' tab.")
            else:
                # Render each PU with its own heading and figure
                for pu_idx, pu_model in enumerate(pu_models):
                    st.subheader(f"{pu_model.get('name', f'Processing Unit {pu_idx+1}')}")
                    fig = build_figure_for_pu(pu_model, rpm, vibration, temp, flow, voltage, pressure, anomaly==1)
                    st.plotly_chart(fig, width='stretch')
                
                # Show metrics once below all PUs
                render_metrics(rpm, vibration, temp, flow, anomaly==1)
                render_extended_metrics(voltage, pressure, anomaly==1)
    else:
        with placeholder.container():
            st.info("No snapshot to show. Start the pipeline.")
