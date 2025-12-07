#!/usr/bin/env python3
# sensor_sim.py
import time, csv, random, math, argparse, sys, os
from datetime import datetime

OUTFILE = "data.csv"
ALERTS_FILE = "alerts.json"

def write_header():
    if not os.path.exists(OUTFILE) or os.path.getsize(OUTFILE)==0:
        with open(OUTFILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","rpm","vibration","temp","flow","voltage","pressure","anomaly"])

def generate_row(t, normal=True, force_fault=None):
    # use local timezone-aware timestamp so displayed times match the host device
    ts = datetime.now().astimezone().isoformat()
    rpm = 1450 + 20*math.sin(t/20.0) + random.gauss(0,5)
    vibration = 0.6 + 0.1*math.sin(t/5.0) + random.gauss(0,0.03)
    temp = 35 + 0.5*math.sin(t/50.0) + random.gauss(0,0.3)
    flow = 120 + 3*math.sin(t/15.0) + random.gauss(0,1.5)
    # additional telemetry
    voltage = 230 + 1.0*math.sin(t/60.0) + random.gauss(0,0.5)
    pressure = 3.0 + 0.05*math.sin(t/30.0) + random.gauss(0,0.05)

    anomaly = 0
    # inject fault if forced or random when normal==False
    if force_fault:
        kind = force_fault
    else:
        kind = None
        if not normal and random.random() < 0.05:
            kind = random.choice(["vib_spike","flow_drop","overheat"])
    if kind == "vib_spike":
        vibration += random.uniform(2.0,4.0); fault = 1
    elif kind == "flow_drop":
        flow -= random.uniform(30,60); fault = 2
    elif kind == "overheat":
        temp += random.uniform(15,30); fault = 3
    # map to anomaly variable for CSV
    if kind == "vib_spike":
        anomaly = 1
    elif kind == "flow_drop":
        anomaly = 2
    elif kind == "overheat":
        anomaly = 3

    return [ts, round(rpm,2), round(vibration,3), round(temp,2), round(flow,2), round(voltage,2), round(pressure,3), anomaly]

def append_row(row):
    with open(OUTFILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)
    # if the row indicates an anomaly, also write a human-friendly alert with suggestion
    try:
        anomaly = int(row[-1])
    except Exception:
        anomaly = 0
    if anomaly and anomaly > 0:
        try:
            write_alert(row, anomaly)
        except Exception:
            # don't let alert writing break the simulator
            pass

def write_alert(row, anomaly):
    """Append an alert entry to ALERTS_FILE with a suggested action based on anomaly type."""
    # row format: [ts, rpm, vibration, temp, flow, voltage, pressure, anomaly]
    suggestion_map = {
        1: ("Vibration spike detected — possible imbalance, loosened mounts, or bearing failure. Reduce RPM, inspect mounts and bearings, schedule maintenance.", "warning"),
        2: ("Flow drop detected — possible blockage, valve closed, or pump wear. Check inlet filters, valves, piping, and pump priming.", "critical"),
        3: ("Overheat detected — possible cooling failure or overload. Check cooling system, ventilation, reduce load, inspect motor winding temps.", "critical"),
    }
    suggestion, severity = suggestion_map.get(anomaly, ("Unknown anomaly detected. Investigate immediately.", "warning"))

    ts = row[0] if len(row) > 0 else datetime.now().astimezone().isoformat()
    try:
        rpm = float(row[1])
    except Exception:
        rpm = None
    try:
        vibration = float(row[2])
    except Exception:
        vibration = None
    try:
        temp = float(row[3])
    except Exception:
        temp = None
    try:
        flow = float(row[4])
    except Exception:
        flow = None
    try:
        voltage = float(row[5])
    except Exception:
        voltage = None
    try:
        pressure = float(row[6])
    except Exception:
        pressure = None

    alert = {
        "timestamp": ts,
        "rpm": rpm,
        "vibration": vibration,
        "temp": temp,
        "flow": flow,
        "voltage": voltage,
        "pressure": pressure,
        "anomaly": int(anomaly),
        "severity": severity,
        "suggestion": suggestion,
    }

    # load existing alerts, append, and save (keep most recent 200)
    try:
        if os.path.exists(ALERTS_FILE):
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
    except Exception:
        existing = []
    existing.append(alert)
    # cap alert history
    if len(existing) > 200:
        existing = existing[-200:]
    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

def run_stream(interval=0.5, normal=True, duration=None):
    write_header()
    start = time.time()
    t = 0.0
    try:
        while True:
            if duration and (time.time()-start) > duration:
                break
            row = generate_row(t, normal=normal)
            append_row(row)
            t += interval
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Simulator stopped by user.")

def inject_once(kind="vib_spike"):
    write_header()
    row = generate_row(0, normal=False, force_fault=kind)
    append_row(row)
    print("Injected row:", row)

def main():
    p = argparse.ArgumentParser(description="Sensor simulator for SmartTwin")
    p.add_argument("--mode", choices=["normal","fault"], default="normal", help="normal: mostly normal data; fault: occasionally inject faults")
    p.add_argument("--duration", type=float, default=None, help="duration in seconds (optional)")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between records")
    p.add_argument("--inject-once", choices=["vib_spike","flow_drop","overheat"], help="append one faulty data row and exit (useful for UI tests)")
    args = p.parse_args()

    if args.inject_once:
        inject_once(args.inject_once)
        sys.exit(0)

    normal = (args.mode=="normal")
    print(f"Starting simulator mode={args.mode} interval={args.interval}s duration={args.duration}")
    run_stream(interval=args.interval, normal=normal, duration=args.duration)

if __name__ == "__main__":
    main()
