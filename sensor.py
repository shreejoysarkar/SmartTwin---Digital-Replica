#!/usr/bin/env python3
# sensor_sim.py
import time, csv, random, math, argparse, sys, os
from datetime import datetime

OUTFILE = "data.csv"

def write_header():
    if not os.path.exists(OUTFILE) or os.path.getsize(OUTFILE)==0:
        with open(OUTFILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","rpm","vibration","temp","flow","fault"])

def generate_row(t, normal=True, force_fault=None):
    ts = datetime.utcnow().isoformat()
    rpm = 1450 + 20*math.sin(t/20.0) + random.gauss(0,5)
    vibration = 0.6 + 0.1*math.sin(t/5.0) + random.gauss(0,0.03)
    temp = 35 + 0.5*math.sin(t/50.0) + random.gauss(0,0.3)
    flow = 120 + 3*math.sin(t/15.0) + random.gauss(0,1.5)
    fault = 0
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
    return [ts, round(rpm,2), round(vibration,3), round(temp,2), round(flow,2), fault]

def append_row(row):
    with open(OUTFILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

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
