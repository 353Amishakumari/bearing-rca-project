"""
generate_scada_logs.py
------------------------
SCADA text logs generate karta hai — Phase 3 (Multimodal Fusion) ke liye.

Kaam: har bearing fault class ke liye realistic SCADA-style maintenance
log text banata hai (temperature, vibration RMS, motor current, timestamp
ke saath), aur X_segments.npy / y_labels.npy jaisa hi count + order rakhta
hai — taaki baad me index-wise (sensor <-> SCADA text <-> image) fusion
karna easy ho.

Output:
    dataset/processed/scada_logs.csv   -> columns: index, label, log_text
    dataset/processed/scada_logs.json  -> same data, list-of-dicts form
                                           (HuggingFace datasets ke liye easy)

Run: python src/generate_scada_logs.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ------------------------------------------------------------------
# Config — actual class distribution match 
# (Normal=474, Inner Race=119, Ball=118, Outer Race=118 -> total 829)
# ------------------------------------------------------------------
DATA_DIR = "dataset/processed"
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------
# # Get the exact count of each class from y_labels.npy
# so the SCADA logs have the same distribution as the sensor data
# ------------------------------------------------------------------
def get_label_counts():
    y_path = os.path.join(DATA_DIR, "y_labels.npy")
    if os.path.exists(y_path):
        y = np.load(y_path)
        counts = {c: int(np.sum(y == c)) for c in range(len(CLASS_NAMES))}
        print(f"y_labels.npy se counts liye: {counts}")
        return y  # actual order bhi return karo, taaki index align rahe
    else:
        # Fallback (agar y_labels.npy na mile) — screenshot ke numbers use karo
        print("⚠️ y_labels.npy nahi mila, default counts use kar rahe hain.")
        y = np.array([0]*474 + [1]*119 + [2]*118 + [3]*118)
        np.random.shuffle(y)
        return y


# ------------------------------------------------------------------
# 2. Per-class "vocabulary" — realistic phrase templates
#    (isse har log thoda alag lagega, bilkul copy-paste jaisa nahi)
# ------------------------------------------------------------------
TEMPLATES = {
    0: {  # Normal
        "phrases": [
            "All parameters within nominal range.",
            "No anomaly detected during routine SCADA sweep.",
            "Vibration and thermal readings stable across shift.",
            "Motor operating within standard tolerance band.",
        ],
        "temp_range": (35.0, 48.0),
        "vib_rms_range": (0.05, 0.15),
        "current_range": (4.5, 5.5),
    },
    1: {  # Inner Race Fault
        "phrases": [
            "Elevated vibration signature detected near shaft inner race region.",
            "Periodic impact pattern observed in vibration spectrum, consistent with inner race defect.",
            "Slight radial runout deviation flagged by monitoring system.",
            "Recurring high-frequency spikes logged, correlating with inner race fault signature.",
        ],
        "temp_range": (52.0, 68.0),
        "vib_rms_range": (0.35, 0.60),
        "current_range": (5.6, 6.4),
    },
    2: {  # Ball Fault
        "phrases": [
            "Irregular impulsive vibration bursts detected, indicative of rolling element defect.",
            "Ball-pass frequency harmonics elevated above baseline threshold.",
            "Intermittent noise and vibration spikes logged during operation.",
            "Contamination-related wear pattern suspected based on vibration signature.",
        ],
        "temp_range": (55.0, 70.0),
        "vib_rms_range": (0.40, 0.65),
        "current_range": (5.7, 6.6),
    },
    3: {  # Outer Race Fault
        "phrases": [
            "Consistent high-amplitude vibration detected at outer race characteristic frequency.",
            "Repetitive impact signature logged, aligned with outer race fault pattern.",
            "Load zone vibration levels significantly above normal operating baseline.",
            "Elevated structural vibration transmitted through housing, outer race fault suspected.",
        ],
        "temp_range": (58.0, 72.0),
        "vib_rms_range": (0.45, 0.70),
        "current_range": (5.8, 6.8),
    },
}


# ------------------------------------------------------------------
# 3. SCADA log entry generate 
# ------------------------------------------------------------------
def generate_log_entry(class_id, idx, base_time):
    cfg = TEMPLATES[class_id]
    fault_name = CLASS_NAMES[class_id]

    temp = round(random.uniform(*cfg["temp_range"]), 1)
    vib_rms = round(random.uniform(*cfg["vib_rms_range"]), 3)
    current = round(random.uniform(*cfg["current_range"]), 2)
    phrase = random.choice(cfg["phrases"])
    timestamp = (base_time + timedelta(minutes=idx * 3)).strftime("%Y-%m-%d %H:%M:%S")

    log_text = (
        f"[{timestamp}] SCADA-LOG | Sensor Node: BRG-{idx:04d} | "
        f"Bearing Temp: {temp}°C | Vibration RMS: {vib_rms}g | "
        f"Motor Current: {current}A | Status: {fault_name}. {phrase}"
    )
    return log_text, temp, vib_rms, current, timestamp


# ------------------------------------------------------------------
# 4. Main — all logs generate & save 
# ------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    y = get_label_counts()

    base_time = datetime(2026, 1, 1, 6, 0, 0)
    records = []

    for idx, class_id in enumerate(y):
        class_id = int(class_id)
        log_text, temp, vib_rms, current, timestamp = generate_log_entry(class_id, idx, base_time)
        records.append({
            "index": idx,
            "label": class_id,
            "class_name": CLASS_NAMES[class_id],
            "timestamp": timestamp,
            "bearing_temp_c": temp,
            "vibration_rms_g": vib_rms,
            "motor_current_a": current,
            "log_text": log_text,
        })

    df = pd.DataFrame(records)

    csv_path = os.path.join(DATA_DIR, "scada_logs.csv")
    json_path = os.path.join(DATA_DIR, "scada_logs.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n✅ {len(records)} SCADA log entries generated.")
    print(f"✅ Saved: {csv_path}")
    print(f"✅ Saved: {json_path}")
    print("\nSample entries:")
    for cid in range(len(CLASS_NAMES)):
        sample = df[df["label"] == cid].iloc[0]
        print(f"\n[{CLASS_NAMES[cid]}]")
        print(sample["log_text"])


if __name__ == "__main__":
    main()