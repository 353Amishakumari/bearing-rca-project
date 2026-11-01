# -*- coding: utf-8 -*-
"""
motor_dataset_loader.py
-----------------------
Extracts and curates a balanced experimental subset of 16-bit radiometric
thermal images from the Squirrel-Cage Induction Motor Fault Diagnosis Dataset
(Zenodo: 10.5281/zenodo.8203070).
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = Path(r"C:\Users\KIIT\Downloads\thermal_images (1).zip")
TARGET_DIR = PROJECT_ROOT / "dataset" / "motor_thermal"
INDEX_CSV = PROJECT_ROOT / "dataset" / "processed" / "motor_thermal_index.csv"

# Fault Category Mapping
CATEGORY_MAP = {
    "proper": "Normal Operation",
    "misalignment": "Shaft Misalignment",
    "rotor": "Broken Rotor Bars"
}

def classify_experiment(condition_name):
    cond_lower = condition_name.lower()
    if "misalignment" in cond_lower:
        return 1, "Shaft Misalignment"
    elif "rotor" in cond_lower:
        return 2, "Broken Rotor Bars"
    elif "current-load" in cond_lower or "start-up" in cond_lower:
        return 0, "Normal Operation"
    return 0, "Normal Operation"

def extract_load_current(condition_name):
    for part in condition_name.split("-"):
        if part.endswith("A") and part[:-1].replace(".", "").isdigit():
            return float(part[:-1])
    return 0.0

def extract_curated_dataset(samples_per_condition=15):
    """
    Extracts a balanced selection of 16-bit radiometric PNG frames
    for each condition, coupling, and camera type.
    """
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Source zip not found at: {ZIP_PATH}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "dataset" / "processed").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Inspecting zip archive: {ZIP_PATH}")
    records = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_pngs = [n for n in z.namelist() if n.endswith(".png")]
        print(f"[INFO] Found {len(all_pngs)} total PNG files in archive.")

        from collections import defaultdict
        groups = defaultdict(list)

        for name in all_pngs:
            parts = name.split("/")
            if len(parts) >= 4:
                coupling = parts[1]
                condition = parts[2]
                camera = parts[3]
                groups[(coupling, condition, camera)].append(name)

        print(f"[INFO] Found {len(groups)} distinct experiment branches.")
        extracted_count = 0

        for (coupling, condition, camera), filenames in sorted(groups.items()):
            class_id, class_name = classify_experiment(condition)
            load_current = extract_load_current(condition)

            sorted_files = sorted(filenames)
            total_avail = len(sorted_files)
            step = max(1, total_avail // samples_per_condition)
            selected = sorted_files[::step][:samples_per_condition]

            rel_dest_dir = Path(coupling) / condition / camera
            full_dest_dir = TARGET_DIR / rel_dest_dir
            full_dest_dir.mkdir(parents=True, exist_ok=True)

            for src_name in selected:
                filename = Path(src_name).name
                dest_path = full_dest_dir / filename
                
                if not dest_path.exists():
                    with open(dest_path, "wb") as f_out:
                        f_out.write(z.read(src_name))

                ts_str = filename.replace(".png", "")
                timestamp_ns = int(ts_str) if ts_str.isdigit() else 0

                records.append({
                    "rel_path": str(rel_dest_dir / filename).replace("\\", "/"),
                    "filename": filename,
                    "coupling": coupling,
                    "condition": condition,
                    "camera": camera,
                    "class_id": class_id,
                    "class_name": class_name,
                    "load_current_A": load_current,
                    "timestamp_ns": timestamp_ns
                })
                extracted_count += 1

    df = pd.DataFrame(records)
    df.to_csv(INDEX_CSV, index=False)
    print(f"[SUCCESS] Extraction complete!")
    print(f"Total curated images: {len(df)}")
    print(f"Index CSV saved to: {INDEX_CSV}")
    print("Class Distribution:")
    print(df["class_name"].value_counts())
    print("Camera Distribution:")
    print(df["camera"].value_counts())
    return df

if __name__ == "__main__":
    df = extract_curated_dataset(samples_per_condition=15)
