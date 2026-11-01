import os
import sys
import glob
import numpy as np
import pandas as pd
import torch

from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "cross_dataset",
    "paderborn"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "bearing_cnn.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "paderborn_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add src to Python path
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import Bearing1DCNN


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 1024

# Paderborn selected classes
CLASS_MAPPING = {
    "K001": 0,   # Normal
    "KI04": 1,   # Inner Race
    "KA04": 3    # Outer Race
}

CLASS_NAMES = {
    0: "Normal",
    1: "Inner Race",
    3: "Outer Race"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PADERBORN EXTERNAL DATASET EVALUATION")
print("=" * 70)

print("\nPreprocessing:")
print("  Dataset          : Paderborn")
print("  Signal           : Mech_4kHz")
print("  Resampling       : None")
print("  Window size      :", WINDOW_SIZE)
print("  Normalization    : Per-window Z-score")
print("  Model            : Bearing1DCNN")

print("\nDataset directory:")
print(DATASET_DIR)

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


# ============================================================
# EXTRACT MECH_4KHZ SIGNAL
# ============================================================

def extract_mech_signal(mat_path):

    mat_data = loadmat(mat_path)

    # Paderborn MAT file contains one main structure.
    main_keys = [
        key for key in mat_data.keys()
        if not key.startswith("__")
    ]

    if not main_keys:
        raise ValueError("No valid MATLAB structure found.")

    root = mat_data[main_keys[0]]

    # Root structure contains X
    if "X" not in root.dtype.names:
        raise ValueError("X field not found in Paderborn structure.")

    X = root["X"][0, 0]

    # X contains multiple measurement channels
    if "Raster" not in X.dtype.names:
        raise ValueError("Raster field not found.")

    rasters = X["Raster"][0]

    mech_index = None

    for i in range(len(rasters)):

        raster_value = rasters[i]

        if isinstance(raster_value, np.ndarray):
            if raster_value.size > 0:
                raster_value = str(raster_value.flatten()[0])
            else:
                raster_value = ""
        else:
            raster_value = str(raster_value)

        if "Mech_4kHz" in raster_value:
            mech_index = i
            break

    if mech_index is None:
        raise ValueError(
            "Mech_4kHz channel not found in: "
            + os.path.basename(mat_path)
        )

    signal = X["Data"][0, mech_index]

    signal = np.asarray(signal).astype(np.float32).flatten()

    return signal


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(signal, window_size=1024):

    num_windows = len(signal) // window_size

    if num_windows == 0:
        return np.empty((0, window_size), dtype=np.float32)

    signal = signal[:num_windows * window_size]

    windows = signal.reshape(num_windows, window_size)

    return windows.astype(np.float32)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_windows(X):

    mean = np.mean(
        X,
        axis=1,
        keepdims=True
    )

    std = np.std(
        X,
        axis=1,
        keepdims=True
    )

    std = std + 1e-8

    X_norm = (X - mean) / std

    X_norm = np.expand_dims(
        X_norm,
        axis=1
    )

    return X_norm.astype(np.float32)


# ============================================================
# LOAD PADERBORN DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING PADERBORN DATASET")
print("=" * 70)

all_X = []
all_y = []
all_files = []

total_files = 0
skipped_files = 0


for folder_name, label in CLASS_MAPPING.items():

    folder_path = os.path.join(
        DATASET_DIR,
        folder_name
    )

    if not os.path.isdir(folder_path):

        print(
            f"\nWARNING: Folder not found: {folder_name}"
        )

        continue

    mat_files = sorted(
        glob.glob(
            os.path.join(
                folder_path,
                "*.mat"
            )
        )
    )

    print(
        f"\n{folder_name} -> "
        f"{CLASS_NAMES[label]}"
    )

    for mat_path in mat_files:

        total_files += 1

        try:

            signal = extract_mech_signal(mat_path)

            windows = create_windows(
                signal,
                WINDOW_SIZE
            )

            if len(windows) == 0:

                print(
                    "  SKIPPED:",
                    os.path.basename(mat_path),
                    "| Signal too short"
                )

                skipped_files += 1
                continue

            all_X.append(windows)

            all_y.extend(
                [label] * len(windows)
            )

            all_files.extend(
                [os.path.basename(mat_path)] *
                len(windows)
            )

            print(
                f"  Loaded: "
                f"{os.path.basename(mat_path)} "
                f"| Signal length: {len(signal)} "
                f"| Windows: {len(windows)}"
            )

        except Exception as e:

            print(
                f"  ERROR: "
                f"{os.path.basename(mat_path)} "
                f"| {e}"
            )

            skipped_files += 1


# ============================================================
# COMBINE DATA
# ============================================================

if not all_X:

    raise RuntimeError(
        "No valid Paderborn vibration data found."
    )

X = np.concatenate(
    all_X,
    axis=0
)

y = np.asarray(
    all_y,
    dtype=np.int64
)

all_files = np.asarray(
    all_files
)


print("\n" + "-" * 70)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

print("Total files:", total_files)
print("Skipped files:", skipped_files)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

for label in [0, 1, 3]:

    count = np.sum(y == label)

    print(
        f"  {label} - "
        f"{CLASS_NAMES[label]:12s}: "
        f"{count}"
    )


# ============================================================
# NORMALIZE
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING SIGNALS")
print("=" * 70)

X = normalize_windows(X)

print("Normalized X shape:", X.shape)


# ============================================================
# LOAD CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED CNN")
print("=" * 70)

model = Bearing1DCNN(
    num_classes=4
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support both plain state_dict and checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

batch_size = 64

predictions = []
probabilities = []


with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        batch_size
    ):

        batch = X_tensor[
            start:start + batch_size
        ].to(device)

        outputs = model(batch)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        probabilities.extend(
            probs.cpu().numpy()
        )


y_pred = np.asarray(
    predictions,
    dtype=np.int64
)

probabilities = np.asarray(
    probabilities
)


# ============================================================
# FILTER TO EVALUATED CLASSES
# ============================================================

# The model has 4 classes, but Paderborn selected subset
# contains only Normal, Inner Race and Outer Race.

valid_indices = np.isin(
    y_pred,
    [0, 1, 3]
)

# We DO NOT remove samples based on predictions here.
# All samples remain part of the evaluation.

# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 70)
print("PADERBORN EXTERNAL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(
    y,
    y_pred
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y,
        y_pred,
        labels=[0, 1, 3],
        average=None,
        zero_division=0
    )
)

macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)


print(
    f"\nOverall Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision  : {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall     : {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1         : {macro_f1 * 100:.2f}%"
)


# ============================================================
# PER CLASS RESULTS
# ============================================================

print("\nPer-Class Results:")

for i, label in enumerate([0, 1, 3]):

    print(f"\n{CLASS_NAMES[label]}")

    print(
        f"  Precision : {precision[i] * 100:.2f}%"
    )

    print(
        f"  Recall    : {recall[i] * 100:.2f}%"
    )

    print(
        f"  F1 Score  : {f1[i] * 100:.2f}%"
    )

    print(
        f"  Support   : {support[i]}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    y_pred,
    labels=[0, 1, 3]
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y,
    y_pred,
    labels=[0, 1, 3],
    target_names=[
        "Normal",
        "Inner Race",
        "Outer Race"
    ],
    digits=4,
    zero_division=0
)

print("\nClassification Report:")

print(report)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "file": all_files,

    "true_label": y,

    "true_class": [
        CLASS_NAMES[int(v)]
        for v in y
    ],

    "predicted_label": y_pred,

    "predicted_class": [
        CLASS_NAMES.get(
            int(v),
            f"Class {v}"
        )
        for v in y_pred
    ],

    "confidence": np.max(
        probabilities,
        axis=1
    )
})


prediction_path = os.path.join(
    OUTPUT_DIR,
    "paderborn_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({

    "metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],

    "value": [
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1
    ]
})


summary_path = os.path.join(
    OUTPUT_DIR,
    "paderborn_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PADERBORN EXTERNAL DATASET EVALUATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Macro Precision: {macro_precision:.6f}\n"
    )

    f.write(
        f"Macro Recall: {macro_recall:.6f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.6f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(report)


# ============================================================
# SAVE RAW ARRAYS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_true.npy"
    ),
    y
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_pred.npy"
    ),
    y_pred
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print("\nPrediction CSV:")
print(prediction_path)

print("\nSummary CSV:")
print(summary_path)

print("\nConfusion Matrix:")
print(cm_path)

print("\nClassification Report:")
print(report_path)

print("\n" + "=" * 70)
print("PADERBORN EXTERNAL EVALUATION COMPLETED")
print("=" * 70)