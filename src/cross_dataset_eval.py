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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import Bearing1DCNN


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 4
WINDOW_SIZE = 1024

CLASS_NAMES = [
    "Normal",
    "Inner Race",
    "Ball",
    "Outer Race"
]

LABEL_MAP = {
    "Normal": 0,
    "IR007": 1,
    "B007": 2,
    "OR007": 3
}


# ============================================================
# LABEL INFERENCE
# ============================================================

def get_label_from_filename(file_path):

    filename = os.path.basename(file_path).upper()

    # Order matters:
    # OR007 must be checked before generic "O"
    if filename.startswith("NORMAL"):
        return 0

    if filename.startswith("IR007"):
        return 1

    if filename.startswith("B007"):
        return 2

    if filename.startswith("OR007"):
        return 3

    return None


# ============================================================
# FIND SIGNAL
# ============================================================

def extract_signal(mat_data):

    """
    Prefer DE_time because the training dataset loader also
    prefers DE_time.

    If DE_time is unavailable, use the largest numeric array.
    """

    # --------------------------------------------------------
    # First preference: DE_time
    # --------------------------------------------------------

    de_keys = [
        key
        for key in mat_data.keys()
        if "DE_time" in key
    ]

    if de_keys:

        signal = np.asarray(
            mat_data[de_keys[0]]
        ).flatten()

        return signal

    # --------------------------------------------------------
    # Fallback: largest numeric array
    # --------------------------------------------------------

    candidates = []

    for key, value in mat_data.items():

        if key.startswith("__"):
            continue

        try:

            arr = np.asarray(value)

            if np.issubdtype(
                arr.dtype,
                np.number
            ):

                if arr.size > 1000:

                    candidates.append(
                        (arr.size, arr)
                    )

        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0][1].flatten()


# ============================================================
# NORMALIZE WINDOW
# ============================================================

def normalize_window(window):

    window = window.astype(
        np.float32
    )

    mean = np.mean(window)

    std = np.std(window)

    std = std + 1e-8

    window = (
        window - mean
    ) / std

    return window


# ============================================================
# LOAD AND SEGMENT ONE MAT FILE
# ============================================================

def process_mat_file(
    file_path,
    label
):

    try:

        mat_data = loadmat(
            file_path
        )

    except Exception as e:

        print(
            f"  ERROR loading "
            f"{os.path.basename(file_path)}: {e}"
        )

        return [], []

    signal = extract_signal(
        mat_data
    )

    if signal is None:

        print(
            f"  SKIPPED: "
            f"{os.path.basename(file_path)} "
            f"(no suitable signal found)"
        )

        return [], []

    signal = np.asarray(
        signal,
        dtype=np.float32
    ).flatten()

    total_length = len(signal)

    num_windows = (
        total_length // WINDOW_SIZE
    )

    if num_windows == 0:

        print(
            f"  SKIPPED: "
            f"{os.path.basename(file_path)} "
            f"(signal too short)"
        )

        return [], []

    windows = []
    labels = []

    for i in range(num_windows):

        start = (
            i * WINDOW_SIZE
        )

        end = (
            start + WINDOW_SIZE
        )

        window = signal[
            start:end
        ]

        if len(window) != WINDOW_SIZE:
            continue

        window = normalize_window(
            window
        )

        windows.append(
            window
        )

        labels.append(
            label
        )

    print(
        f"  Loaded: "
        f"{os.path.basename(file_path)} | "
        f"Signal length: {total_length} | "
        f"Windows: {len(windows)} | "
        f"Label: {label} "
        f"({CLASS_NAMES[label]})"
    )

    return windows, labels


# ============================================================
# LOAD COMPLETE EXTERNAL DATASET
# ============================================================

def load_external_dataset(
    raw_dir
):

    print("\n" + "=" * 70)
    print("LOADING EXTERNAL DATASET")
    print("=" * 70)

    mat_files = sorted(
        glob.glob(
            os.path.join(
                raw_dir,
                "*.mat"
            )
        )
    )

    if not mat_files:

        raise RuntimeError(
            f"No .mat files found in: "
            f"{raw_dir}"
        )

    print(
        f"\nFound {len(mat_files)} .mat files"
    )

    all_windows = []
    all_labels = []
    file_names = []

    skipped = 0

    for file_path in mat_files:

        label = get_label_from_filename(
            file_path
        )

        if label is None:

            print(
                f"  SKIPPED: "
                f"{os.path.basename(file_path)} "
                f"(unknown label)"
            )

            skipped += 1

            continue

        windows, labels = process_mat_file(
            file_path,
            label
        )

        if not windows:

            skipped += 1

            continue

        all_windows.extend(
            windows
        )

        all_labels.extend(
            labels
        )

        file_names.extend(
            [os.path.basename(file_path)]
            * len(windows)
        )

    if not all_windows:

        raise RuntimeError(
            "No valid windows extracted."
        )

    X = np.asarray(
        all_windows,
        dtype=np.float32
    )

    y = np.asarray(
        all_labels,
        dtype=np.int64
    )

    print("\n" + "-" * 70)

    print(
        f"Final X shape: {X.shape}"
    )

    print(
        f"Final y shape: {y.shape}"
    )

    print(
        f"Skipped files: {skipped}"
    )

    print("\nClass distribution:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        count = np.sum(
            y == class_id
        )

        print(
            f"  {class_id} - "
            f"{class_name:<12}: "
            f"{count}"
        )

    return X, y, file_names


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path,
    device
):

    print("\n" + "=" * 70)
    print("LOADING TRAINED CNN MODEL")
    print("=" * 70)

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found: "
            f"{model_path}"
        )

    model = Bearing1DCNN(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Support both state_dict and checkpoint dictionaries
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"Model loaded successfully:"
    )

    print(
        model_path
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    X,
    device,
    batch_size=32
):

    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    predictions = []
    probabilities = []

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    # CNN input:
    # [N, 1, 1024]

    X_tensor = X_tensor.unsqueeze(
        1
    )

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X_tensor)
            )

            batch = X_tensor[
                start:end
            ]

            batch = batch.to(
                device
            )

            outputs = model(
                batch
            )

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

    predictions = np.asarray(
        predictions
    )

    probabilities = np.asarray(
        probabilities
    )

    return (
        predictions,
        probabilities
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    y_true,
    y_pred
):

    print("\n" + "=" * 70)
    print("CROSS-DATASET EVALUATION")
    print("=" * 70)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            zero_division=0
        )
    )

    macro_precision = np.mean(
        precision
    )

    macro_recall = np.mean(
        recall
    )

    macro_f1 = np.mean(
        f1
    )

    print(
        f"\nOverall Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision  : "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall     : "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1         : "
        f"{macro_f1 * 100:.2f}%"
    )

    print("\nPer-Class Results:")

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"\n{class_name}"
        )

        print(
            f"  Precision : "
            f"{precision[i] * 100:.2f}%"
        )

        print(
            f"  Recall    : "
            f"{recall[i] * 100:.2f}%"
        )

        print(
            f"  F1 Score  : "
            f"{f1[i] * 100:.2f}%"
        )

        print(
            f"  Support   : "
            f"{support[i]}"
        )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(
        "\nClassification Report:"
    )

    print(report)

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
        "classification_report": report
    }

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    output_dir,
    file_names,
    y_true,
    y_pred,
    probabilities,
    metrics
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Prediction CSV
    # --------------------------------------------------------

    result_data = {
        "file": file_names,
        "true_label": y_true,
        "true_class": [
            CLASS_NAMES[i]
            for i in y_true
        ],
        "predicted_label": y_pred,
        "predicted_class": [
            CLASS_NAMES[i]
            for i in y_pred
        ],
        "confidence": np.max(
            probabilities,
            axis=1
        )
    }

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        result_data[
            f"prob_{class_name.replace(' ', '_')}"
        ] = probabilities[:, i]

    predictions_df = pd.DataFrame(
        result_data
    )

    predictions_path = os.path.join(
        output_dir,
        "cross_dataset_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_rows = []

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        summary_rows.append({
            "class": class_name,
            "precision": metrics["precision"][i],
            "recall": metrics["recall"][i],
            "f1": metrics["f1"][i],
            "support": metrics["support"][i]
        })

    summary_rows.append({
        "class": "Macro Average",
        "precision": metrics["macro_precision"],
        "recall": metrics["macro_recall"],
        "f1": metrics["macro_f1"],
        "support": len(y_true)
    })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        output_dir,
        "cross_dataset_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm_path = os.path.join(
        output_dir,
        "confusion_matrix.npy"
    )

    np.save(
        cm_path,
        metrics["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_path = os.path.join(
        output_dir,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CROSS-DATASET EVALUATION\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{metrics['accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"Macro Precision: "
            f"{metrics['macro_precision'] * 100:.2f}%\n"
        )

        f.write(
            f"Macro Recall: "
            f"{metrics['macro_recall'] * 100:.2f}%\n"
        )

        f.write(
            f"Macro F1: "
            f"{metrics['macro_f1'] * 100:.2f}%\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(
                metrics["confusion_matrix"]
            )
        )

        f.write(
            "\n\nClassification Report:\n"
        )

        f.write(
            metrics["classification_report"]
        )

    # --------------------------------------------------------
    # NumPy arrays
    # --------------------------------------------------------

    np.save(
        os.path.join(
            output_dir,
            "y_true.npy"
        ),
        y_true
    )

    np.save(
        os.path.join(
            output_dir,
            "y_pred.npy"
        ),
        y_pred
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"\nPrediction CSV:"
    )

    print(
        predictions_path
    )

    print(
        f"\nSummary CSV:"
    )

    print(
        summary_path
    )

    print(
        f"\nClassification report:"
    )

    print(
        report_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "CROSS-DATASET BEARING FAULT EVALUATION"
    )

    print("=" * 70)

    print(
        "\nPreprocessing used:"
    )

    print(
        "  Signal source : DE_time"
    )

    print(
        "  Resampling    : None"
    )

    print(
        "  Window size   : 1024"
    )

    print(
        "  Normalization : Per-window Z-score"
    )

    print(
        "  Model         : Bearing1DCNN"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    raw_dir = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "raw"
    )

    model_path = os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "bearing_cnn.pth"
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "cross_dataset_results"
    )

    print(
        f"\nRaw data directory:"
    )

    print(
        raw_dir
    )

    print(
        f"\nModel:"
    )

    print(
        model_path
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load external dataset
    # --------------------------------------------------------

    X, y_true, file_names = (
        load_external_dataset(
            raw_dir
        )
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_model(
        model_path,
        device
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred, probabilities = predict(
        model,
        X,
        device
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        output_dir,
        file_names,
        y_true,
        y_pred,
        probabilities,
        metrics
    )

    print("\n" + "=" * 70)

    print(
        "CROSS-DATASET EVALUATION COMPLETED"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()