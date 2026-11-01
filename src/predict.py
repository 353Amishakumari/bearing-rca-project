import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(CURRENT_DIR)

from model import Bearing1DCNN


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "bearing_cnn.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CLASS_NAMES = {
    0: "Normal",
    1: "Inner Race Fault (IR007)",
    2: "Ball Fault",
    3: "Outer Race Fault"
}


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    model = Bearing1DCNN(
        num_classes=4
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"✅ Bearing CNN model loaded successfully on: {DEVICE}"
    )

    return model


# Load model once
model = load_model()


# ============================================================
# PREPROCESS SINGLE SIGNAL
# MUST MATCH preprocess.py
# ============================================================

def preprocess_signal(signal):

    # Convert to float32
    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    # Remove unnecessary dimensions
    signal = signal.squeeze()

    # Check signal length
    if signal.ndim != 1:
        raise ValueError(
            f"Expected 1D signal, but got shape {signal.shape}"
        )

    # ========================================================
    # PER-SAMPLE Z-SCORE NORMALIZATION
    # Same as preprocess.py
    # ========================================================

    mean = np.mean(
        signal,
        keepdims=True
    )

    std = np.std(
        signal,
        keepdims=True
    ) + 1e-8

    signal = (
        signal - mean
    ) / std

    # ========================================================
    # ADD CHANNEL + BATCH DIMENSION
    #
    # Final shape:
    # (1, 1, 1024)
    # ========================================================

    signal = np.expand_dims(
        signal,
        axis=0
    )

    signal = np.expand_dims(
        signal,
        axis=0
    )

    return signal


# ============================================================
# PREDICT A SINGLE VIBRATION SIGNAL
# ============================================================

def predict_fault(signal):

    # Same preprocessing used during training
    processed_signal = preprocess_signal(
        signal
    )

    # Convert to PyTorch Tensor
    signal_tensor = torch.tensor(
        processed_signal,
        dtype=torch.float32
    ).to(DEVICE)

    # Prediction
    with torch.no_grad():

        outputs = model(
            signal_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    class_id = int(
        predicted_class.item()
    )

    confidence_score = float(
        confidence.item() * 100
    )

    predicted_fault = CLASS_NAMES.get(
        class_id,
        "Unknown Fault"
    )

    return {
        "class_id": class_id,
        "fault": predicted_fault,
        "confidence": round(
            confidence_score,
            2
        ),
        "probabilities": probabilities.cpu().numpy()[0]
    }


# ============================================================
# TEST PREDICTION FROM DATASET
# ============================================================

def predict_from_dataset(index=0):

    data_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "X_segments.npy"
    )

    label_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "y_labels.npy"
    )

    X_segments = np.load(
        data_path
    )

    y_labels = np.load(
        label_path
    )

    print(
        f"📊 Dataset shape: {X_segments.shape}"
    )

    # Safety check
    if index < 0 or index >= len(X_segments):
        raise IndexError(
            f"Index must be between 0 and "
            f"{len(X_segments) - 1}"
        )

    signal = X_segments[index]

    # Actual dataset label
    actual_class_id = int(
        y_labels[index]
    )

    actual_fault = CLASS_NAMES.get(
        actual_class_id,
        "Unknown"
    )

    # Model prediction
    result = predict_fault(
        signal
    )

    return signal, actual_class_id, actual_fault, result


# ============================================================
# TEST ALL CLASSES
# ============================================================

def test_all_classes():

    data_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "X_segments.npy"
    )

    label_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "y_labels.npy"
    )

    X_segments = np.load(data_path)
    y_labels = np.load(label_path)

    print("\n" + "=" * 65)
    print("🔍 TESTING ONE SAMPLE FROM EACH BEARING CLASS")
    print("=" * 65)

    for class_id in sorted(CLASS_NAMES.keys()):

        matching_indices = np.where(
            y_labels == class_id
        )[0]

        if len(matching_indices) == 0:

            print(
                f"\n❌ No sample found for "
                f"Class {class_id}"
            )

            continue

        # Take first sample of that class
        index = matching_indices[0]

        signal = X_segments[index]

        result = predict_fault(
            signal
        )

        actual_fault = CLASS_NAMES[class_id]

        print("\n" + "-" * 65)

        print(
            f"Actual Class      : "
            f"{class_id} - {actual_fault}"
        )

        print(
            f"Predicted Class   : "
            f"{result['class_id']} - "
            f"{result['fault']}"
        )

        print(
            f"Confidence        : "
            f"{result['confidence']}%"
        )

        if result["class_id"] == class_id:

            print("Result            : ✅ CORRECT")

        else:

            print("Result            : ❌ INCORRECT")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Test all four classes
    test_all_classes()