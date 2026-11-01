"""
Model Evaluation Script
------------------------
Trained 1D-CNN (bearing_cnn.pth) ko TEST set pe evaluate karta hai.
Output: Accuracy, Precision, Recall, F1-score (per class + overall),
        aur Confusion Matrix (terminal + saved image).

Run: python src/evaluate.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Taaki 'src' folder ke andar se bhi aur bahar se bhi chal sake
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Bearing1DCNN
from preprocess import get_dataloaders

MODEL_PATH = os.path.join("saved_models", "bearing_cnn.pth")
OUTPUT_DIR = "saved_models"  # confusion matrix image yahan save hogi

CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load test data (same split as training, random_state=42 fixed) ----
    _, _, test_loader = get_dataloaders(batch_size=32)

    # ---- Load trained model ----
    model = Bearing1DCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    macro_f1 = f1.mean()

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (Test Set)")
    print("=" * 60)
    print(f"Overall Accuracy : {acc*100:.2f}%")
    print(f"Macro F1-Score   : {macro_f1*100:.2f}%")
    print("-" * 60)
    print(classification_report(
        all_labels, all_preds, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0
    ))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASS_NAMES)))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    # Plot and save confusion matrix as image
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix - Bearing Fault Classification")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Confusion matrix image saved to: {save_path}")

    # Also save metrics as a small text summary for the dashboard/report
    summary_path = os.path.join(OUTPUT_DIR, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Overall Accuracy: {acc*100:.2f}%\n")
        f.write(f"Macro F1-Score: {macro_f1*100:.2f}%\n\n")
        f.write(classification_report(all_labels, all_preds, labels=range(len(CLASS_NAMES)),
                                       target_names=CLASS_NAMES, zero_division=0))
    print(f"✅ Summary text saved to: {summary_path}")

    # ---- NEW: Save raw y_true / y_pred arrays for the Streamlit dashboard ----
    # Dashboard ka "Select Fault Type" dropdown inhi arrays se per-class
    # (selected_class ke hisaab se) precision/recall/f1/confusion-row
    # dynamically compute kar sakega, static eval_summary.txt/confusion_matrix.png
    # ki tarah hardcoded/overall hi nahi rahega.
    y_true_path = os.path.join(OUTPUT_DIR, "y_true.npy")
    y_pred_path = os.path.join(OUTPUT_DIR, "y_pred.npy")
    np.save(y_true_path, all_labels)
    np.save(y_pred_path, all_preds)
    print(f"✅ y_true saved to: {y_true_path}")
    print(f"✅ y_pred saved to: {y_pred_path}")

    return acc, macro_f1, cm


if __name__ == "__main__":
    evaluate_model()