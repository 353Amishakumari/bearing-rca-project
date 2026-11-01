# -*- coding: utf-8 -*-
"""
eval_thermal.py
------------------
Evaluates the trained RadiometricThermalCNN on the genuine held-out TEST
set — experiment folders that were never used during training or for
checkpoint (best-model) selection. This is the number that should be
reported/quoted (not the val_acc printed during training).

Produces:
    saved_models/thermal_eval_summary.txt   (accuracy, macro F1, per-class
                                              precision/recall/F1, support)
    saved_models/thermal_confusion_matrix.png

Run from the project root, AFTER train_thermal.py has finished:
    python src/eval_thermal.py
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
)

from thermal_dataset import ThermalMotorDataset, IDX_TO_NAME
from thermal_image_module import RadiometricThermalCNN, DEVICE
from train_thermal import (
    COUPLING_DIRS,
    RESOLUTION_FOLDER,
    VAL_FRACTION_OF_FOLDERS,
    TEST_FRACTION_OF_FOLDERS,
    SEED,
    build_folder_level_split,
    split_dataset_indices,
)

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)

MODEL_PATH = os.path.join(_project_root, "saved_models", "radiometric_thermal_cnn.pt")
SUMMARY_PATH = os.path.join(_project_root, "saved_models", "thermal_eval_summary.txt")
CM_PATH = os.path.join(_project_root, "saved_models", "thermal_confusion_matrix.png")
BATCH_SIZE = 32

CLASS_NAMES = [IDX_TO_NAME[i] for i in range(len(IDX_TO_NAME))]


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained weights not found at {MODEL_PATH}. Run train_thermal.py first."
        )

    print("Rebuilding the same 3-way experiment-folder-level split used for training...")
    train_folders, val_folders, test_folders = build_folder_level_split(
        COUPLING_DIRS, RESOLUTION_FOLDER, VAL_FRACTION_OF_FOLDERS, TEST_FRACTION_OF_FOLDERS, SEED
    )

    dataset = ThermalMotorDataset(root_dirs=COUPLING_DIRS, resolution_folder=RESOLUTION_FOLDER)
    _train_idx, _val_idx, test_idx = split_dataset_indices(
        dataset, train_folders, val_folders, test_folders
    )
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Evaluating on {len(test_ds)} held-out TEST samples "
          f"(from {len(test_folders)} experiment folders NEVER used in training "
          f"or checkpoint selection)...")

    model = RadiometricThermalCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(DEVICE)
            logits, _ = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    report_str = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0
    )

    # ---------------- Write summary file ----------------
    lines = []
    lines.append("THERMAL CNN — HELD-OUT TEST EVALUATION SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Test samples       : {len(test_ds)}")
    lines.append(f"Held-out folders   : {len(test_folders)} "
                 f"(never used in training or checkpoint selection)")
    lines.append("")
    lines.append(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    lines.append(f"Macro F1-Score: {macro_f1 * 100:.2f}%")
    lines.append("")
    lines.append(f"{'Class':<22}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}{'Support':<8}")
    for i, name in enumerate(CLASS_NAMES):
        lines.append(
            f"{name:<22}{precision[i]:<12.3f}{recall[i]:<10.3f}{f1[i]:<10.3f}{support[i]:<8d}"
        )
    lines.append("")
    lines.append("Full classification report (sklearn):")
    lines.append(report_str)
    lines.append("")
    lines.append("Split methodology: 3-way experiment-folder-level split, stratified per "
                 "class. TRAIN folders were used to fit the model; VAL folders were used "
                 "only during training to select the best checkpoint; TEST folders were "
                 "never touched until this evaluation, so this accuracy is a genuine "
                 "held-out estimate. See saved_models/thermal_split_folders.txt for the "
                 "exact folder assignment.")

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved summary to: {SUMMARY_PATH}")

    # ---------------- Confusion matrix plot ----------------
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Thermal CNN — Held-Out TEST Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to: {CM_PATH}")

    print(f"\nHeld-out TEST Accuracy: {overall_acc * 100:.2f}% | Macro F1: {macro_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()