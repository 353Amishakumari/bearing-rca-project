# -*- coding: utf-8 -*-
"""
train_thermal.py
------------------
Trains RadiometricThermalCNN (from thermal_image_module.py) on the
Squirrel-Cage Induction Motor thermal dataset (via thermal_dataset.py).

SPLIT STRATEGY — 3-way, experiment-folder-level:
Consecutive frames inside one experiment folder (e.g. current-load-6A) are
captured milliseconds apart and are visually near-identical. A random
image-level split would leak near-duplicate frames across sets. To avoid
this, whole experiment folders are assigned entirely to one of three sets,
stratified per broad class (normal / misalignment / rotor_fault):

    - TRAIN : the model learns from these folders.
    - VAL   : used only during training to pick the best checkpoint
              (early-stopping style). Never used to report a final number.
    - TEST  : held out completely — never seen during training or
              checkpoint selection. Only eval_thermal.py touches this set,
              and only after training is finished. This is the number
              that should be reported/quoted.

Run from the project root:
    python src/train_thermal.py
"""

import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from thermal_dataset import ThermalMotorDataset, IDX_TO_NAME, get_label_from_folder
from thermal_image_module import RadiometricThermalCNN, DEVICE

# ---------------- Config ----------------
_here = os.path.dirname(os.path.abspath(__file__))          # .../bearing-rca-project/src
_project_root = os.path.dirname(_here)                       # .../bearing-rca-project

BASE = os.path.join(_project_root, "dataset", "thermal_squirrel_cage", "thermal_images")
COUPLING_DIRS = [os.path.join(BASE, "coupling_1"), os.path.join(BASE, "coupling_2")]
RESOLUTION_FOLDER = "workswell_wic_640"
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3

# Fractions of experiment FOLDERS per class assigned to val / test.
# Remainder goes to train.
VAL_FRACTION_OF_FOLDERS = 0.15
TEST_FRACTION_OF_FOLDERS = 0.20
SEED = 42

SAVE_PATH = os.path.join(_project_root, "saved_models", "radiometric_thermal_cnn.pt")
SPLIT_FILE_PATH = os.path.join(_project_root, "saved_models", "thermal_split_folders.txt")
# -----------------------------------------


def build_folder_level_split(root_dirs, resolution_folder, val_fraction, test_fraction, seed):
    """
    Returns (train_folders, val_folders, test_folders) — sets of full paths
    to experiment folders (e.g. .../coupling_1/current-load-6A). Stratified
    per broad label so every split has a mix of classes.
    """
    folders_by_label = defaultdict(list)

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue
        for exp_folder in sorted(os.listdir(root_dir)):
            exp_path = os.path.join(root_dir, exp_folder)
            img_dir = os.path.join(exp_path, resolution_folder)
            if not os.path.isdir(img_dir):
                continue
            broad_label, _, _ = get_label_from_folder(exp_folder)
            folders_by_label[broad_label].append(exp_path)

    rng = random.Random(seed)
    train_folders, val_folders, test_folders = set(), set(), set()

    for label, folders in folders_by_label.items():
        folders = sorted(folders)
        rng.shuffle(folders)

        n = len(folders)
        n_test = max(1, round(n * test_fraction))
        n_val = max(1, round(n * val_fraction))
        # guard against over-allocating on classes with very few folders
        n_test = min(n_test, n - 2) if n > 2 else 1
        n_val = min(n_val, max(0, n - n_test - 1)) if n - n_test > 1 else 0

        test_pick = folders[:n_test]
        val_pick = folders[n_test:n_test + n_val]
        train_pick = folders[n_test + n_val:]

        test_folders.update(test_pick)
        val_folders.update(val_pick)
        train_folders.update(train_pick)

        print(f"  class '{label}': {len(train_pick)} -> train, "
              f"{len(val_pick)} -> val, {len(test_pick)} -> test")

    return train_folders, val_folders, test_folders


def split_dataset_indices(dataset, train_folders, val_folders, test_folders=None):
    """Assigns each sample index to a set based on which experiment folder
    its image file lives under. If test_folders is None, only train/val
    are computed (used by eval_thermal.py convenience calls)."""
    train_idx, val_idx, test_idx = [], [], []

    for i, (img_path, _broad, _sev, _load) in enumerate(dataset.samples):
        # img_path looks like: .../coupling_X/<experiment_folder>/workswell_wic_640/<file>.png
        exp_folder = os.path.dirname(os.path.dirname(img_path))
        if exp_folder in train_folders:
            train_idx.append(i)
        elif exp_folder in val_folders:
            val_idx.append(i)
        elif test_folders is not None and exp_folder in test_folders:
            test_idx.append(i)

    if test_folders is not None:
        return train_idx, val_idx, test_idx
    return train_idx, val_idx


def save_split_record(train_folders, val_folders, test_folders):
    os.makedirs(os.path.dirname(SPLIT_FILE_PATH), exist_ok=True)
    with open(SPLIT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("THERMAL DATASET — EXPERIMENT-FOLDER SPLIT RECORD\n")
        f.write("=" * 50 + "\n")
        f.write("TRAIN folders:\n")
        for p in sorted(train_folders):
            f.write(f"  {p}\n")
        f.write("\nVAL folders (checkpoint selection only):\n")
        for p in sorted(val_folders):
            f.write(f"  {p}\n")
        f.write("\nTEST folders (held out, final evaluation only):\n")
        for p in sorted(test_folders):
            f.write(f"  {p}\n")


def main():
    print("Building experiment-folder-level 3-way split (train/val/test, no leakage)...")
    train_folders, val_folders, test_folders = build_folder_level_split(
        COUPLING_DIRS, RESOLUTION_FOLDER, VAL_FRACTION_OF_FOLDERS, TEST_FRACTION_OF_FOLDERS, SEED
    )
    save_split_record(train_folders, val_folders, test_folders)
    print(f"Split record saved to: {SPLIT_FILE_PATH}")

    dataset = ThermalMotorDataset(root_dirs=COUPLING_DIRS, resolution_folder=RESOLUTION_FOLDER)

    train_idx, val_idx, test_idx = split_dataset_indices(
        dataset, train_folders, val_folders, test_folders
    )
    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)} | "
          f"Test samples (untouched during training): {len(test_idx)}")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = RadiometricThermalCNN(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # ---- validation (checkpoint selection only, not a reportable number) ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(images)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> saved new best model (val_acc={val_acc:.3f}) to {SAVE_PATH}")

    print(f"\nTraining done. Best VAL acc (checkpoint selection only): {best_val_acc:.3f}")
    print(f"Weights at: {SAVE_PATH}")
    print(f"Classes: {IDX_TO_NAME}")
    print(f"\nIMPORTANT: run 'python src/eval_thermal.py' next to get the genuine "
          f"held-out TEST accuracy — that is the number to report, not val_acc above.")


if __name__ == "__main__":
    main()