import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROCESSED_DIR = os.path.join("dataset", "processed")


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

def make_group_aware_split(y, groups):
    """
    Creates completely disjoint Train / Validation / Test splits.

    Rule:
    - Segments from the same original source file (group)
      can appear in ONLY ONE split.
    - Every class must have at least 3 unique groups:
        1 group -> Test
        1 group -> Validation
        Remaining groups -> Train

    This prevents source-file data leakage.
    """

    unique_groups = np.unique(groups)

    # Find the label/class associated with every group
    group_label = {}

    for group in unique_groups:
        labels_in_group = np.unique(y[groups == group])

        # Safety check: one source group should belong to one class
        if len(labels_in_group) != 1:
            raise ValueError(
                f"❌ Group '{group}' contains multiple classes: "
                f"{labels_in_group.tolist()}"
            )

        group_label[group] = labels_in_group[0]

    # Organize groups class-wise
    label_to_groups = defaultdict(list)

    for group, label in group_label.items():
        label_to_groups[int(label)].append(group)

    train_groups = []
    val_groups = []
    test_groups = []

    print("\n" + "=" * 60)
    print("📊 GROUP DISTRIBUTION")
    print("=" * 60)

    # Create separate groups for each class
    for label in sorted(label_to_groups.keys()):

        class_groups = sorted(label_to_groups[label])
        n_groups = len(class_groups)

        print(
            f"Class {label}: {n_groups} unique source groups "
            f"-> {class_groups}"
        )

        # At least 3 groups are needed for proper train/val/test split
        if n_groups < 3:
            raise ValueError(
                f"\n❌ Class {label} has only {n_groups} unique source group(s).\n"
                f"At least 3 groups are required for a completely "
                f"separate Train / Validation / Test split.\n"
                f"Current groups: {class_groups}"
            )

        # Last group -> Test
        # Second-last group -> Validation
        # Remaining groups -> Training
        test_group = class_groups[-1]
        val_group = class_groups[-2]
        train_class_groups = class_groups[:-2]

        test_groups.append(test_group)
        val_groups.append(val_group)
        train_groups.extend(train_class_groups)

    # Convert to sets
    train_groups = set(train_groups)
    val_groups = set(val_groups)
    test_groups = set(test_groups)

    # ========================================================
    # SAFETY CHECK: NO OVERLAP BETWEEN SPLITS
    # ========================================================

    assert train_groups.isdisjoint(val_groups), \
        "❌ Data leakage: Train and Validation groups overlap!"

    assert train_groups.isdisjoint(test_groups), \
        "❌ Data leakage: Train and Test groups overlap!"

    assert val_groups.isdisjoint(test_groups), \
        "❌ Data leakage: Validation and Test groups overlap!"

    print("\n" + "=" * 60)
    print("✅ GROUP-AWARE SPLIT CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Train groups ({len(train_groups)}): {sorted(train_groups)}")
    print(f"Val groups   ({len(val_groups)}): {sorted(val_groups)}")
    print(f"Test groups  ({len(test_groups)}): {sorted(test_groups)}")

    return train_groups, val_groups, test_groups


# ============================================================
# DATA LOADER
# ============================================================

def get_dataloaders(batch_size=32):
    """
    Loads processed vibration segments and creates:
    Train Loader
    Validation Loader
    Test Loader
    """

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    x_path = os.path.join(PROCESSED_DIR, "X_segments.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_labels.npy")
    groups_path = os.path.join(PROCESSED_DIR, "groups.npy")

    # Check files
    required_files = [x_path, y_path, groups_path]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"\n❌ Required file not found:\n{file_path}\n"
            )

    X = np.load(x_path)
    y = np.load(y_path)
    groups = np.load(groups_path)

    print("\n" + "=" * 60)
    print("📂 DATASET LOADED")
    print("=" * 60)

    print(f"X shape      : {X.shape}")
    print(f"y shape      : {y.shape}")
    print(f"groups shape : {groups.shape}")

    # Safety checks
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(
            "❌ X, y and groups must contain the same number of samples!"
        )

    # --------------------------------------------------------
    # 2. CHECK CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\n📊 CLASS DISTRIBUTION:")

    for label in sorted(np.unique(y)):
        count = np.sum(y == label)
        print(f"Class {label}: {count} segments")

    # --------------------------------------------------------
    # 3. PER-SAMPLE Z-SCORE NORMALIZATION
    # --------------------------------------------------------

    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)

    # Avoid division by zero
    std = std + 1e-8

    X_norm = (X - mean) / std

    # --------------------------------------------------------
    # 4. ADD CHANNEL DIMENSION FOR 1D CNN
    # --------------------------------------------------------

    # Before: (samples, 1024)
    # After : (samples, 1, 1024)

    X_norm = np.expand_dims(X_norm, axis=1)

    print(f"\nNormalized X shape: {X_norm.shape}")

    # --------------------------------------------------------
    # 5. CREATE GROUP-AWARE SPLIT
    # --------------------------------------------------------

    train_groups, val_groups, test_groups = make_group_aware_split(
        y,
        groups
    )

    # Create masks based on source-file groups
    train_mask = np.isin(groups, list(train_groups))
    val_mask = np.isin(groups, list(val_groups))
    test_mask = np.isin(groups, list(test_groups))

    # --------------------------------------------------------
    # 6. CREATE TRAIN / VAL / TEST DATA
    # --------------------------------------------------------

    X_train = X_norm[train_mask]
    y_train = y[train_mask]

    X_val = X_norm[val_mask]
    y_val = y[val_mask]

    X_test = X_norm[test_mask]
    y_test = y[test_mask]

    # --------------------------------------------------------
    # 7. FINAL SAFETY CHECK
    # --------------------------------------------------------

    if len(X_train) == 0:
        raise ValueError("❌ Training set is empty!")

    if len(X_val) == 0:
        raise ValueError("❌ Validation set is empty!")

    if len(X_test) == 0:
        raise ValueError("❌ Test set is empty!")

    print("\n" + "=" * 60)
    print("📦 FINAL DATA SPLIT")
    print("=" * 60)

    print(f"Train: {len(X_train)} segments")
    print(f"Val  : {len(X_val)} segments")
    print(f"Test : {len(X_test)} segments")

    print("\nClasses in each split:")

    print(
        "Train:",
        sorted(np.unique(y_train).tolist())
    )

    print(
        "Val  :",
        sorted(np.unique(y_val).tolist())
    )

    print(
        "Test :",
        sorted(np.unique(y_test).tolist())
    )

    # Ensure all four classes are present
    all_classes = set(np.unique(y))

    if set(np.unique(y_train)) != all_classes:
        raise ValueError("❌ Some classes are missing from Training set!")

    if set(np.unique(y_val)) != all_classes:
        raise ValueError("❌ Some classes are missing from Validation set!")

    if set(np.unique(y_test)) != all_classes:
        raise ValueError("❌ Some classes are missing from Test set!")

    # --------------------------------------------------------
    # 8. CONVERT TO PYTORCH TENSORS
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    # --------------------------------------------------------
    # 9. CREATE DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("\n✅ DataLoaders created successfully!")
    print("=" * 60)

    return train_loader, val_loader, test_loader


# ============================================================
# TEST PREPROCESSING DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("🚀 Testing preprocessing pipeline...")

    train_loader, val_loader, test_loader = get_dataloaders()

    print("\n🔹 Train batches:", len(train_loader))
    print("🔹 Val batches  :", len(val_loader))
    print("🔹 Test batches :", len(test_loader))

    # Check one batch
    X_batch, y_batch = next(iter(train_loader))

    print("\n📌 Sample batch:")
    print("X batch shape:", X_batch.shape)
    print("y batch shape:", y_batch.shape)
    print("Labels:", y_batch[:10].tolist())