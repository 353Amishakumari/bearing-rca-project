import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
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

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "cross_validation_results")

X_PATH = os.path.join(DATA_DIR, "X_segments.npy")
Y_PATH = os.path.join(DATA_DIR, "y_labels.npy")
GROUP_PATH = os.path.join(DATA_DIR, "groups.npy")

NUM_CLASSES = 4
NUM_FOLDS = 5

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

PATIENCE = 7
LR_PATIENCE = 3

RANDOM_SEED = 42

CLASS_NAMES = [
    "Normal",
    "Inner Race",
    "Ball",
    "Outer Race"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X file not found: {X_PATH}")

    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Y file not found: {Y_PATH}")

    if not os.path.exists(GROUP_PATH):
        raise FileNotFoundError(f"Groups file not found: {GROUP_PATH}")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    groups = np.load(GROUP_PATH)

    print(f"X shape      : {X.shape}")
    print(f"y shape      : {y.shape}")
    print(f"groups shape : {groups.shape}")

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must have the same number of samples.")

    print("\nClass distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = np.sum(y == class_id)

        print(
            f"  Class {class_id} - "
            f"{class_name:<12} : {count}"
        )

    print("\nUnique source groups:")

    unique_groups = np.unique(groups)

    for group in unique_groups:

        group_labels = np.unique(y[groups == group])

        print(
            f"  Group {group:<3} -> "
            f"Classes {group_labels.tolist()} | "
            f"Samples {np.sum(groups == group)}"
        )

    return X, y, groups


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(X):

    """
    Same normalization strategy used by your current preprocess.py.

    Each individual vibration segment is normalized independently:
        X_norm = (X - mean) / std
    """

    X = X.astype(np.float32)

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

    # CNN expects:
    # [samples, channels, sequence_length]

    X_norm = np.expand_dims(
        X_norm,
        axis=1
    )

    return X_norm.astype(np.float32)


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_loader(X, y, shuffle=False):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

    return loader


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * X_batch.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    loader,
    device
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold_number,
    X_train,
    y_train,
    X_test,
    y_test,
    device
):

    print("\n" + "-" * 70)
    print(f"FOLD {fold_number}")
    print("-" * 70)

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Normalize train and test
    # --------------------------------------------------------

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True
    )

    test_loader = create_loader(
        X_test,
        y_test,
        shuffle=False
    )

    # --------------------------------------------------------
    # Fresh model for this fold
    # --------------------------------------------------------

    model = Bearing1DCNN(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=LR_PATIENCE
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_accuracy = 0.0
    best_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        y_true_epoch, y_pred_epoch = evaluate_model(
            model,
            test_loader,
            device
        )

        test_accuracy = accuracy_score(
            y_true_epoch,
            y_pred_epoch
        )

        scheduler.step(test_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Test Acc: {test_accuracy * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save best model state
        # ----------------------------------------------------

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    y_true, y_pred = evaluate_model(
        model,
        test_loader,
        device
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    print("\nFold Results:")
    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )
    print(
        f"Precision : {precision * 100:.2f}%"
    )
    print(
        f"Recall    : {recall * 100:.2f}%"
    )
    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "model": model
    }


# ============================================================
# MAIN CROSS-VALIDATION
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("5-FOLD GROUP CROSS-VALIDATION")
    print("Industrial Multimodal Bearing Fault Detection")
    print("=" * 70)

    print(
        "\nImportant:"
    )

    print(
        "Segments from the same source group are kept together."
    )

    print(
        "Therefore, there is no group leakage between train and test."
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
    # Load dataset
    # --------------------------------------------------------

    X, y, groups = load_data()

    # --------------------------------------------------------
    # Check number of unique groups
    # --------------------------------------------------------

    unique_groups = np.unique(groups)

    print(
        f"\nTotal unique groups: {len(unique_groups)}"
    )

    if len(unique_groups) < NUM_FOLDS:

        raise ValueError(
            f"Need at least {NUM_FOLDS} unique groups "
            f"for {NUM_FOLDS}-fold GroupKFold."
        )

    # --------------------------------------------------------
    # Group K-Fold
    # --------------------------------------------------------

    group_kfold = GroupKFold(
        n_splits=NUM_FOLDS
    )

    fold_results = []

    all_true = []
    all_pred = []

    # ========================================================
    # FOLD LOOP
    # ========================================================

    for fold_number, (train_idx, test_idx) in enumerate(
        group_kfold.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        train_groups = np.unique(
            groups[train_idx]
        )

        test_groups = np.unique(
            groups[test_idx]
        )

        print("\n" + "=" * 70)

        print(
            f"FOLD {fold_number}/{NUM_FOLDS}"
        )

        print(
            f"Train Groups: {train_groups.tolist()}"
        )

        print(
            f"Test Groups : {test_groups.tolist()}"
        )

        # ----------------------------------------------------
        # Leakage check
        # ----------------------------------------------------

        overlap = set(train_groups).intersection(
            set(test_groups)
        )

        if len(overlap) > 0:

            raise RuntimeError(
                f"DATA LEAKAGE DETECTED! "
                f"Overlapping groups: {overlap}"
            )

        print(
            "Group leakage check: PASSED"
        )

        # ----------------------------------------------------
        # Prepare fold data
        # ----------------------------------------------------

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        # ----------------------------------------------------
        # Train fold
        # ----------------------------------------------------

        result = train_fold(
            fold_number,
            X_train,
            y_train,
            X_test,
            y_test,
            device
        )

        fold_results.append(result)

        all_true.extend(
            result["y_true"]
        )

        all_pred.extend(
            result["y_pred"]
        )

        # ----------------------------------------------------
        # Save fold model
        # ----------------------------------------------------

        fold_model_path = os.path.join(
            RESULT_DIR,
            f"fold_{fold_number}_model.pth"
        )

        torch.save(
            result["model"].state_dict(),
            fold_model_path
        )

        print(
            f"\nFold model saved:"
        )

        print(
            fold_model_path
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    accuracies = np.array([
        result["accuracy"]
        for result in fold_results
    ])

    precisions = np.array([
        result["precision"]
        for result in fold_results
    ])

    recalls = np.array([
        result["recall"]
        for result in fold_results
    ])

    f1_scores = np.array([
        result["f1"]
        for result in fold_results
    ])

    # --------------------------------------------------------
    # Mean and Standard Deviation
    # --------------------------------------------------------

    mean_accuracy = np.mean(
        accuracies
    )

    std_accuracy = np.std(
        accuracies
    )

    mean_precision = np.mean(
        precisions
    )

    std_precision = np.std(
        precisions
    )

    mean_recall = np.mean(
        recalls
    )

    std_recall = np.std(
        recalls
    )

    mean_f1 = np.mean(
        f1_scores
    )

    std_f1 = np.std(
        f1_scores
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("5-FOLD GROUP CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print("\nFold-wise Results:")

    for i in range(NUM_FOLDS):

        print(
            f"\nFold {i + 1}:"
        )

        print(
            f"  Accuracy  : {accuracies[i] * 100:.2f}%"
        )

        print(
            f"  Precision : {precisions[i] * 100:.2f}%"
        )

        print(
            f"  Recall    : {recalls[i] * 100:.2f}%"
        )

        print(
            f"  F1 Score  : {f1_scores[i] * 100:.2f}%"
        )

    print("\n" + "-" * 70)

    print(
        f"Mean Accuracy  : "
        f"{mean_accuracy * 100:.2f}% "
        f"+/- {std_accuracy * 100:.2f}%"
    )

    print(
        f"Mean Precision : "
        f"{mean_precision * 100:.2f}% "
        f"+/- {std_precision * 100:.2f}%"
    )

    print(
        f"Mean Recall    : "
        f"{mean_recall * 100:.2f}% "
        f"+/- {std_recall * 100:.2f}%"
    )

    print(
        f"Mean F1 Score  : "
        f"{mean_f1 * 100:.2f}% "
        f"+/- {std_f1 * 100:.2f}%"
    )

    print("-" * 70)

    # ========================================================
    # OVERALL CONFUSION MATRIX
    # ========================================================

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_cm = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(NUM_CLASSES))
    )

    print("\nOverall Confusion Matrix:")
    print(overall_cm)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\nOverall Classification Report:")
    print(report)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary_path = os.path.join(
        RESULT_DIR,
        "cross_validation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "5-FOLD GROUP CROSS-VALIDATION RESULTS\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Number of folds: {NUM_FOLDS}\n"
        )

        f.write(
            f"Number of groups: {len(unique_groups)}\n"
        )

        f.write(
            f"Device: {device}\n\n"
        )

        for i in range(NUM_FOLDS):

            f.write(
                f"Fold {i + 1}\n"
            )

            f.write(
                f"Accuracy: "
                f"{accuracies[i] * 100:.2f}%\n"
            )

            f.write(
                f"Precision: "
                f"{precisions[i] * 100:.2f}%\n"
            )

            f.write(
                f"Recall: "
                f"{recalls[i] * 100:.2f}%\n"
            )

            f.write(
                f"F1: "
                f"{f1_scores[i] * 100:.2f}%\n\n"
            )

        f.write(
            "MEAN +/- STANDARD DEVIATION\n"
        )

        f.write(
            f"Accuracy: "
            f"{mean_accuracy * 100:.2f}% "
            f"+/- {std_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Precision: "
            f"{mean_precision * 100:.2f}% "
            f"+/- {std_precision * 100:.2f}%\n"
        )

        f.write(
            f"Recall: "
            f"{mean_recall * 100:.2f}% "
            f"+/- {std_recall * 100:.2f}%\n"
        )

        f.write(
            f"F1: "
            f"{mean_f1 * 100:.2f}% "
            f"+/- {std_f1 * 100:.2f}%\n\n"
        )

        f.write(
            "OVERALL CONFUSION MATRIX\n"
        )

        f.write(
            str(overall_cm)
        )

        f.write(
            "\n\nCLASSIFICATION REPORT\n"
        )

        f.write(
            report
        )

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.save(
        os.path.join(
            RESULT_DIR,
            "cv_y_true.npy"
        ),
        all_true
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "cv_y_pred.npy"
        ),
        all_pred
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "cv_confusion_matrix.npy"
        ),
        overall_cm
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "fold_accuracies.npy"
        ),
        accuracies
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULT_DIR
    )

    print(
        f"\nSummary file:"
    )

    print(
        summary_path
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()