import os
import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import get_dataloaders
from model import Bearing1DCNN


# ============================================================
# PATH CONFIGURATION
# ============================================================

MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_SAVE_DIR,
    "bearing_cnn.pth"
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(epochs=30, lr=0.001, patience=7):

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = get_dataloaders()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 60)
    print("🚀 CNN TRAINING")
    print("=" * 60)

    print(f"Device : {device}")
    print(f"Epochs : {epochs}")
    print(f"Learning Rate : {lr}")
    print(f"Early Stopping Patience : {patience}")

    # --------------------------------------------------------
    # 2. CREATE MODEL
    # --------------------------------------------------------

    model = Bearing1DCNN(
        num_classes=4
    ).to(device)

    # --------------------------------------------------------
    # 3. LOSS FUNCTION
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # 4. OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # 5. LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # 6. BEST MODEL TRACKING
    # --------------------------------------------------------

    best_val_acc = 0.0
    patience_counter = 0

    # --------------------------------------------------------
    # 7. TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += (
                loss.item() * inputs.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        # ----------------------------------------------------
        # TRAINING METRICS
        # ----------------------------------------------------

        train_loss = running_loss / total
        train_acc = (
            correct / total
        ) * 100

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                _, predictions = torch.max(
                    outputs,
                    1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_acc = (
            val_correct / val_total
        ) * 100

        # ----------------------------------------------------
        # LEARNING RATE UPDATE
        # ----------------------------------------------------

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"   ✅ Best model saved "
                f"(Val Acc: {val_acc:.2f}%)"
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= patience:

            print(
                "\n⏹️ Early stopping triggered."
            )

            break

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best Validation Accuracy : "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Best model saved to : "
        f"{MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()