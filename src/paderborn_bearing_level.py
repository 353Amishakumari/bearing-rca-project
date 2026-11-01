import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ============================================================
# PATH
# ============================================================

RESULTS_DIR = "paderborn_results"

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "paderborn_predictions.csv"
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_NAMES = {
    0: "Normal",
    1: "Inner Race",
    3: "Outer Race"
}


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("PADERBORN FILE / BEARING LEVEL EVALUATION")
print("=" * 70)

if not os.path.exists(PREDICTION_FILE):
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTION_FILE}"
    )

df = pd.read_csv(PREDICTION_FILE)

print("\nPrediction file:")
print(PREDICTION_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nTotal window predictions:", len(df))


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


true_col = find_column([
    "true_label",
    "y_true",
    "true",
    "label",
    "actual_label"
])

pred_col = find_column([
    "predicted_label",
    "y_pred",
    "prediction",
    "pred"
])

file_col = find_column([
    "file",
    "file_name",
    "filename",
    "source_file",
    "mat_file"
])


print("\nDetected columns:")
print("True label column :", true_col)
print("Prediction column :", pred_col)
print("File column       :", file_col)


if true_col is None:
    raise ValueError(
        "Could not identify true-label column."
    )

if pred_col is None:
    raise ValueError(
        "Could not identify prediction column."
    )

if file_col is None:
    raise ValueError(
        "Could not identify file column."
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    [file_col, true_col, pred_col]
].copy()

df.columns = [
    "file",
    "true_label",
    "predicted_label"
]

df["true_label"] = df["true_label"].astype(int)
df["predicted_label"] = df["predicted_label"].astype(int)


# ============================================================
# MAJORITY VOTE FUNCTION
# ============================================================

def majority_vote(values):

    values = list(values)

    counts = {}

    for value in values:

        counts[value] = counts.get(
            value,
            0
        ) + 1

    # Highest count wins
    # In a tie, smallest label wins deterministically
    winner = sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    )[0][0]

    return winner


# ============================================================
# FILE-LEVEL AGGREGATION
# ============================================================

file_results = []

grouped = df.groupby("file", sort=True)

print("\n")
print("=" * 70)
print("FILE-LEVEL MAJORITY VOTING")
print("=" * 70)

for file_name, group in grouped:

    true_labels = group["true_label"].values
    predictions = group["predicted_label"].values

    # True label should be same for all windows
    true_label = majority_vote(true_labels)

    predicted_label = majority_vote(predictions)

    # Prediction distribution
    prediction_counts = group[
        "predicted_label"
    ].value_counts()

    file_result = {
        "file": file_name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "num_windows": len(group),
        "correct": int(
            true_label == predicted_label
        )
    }

    # Add class-wise vote counts
    for label in [0, 1, 3]:

        file_result[
            f"votes_{LABEL_NAMES[label].replace(' ', '_')}"
        ] = int(
            prediction_counts.get(
                label,
                0
            )
        )

    file_results.append(file_result)


file_df = pd.DataFrame(file_results)


# ============================================================
# SAVE FILE-LEVEL RESULTS
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "paderborn_file_level_results.csv"
)

file_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# METRICS
# ============================================================

y_true = file_df["true_label"].values

y_pred = file_df["predicted_label"].values

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=[0, 1, 3],
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 3]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FILE-LEVEL RESULTS")
print("=" * 70)

print(
    f"\nTotal files/bearings evaluated: {len(file_df)}"
)

print(
    f"Accuracy                    : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision             : {precision * 100:.2f}%"
)

print(
    f"Macro Recall                : {recall * 100:.2f}%"
)

print(
    f"Macro F1                    : {f1 * 100:.2f}%"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

precision_c, recall_c, f1_c, support_c = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 3],
        average=None,
        zero_division=0
    )
)


print("\n")
print("=" * 70)
print("PER-CLASS FILE-LEVEL RESULTS")
print("=" * 70)

for i, label in enumerate([0, 1, 3]):

    print(f"\n{LABEL_NAMES[label]}")

    print(
        f"  Precision : {precision_c[i] * 100:.2f}%"
    )

    print(
        f"  Recall    : {recall_c[i] * 100:.2f}%"
    )

    print(
        f"  F1 Score  : {f1_c[i] * 100:.2f}%"
    )

    print(
        f"  Support   : {support_c[i]}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("FILE-LEVEL CONFUSION MATRIX")
print("=" * 70)

print("\nRows = True")
print("Columns = Predicted")
print("\nClasses: [Normal, Inner Race, Outer Race]\n")

print(cm)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("FILE-LEVEL CLASS DISTRIBUTION")
print("=" * 70)

print("\nTrue class distribution:")

for label in [0, 1, 3]:

    count = np.sum(
        y_true == label
    )

    print(
        f"  {label} - {LABEL_NAMES[label]:12} : {count}"
    )


print("\nPredicted class distribution:")

for label in [0, 1, 3]:

    count = np.sum(
        y_pred == label
    )

    print(
        f"  {label} - {LABEL_NAMES[label]:12} : {count}"
    )


# ============================================================
# SHOW EACH FILE RESULT
# ============================================================

print("\n")
print("=" * 70)
print("INDIVIDUAL FILE RESULTS")
print("=" * 70)

for _, row in file_df.iterrows():

    true_name = LABEL_NAMES.get(
        int(row["true_label"]),
        str(row["true_label"])
    )

    pred_name = LABEL_NAMES.get(
        int(row["predicted_label"]),
        str(row["predicted_label"])
    )

    status = "CORRECT" if row["correct"] == 1 else "WRONG"

    print(
        f"{row['file']:<40} "
        f"True={true_name:<12} "
        f"Pred={pred_name:<12} "
        f"Windows={row['num_windows']:<3} "
        f"{status}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Number of Files",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "Value": [
        len(file_df),
        accuracy,
        precision,
        recall,
        f1
    ]
})

summary_file = os.path.join(
    RESULTS_DIR,
    "paderborn_file_level_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_file = os.path.join(
    RESULTS_DIR,
    "file_level_confusion_matrix.npy"
)

np.save(
    cm_file,
    cm
)


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("FILE / BEARING LEVEL EVALUATION COMPLETED")
print("=" * 70)

print("\nResults saved:")

print(
    "\nFile-level results:"
)

print(
    os.path.abspath(output_file)
)

print(
    "\nSummary:"
)

print(
    os.path.abspath(summary_file)
)

print(
    "\nConfusion matrix:"
)

print(
    os.path.abspath(cm_file)
)

print("\n")