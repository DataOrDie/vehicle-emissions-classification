from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# -----------------------------------------------------------------------------
# W&B authentication
# -----------------------------------------------------------------------------
print("[SECTION] W&B authentication")
wandb.login()


# -----------------------------------------------------------------------------
# Project paths and imports
# -----------------------------------------------------------------------------
# Resolve paths from this file so running from any cwd still works.
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.one_step import OneStepOptions, preprocess_one_step


# -----------------------------------------------------------------------------
# Dataset load
# -----------------------------------------------------------------------------
print("[SECTION] Loading dataset")
if "df" not in globals():
    df = pd.read_csv(project_root / "data" / "train.csv")


# -----------------------------------------------------------------------------
# Preprocessing options
# -----------------------------------------------------------------------------
print("[SECTION] Configuring preprocessing options")
noemp_option: str = "log"
newexist_option: str = "B"
createjob_option: str = "A"
retainedjob_option: str = "A"
approvaldate_option: str = "A"
approvalfy_option: str = "A"
franchise_option: str = "binary"
urbanrural_option: str = "onehot"
revlinecr_option: str = "C"
lowdoc_option: str = "C"
disbursementgross_option: str = "A"
local_state: str = "IL"

# Accepted values:
# noemp_option: "raw" | "log" | "binning"
# newexist_option: "A" | "B"
# createjob_option: "A" | "B"
# retainedjob_option: "A" | "B"
# approvaldate_option: "A" | "B"
# approvalfy_option: "A" | "B"
# franchise_option: "binary" | "raw"
# urbanrural_option: "onehot" | "text"
# revlinecr_option: "A" | "B" | "C"
# lowdoc_option: "A" | "B" | "C"
# disbursementgross_option: "A" | "B"

options = OneStepOptions(
    noemp_option=noemp_option,
    newexist_option=newexist_option,
    createjob_option=createjob_option,
    retainedjob_option=retainedjob_option,
    approvaldate_option=approvaldate_option,
    approvalfy_option=approvalfy_option,
    franchise_option=franchise_option,
    urbanrural_option=urbanrural_option,
    revlinecr_option=revlinecr_option,
    lowdoc_option=lowdoc_option,
    disbursementgross_option=disbursementgross_option,
    local_state=local_state,
)


# -----------------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------------
print("[SECTION] Running preprocessing")
df_processed = preprocess_one_step(df, options=options)
print(f"Rows: {len(df_processed):,}")
print(f"Features: {df_processed.shape[1]}")
df_processed.head()


# -----------------------------------------------------------------------------
# Split strategy
# -----------------------------------------------------------------------------
print("[SECTION] Building train/holdout split strategy")
target_col = "Accept"
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}\n")

X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train/Val set size: {X_trainval.shape[0]}")
print(f"Holdout set size: {X_holdout.shape[0]}")
print(f"Train/Val target distribution:\n{y_trainval.value_counts()}\n")
print(f"Holdout target distribution:\n{y_holdout.value_counts()}\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"StratifiedKFold splits: {skf.get_n_splits()}")
print("\nCross-validation fold distribution:")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
    print(f"  Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}")
print("\nHoldout set reserved for final reporting (untouched during training)")


# -----------------------------------------------------------------------------
# Model config and W&B run
# -----------------------------------------------------------------------------
print("[SECTION] Initializing model config and W&B run")
use_scaler = True

run = wandb.init(
    project="MS Geometric - SVM",
    config={
        "model_name": "LinearSVC",
        "random_state": 42,
        "max_iter": 10000,
        "use_scaler": use_scaler,
        "noemp_option": noemp_option,
        "newexist_option": newexist_option,
        "createjob_option": createjob_option,
        "retainedjob_option": retainedjob_option,
        "approvaldate_option": approvaldate_option,
        "approvalfy_option": approvalfy_option,
        "franchise_option": franchise_option,
        "urbanrural_option": urbanrural_option,
        "revlinecr_option": revlinecr_option,
        "lowdoc_option": lowdoc_option,
        "disbursementgross_option": disbursementgross_option,
        "local_state": local_state,
        "n_train_rows": int(X_trainval.shape[0]),
        "n_holdout_rows": int(X_holdout.shape[0]),
        "n_features": int(X_trainval.shape[1]),
    },
)


# -----------------------------------------------------------------------------
# Train and evaluate
# -----------------------------------------------------------------------------
print("[SECTION] Training LinearSVC pipeline")
svm_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler() if use_scaler else "passthrough"),
        ("model", LinearSVC(random_state=42, max_iter=10000)),
    ]
)

svm_pipeline.fit(X_trainval, y_trainval)

print("[SECTION] Running holdout predictions and metric evaluation")
y_pred = svm_pipeline.predict(X_holdout)
y_score = svm_pipeline.decision_function(X_holdout)

metrics = {
    "ROC-AUC": roc_auc_score(y_holdout, y_score),
    "PR-AUC": average_precision_score(y_holdout, y_score),
    "F1": f1_score(y_holdout, y_pred),
    "Precision": precision_score(y_holdout, y_pred),
    "Recall": recall_score(y_holdout, y_pred),
    "Accuracy": accuracy_score(y_holdout, y_pred),
}

cm = confusion_matrix(y_holdout, y_pred)
report = classification_report(y_holdout, y_pred, output_dict=True)
positive_rate = float((y_pred == 1).mean())
score_mean = float(y_score.mean())
score_std = float(y_score.std())

print(f"Use StandardScaler: {use_scaler}")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
print(f"Predicted positive rate: {positive_rate:.4f}")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
print("[SECTION] Generating ROC and PR plots")
fpr, tpr, _ = roc_curve(y_holdout, y_score)
precision, recall, _ = precision_recall_curve(y_holdout, y_score)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, label=f"ROC-AUC = {metrics['ROC-AUC']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.7)
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].plot(recall, precision, label=f"PR-AUC = {metrics['PR-AUC']:.4f}")
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# W&B logging
# -----------------------------------------------------------------------------
print("[SECTION] Logging metrics and artifacts to W&B")
wandb.log(
    {
        "roc_auc": metrics["ROC-AUC"],
        "pr_auc": metrics["PR-AUC"],
        "f1": metrics["F1"],
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "accuracy": metrics["Accuracy"],
        "predicted_positive_rate": positive_rate,
        "decision_score_mean": score_mean,
        "decision_score_std": score_std,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
)

wandb.log(
    {
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_holdout.tolist(),
            preds=y_pred.tolist(),
            class_names=["Reject", "Accept"],
        ),
        "roc_pr_curves": wandb.Image(fig),
    }
)

report_rows = []
for label, values in report.items():
    if isinstance(values, dict):
        report_rows.append(
            [
                label,
                float(values.get("precision", 0.0)),
                float(values.get("recall", 0.0)),
                float(values.get("f1-score", 0.0)),
                float(values.get("support", 0.0)),
            ]
        )

report_table = wandb.Table(
    columns=["label", "precision", "recall", "f1_score", "support"],
    data=report_rows,
)
wandb.log({"classification_report": report_table})

print("[SECTION] Finishing W&B run")
run.finish()