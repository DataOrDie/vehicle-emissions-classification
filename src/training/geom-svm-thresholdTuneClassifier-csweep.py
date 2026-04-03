from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
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
from submit.save_model import save_model


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
noemp_option: str = "C"
newexist_option: str = "A"
createjob_option: str = "C" # same
retainedjob_option: str = "B" # same
disbursementgross_option: str = "C" 

approvaldate_option: str = "A" # only A
approvalfy_option: str = "A" # only A
franchise_option: str = "binary" # only binary
urbanrural_option: str = "onehot" # only onehot
revlinecr_option: str = "C" # only C 
lowdoc_option: str = "C" # only C

local_state: str = "IL"

# Accepted values:
# noemp_option: "raw" | "log" | "binning" | "C"
# newexist_option: "A" | "B"
# createjob_option: "A" | "B" | "C"
# retainedjob_option: "A" | "B" | "C"
# approvaldate_option: "A" | "B"
# approvalfy_option: "A" | "B"
# franchise_option: "binary" | "raw"
# urbanrural_option: "onehot" | "text"
# revlinecr_option: "A" | "B" | "C"
# lowdoc_option: "A" | "B" | "C"
# disbursementgross_option: "A" | "B" | "C"

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
n_splits = skf.get_n_splits()


# -----------------------------------------------------------------------------
# Class balancing helpers
# -----------------------------------------------------------------------------
print("[SECTION] Configuring class balance strategy")
balance_strategy: str = "class_weight"
# balance_strategy: str = "oversample_reject"
# balance_strategy: str = "undersample_approve"
# Options:
#   - "none"
#   - "class_weight"
#   - "oversample_reject"
#   - "undersample_approve"
reject_class_weight: float = 2.0

# Optimize decisions for better class balance with macro-F1 as the objective.
optimize_metric: str = "macro_f1"
# Options: "balanced_accuracy" | "macro_f1" | "mcc"
enable_threshold_tuning: bool = True


def rebalance_training_data(X_train, y_train, strategy: str, random_state: int = 42):
    """Return a rebalanced copy of the training split when requested.

    The target uses `Accept`, where 0 means reject. The two resampling modes
    intentionally bias the training data toward the reject class.
    """

    if strategy == "none":
        return X_train, y_train

    train_frame = X_train.copy()
    train_frame["Accept"] = y_train.values

    reject_mask = train_frame["Accept"] == 0
    reject_rows = train_frame[reject_mask]
    approve_rows = train_frame[~reject_mask]

    if reject_rows.empty or approve_rows.empty:
        return X_train, y_train

    if strategy == "oversample_reject":
        # Upsample rejected loans to match approved loans.
        reject_rows = reject_rows.sample(
            n=len(approve_rows),
            replace=True,
            random_state=random_state,
        )
        balanced = pd.concat([approve_rows, reject_rows], ignore_index=True)
    elif strategy == "undersample_approve":
        # Downsample approved loans to match rejected loans.
        if len(approve_rows) <= len(reject_rows):
            return X_train, y_train
        approve_rows = approve_rows.sample(
            n=len(reject_rows),
            replace=False,
            random_state=random_state,
        )
        balanced = pd.concat([approve_rows, reject_rows], ignore_index=True)
    else:
        raise ValueError(
            f"Unknown balance_strategy: {strategy}. Use none, class_weight, oversample_reject, or undersample_approve."
        )

    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced.drop(columns=["Accept"]), balanced["Accept"]


def predict_with_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert decision scores to class labels using a configurable cutoff."""

    return (scores >= threshold).astype(int)


def score_threshold(y_true: pd.Series, y_pred: np.ndarray, metric_name: str) -> float:
    """Evaluate threshold quality for balanced two-class performance."""

    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric_name == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric_name == "mcc":
        return matthews_corrcoef(y_true, y_pred)
    raise ValueError(f"Unknown optimize_metric: {metric_name}")


# -----------------------------------------------------------------------------
# Model config and W&B run
# -----------------------------------------------------------------------------
print("[SECTION] Initializing model config and W&B run")
# use_scaler = True
use_scaler = False
class_weight = {0: reject_class_weight, 1: 1.0} if balance_strategy == "class_weight" else None
candidate_c_values = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]

run = wandb.init(
    project="MS Geometric - SVM",
    config={
        "model_name": "LinearSVC_threshold_tuning_csweep",
        "random_state": 42,
        "max_iter": 10000,
        "use_scaler": use_scaler,
        "balance_strategy": balance_strategy,
        "reject_class_weight": reject_class_weight,
        "optimize_metric": optimize_metric,
        "enable_threshold_tuning": enable_threshold_tuning,
        "candidate_c_values": candidate_c_values,
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
        "cv_n_splits": n_splits,
        "n_train_rows": int(X_trainval.shape[0]),
        "n_holdout_rows": int(X_holdout.shape[0]),
        "n_features": int(X_trainval.shape[1]),
    },
)


# -----------------------------------------------------------------------------
# Train and evaluate
# -----------------------------------------------------------------------------
print("[SECTION] Training LinearSVC pipeline")
def build_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler() if use_scaler else "passthrough"),
            (
                "model",
                LinearSVC(
                    random_state=42,
                    max_iter=10000,
                    C=c_value,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

# Use StratifiedKFold for real cross-validation on the train/val partition.
# This gives fold-level and aggregate estimates before touching the holdout set.
print("[SECTION] Running cross-validation on train/val split")
cv_results_by_c = []

for c_value in candidate_c_values:
    cv_fold_metrics = []
    oof_true = []
    oof_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
        X_fold_train = X_trainval.iloc[train_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        X_fold_val = X_trainval.iloc[val_idx]
        y_fold_val = y_trainval.iloc[val_idx]

        if balance_strategy in {"oversample_reject", "undersample_approve"}:
            X_fold_train, y_fold_train = rebalance_training_data(
                X_fold_train,
                y_fold_train,
                strategy=balance_strategy,
                random_state=42,
            )

        fold_pipeline = build_pipeline(c_value)
        fold_pipeline.fit(X_fold_train, y_fold_train)

        y_fold_pred = fold_pipeline.predict(X_fold_val)
        y_fold_score = fold_pipeline.decision_function(X_fold_val)
        oof_true.append(y_fold_val)
        oof_scores.append(y_fold_score)

        fold_metrics = {
            "fold": fold_idx,
            "roc_auc": roc_auc_score(y_fold_val, y_fold_score),
            "pr_auc": average_precision_score(y_fold_val, y_fold_score),
            "f1": f1_score(y_fold_val, y_fold_pred, zero_division=0),
            "precision": precision_score(y_fold_val, y_fold_pred, zero_division=0),
            "recall": recall_score(y_fold_val, y_fold_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_fold_val, y_fold_pred),
            "macro_f1": f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y_fold_val, y_fold_pred),
            "accuracy": accuracy_score(y_fold_val, y_fold_pred),
        }
        cv_fold_metrics.append(fold_metrics)

    cv_summary = {
        "C": float(c_value),
        "cv_mean_roc_auc": float(np.mean([m["roc_auc"] for m in cv_fold_metrics])),
        "cv_std_roc_auc": float(np.std([m["roc_auc"] for m in cv_fold_metrics])),
        "cv_mean_pr_auc": float(np.mean([m["pr_auc"] for m in cv_fold_metrics])),
        "cv_std_pr_auc": float(np.std([m["pr_auc"] for m in cv_fold_metrics])),
        "cv_mean_f1": float(np.mean([m["f1"] for m in cv_fold_metrics])),
        "cv_std_f1": float(np.std([m["f1"] for m in cv_fold_metrics])),
        "cv_mean_precision": float(np.mean([m["precision"] for m in cv_fold_metrics])),
        "cv_std_precision": float(np.std([m["precision"] for m in cv_fold_metrics])),
        "cv_mean_recall": float(np.mean([m["recall"] for m in cv_fold_metrics])),
        "cv_std_recall": float(np.std([m["recall"] for m in cv_fold_metrics])),
        "cv_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in cv_fold_metrics])),
        "cv_std_balanced_accuracy": float(np.std([m["balanced_accuracy"] for m in cv_fold_metrics])),
        "cv_mean_macro_f1": float(np.mean([m["macro_f1"] for m in cv_fold_metrics])),
        "cv_std_macro_f1": float(np.std([m["macro_f1"] for m in cv_fold_metrics])),
        "cv_mean_mcc": float(np.mean([m["mcc"] for m in cv_fold_metrics])),
        "cv_std_mcc": float(np.std([m["mcc"] for m in cv_fold_metrics])),
        "cv_mean_accuracy": float(np.mean([m["accuracy"] for m in cv_fold_metrics])),
        "cv_std_accuracy": float(np.std([m["accuracy"] for m in cv_fold_metrics])),
    }

    decision_threshold = 0.0
    tuned_objective = cv_summary[f"cv_mean_{optimize_metric}"]
    if enable_threshold_tuning:
        y_oof_true = pd.concat(oof_true).reset_index(drop=True)
        y_oof_scores = np.concatenate(oof_scores)
        threshold_grid = np.quantile(y_oof_scores, np.linspace(0.05, 0.95, 121))

        best_threshold = 0.0
        best_threshold_score = -np.inf
        for threshold in threshold_grid:
            y_oof_pred = predict_with_threshold(y_oof_scores, float(threshold))
            metric_value = score_threshold(y_oof_true, y_oof_pred, optimize_metric)
            if metric_value > best_threshold_score:
                best_threshold_score = metric_value
                best_threshold = float(threshold)

        decision_threshold = best_threshold
        tuned_objective = float(best_threshold_score)
        cv_summary["cv_oof_threshold"] = decision_threshold
        cv_summary["cv_oof_objective"] = tuned_objective

    cv_results_by_c.append((c_value, decision_threshold, tuned_objective, cv_summary, cv_fold_metrics))

best_c_value, decision_threshold, best_oof_objective, cv_summary, cv_fold_metrics = max(
    cv_results_by_c,
    key=lambda item: (item[2], item[3]["cv_mean_roc_auc"]),
)

print("[SECTION] Cross-validation summary")
for c_value, threshold_value, objective_value, summary, _ in cv_results_by_c:
    print(
        f"C={c_value:g} | "
        f"CV MACRO-F1={summary['cv_mean_macro_f1']:.4f} +/- {summary['cv_std_macro_f1']:.4f} "
        f"CV ACC={summary['cv_mean_accuracy']:.4f} +/- {summary['cv_std_accuracy']:.4f} "
        f"Threshold={threshold_value:.5f} "
        f"OOF {optimize_metric}={objective_value:.4f}"
    )

print(
    f"Best C selected by OOF {optimize_metric}: {best_c_value:g} "
    f"(threshold={decision_threshold:.5f}, score={best_oof_objective:.4f})"
)

for metric_name in [
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "accuracy",
]:
    print(
        f"CV {metric_name.upper()}: "
        f"{cv_summary[f'cv_mean_{metric_name}']:.4f} +/- "
        f"{cv_summary[f'cv_std_{metric_name}']:.4f}"
    )

# Log fold-level table and CV aggregate summary to W&B.
cv_table = wandb.Table(
    columns=["fold", "roc_auc", "pr_auc", "f1", "macro_f1", "precision", "recall", "accuracy"],
    data=[
        [
            int(m["fold"]),
            float(m["roc_auc"]),
            float(m["pr_auc"]),
            float(m["f1"]),
            float(m["macro_f1"]),
            float(m["precision"]),
            float(m["recall"]),
            float(m["accuracy"]),
        ]
        for m in cv_fold_metrics
    ],
)

for fold_metrics in cv_fold_metrics:
    wandb.log(
        {
            "cv/fold": int(fold_metrics["fold"]),
            "cv/roc_auc": float(fold_metrics["roc_auc"]),
            "cv/pr_auc": float(fold_metrics["pr_auc"]),
            "cv/f1": float(fold_metrics["f1"]),
            "cv/macro_f1": float(fold_metrics["macro_f1"]),
            "cv/precision": float(fold_metrics["precision"]),
            "cv/recall": float(fold_metrics["recall"]),
            "cv/accuracy": float(fold_metrics["accuracy"]),
        }
    )

csweep_table = wandb.Table(
    columns=[
        "C",
        "threshold",
        "oof_objective",
        "cv_mean_macro_f1",
        "cv_mean_accuracy",
        "cv_mean_roc_auc",
        "cv_mean_pr_auc",
    ],
    data=[
        [
            float(c_value),
            float(threshold_value),
            float(objective_value),
            float(summary["cv_mean_macro_f1"]),
            float(summary["cv_mean_accuracy"]),
            float(summary["cv_mean_roc_auc"]),
            float(summary["cv_mean_pr_auc"]),
        ]
        for c_value, threshold_value, objective_value, summary, _ in cv_results_by_c
    ],
)

wandb.log(
    {
        "cv/folds_table": cv_table,
        "cv/csweep_table": csweep_table,
        "selected_c": best_c_value,
        "threshold/selected": decision_threshold,
        "threshold/oof_objective": best_oof_objective,
        **cv_summary,
    }
)

# Refit on the full train/val data after CV, then evaluate once on holdout.
if balance_strategy in {"oversample_reject", "undersample_approve"}:
    X_trainval_fit, y_trainval_fit = rebalance_training_data(
        X_trainval,
        y_trainval,
        strategy=balance_strategy,
        random_state=42,
    )
else:
    X_trainval_fit, y_trainval_fit = X_trainval, y_trainval

svm_pipeline = build_pipeline(best_c_value)
svm_pipeline.fit(X_trainval_fit, y_trainval_fit)

print("[SECTION] Running holdout predictions and metric evaluation")
y_score = svm_pipeline.decision_function(X_holdout)
y_pred = predict_with_threshold(y_score, decision_threshold)

metrics = {
    "ROC-AUC": roc_auc_score(y_holdout, y_score),
    "PR-AUC": average_precision_score(y_holdout, y_score),
    "F1": f1_score(y_holdout, y_pred, zero_division=0),
    "Precision": precision_score(y_holdout, y_pred, zero_division=0),
    "Recall": recall_score(y_holdout, y_pred, zero_division=0),
    "Balanced-Accuracy": balanced_accuracy_score(y_holdout, y_pred),
    "Macro-F1": f1_score(y_holdout, y_pred, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_holdout, y_pred),
    "Accuracy": accuracy_score(y_holdout, y_pred),
}

cm = confusion_matrix(y_holdout, y_pred)
report = classification_report(y_holdout, y_pred, output_dict=True)
positive_rate = float((y_pred == 1).mean())
score_mean = float(y_score.mean())
score_std = float(y_score.std())

print(f"Use StandardScaler: {use_scaler}")
print(f"Decision threshold: {decision_threshold:.5f}")
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
        "macro_f1": metrics["Macro-F1"],
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "balanced_accuracy": metrics["Balanced-Accuracy"],
        "mcc": metrics["MCC"],
        "accuracy": metrics["Accuracy"],
        "decision_threshold": decision_threshold,
        "predicted_positive_rate": positive_rate,
        "decision_score_mean": score_mean,
        "decision_score_std": score_std,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
)

run.summary["macro_f1"] = metrics["Macro-F1"]
run.summary["holdout/macro_f1"] = metrics["Macro-F1"]

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

# Save the trained model and preprocessing options for submission
saved_paths = save_model(
    model_pipeline=svm_pipeline,
    preprocessing_options=options,
    feature_names=X_trainval.columns.tolist(),
    project_root=project_root,
    model_name="geom-svm-thresholdTuneClassifier-csweep",
)

# Quick read on the metrics:

# Best setting is clearly C=0.01 with threshold about 0.06.
# CV macro-F1 is 0.6473 and holdout Macro-F1 is 0.6439, so the gap is only 
# 0.0034
# 0.0034 which is small and healthy.
# Holdout balanced accuracy 0.6553 and MCC 0.2921 are consistent with real improvement, not just a noisy spike.
# One useful interpretation:

# Predicted positive rate is 0.7248 while holdout positive base rate is about 
# 3203
# /
# 4153
# ≈
# 0.771
# 3203/4153≈0.771, so the tuned threshold is intentionally a bit more conservative on approvals, which is likely helping Macro-F1 balance.
# If you want, I can do one more targeted pass to try for another small gain:

# Narrow C sweep around 0.01 (for example 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03).
# Finer threshold grid around the selected band (for example 0.03 to 0.09).
# Keep the exact same W&B key names so your dashboards stay consistent.


# our cv/macro_f1 increased slightly. we are going good :) 
# looks this is our best so far :) 
# C=0.01 | CV MACRO-F1=0.6473 +/- 0.0134 CV ACC=0.7483 +/- 0.0113 Threshold=0.05999 OOF macro_f1=0.6483
# C=0.1 | CV MACRO-F1=0.6465 +/- 0.0130 CV ACC=0.7465 +/- 0.0108 Threshold=0.04594 OOF macro_f1=0.6471
# C=0.3 | CV MACRO-F1=0.6466 +/- 0.0133 CV ACC=0.7466 +/- 0.0110 Threshold=0.04497 OOF macro_f1=0.6469
# C=1 | CV MACRO-F1=0.6465 +/- 0.0133 CV ACC=0.7466 +/- 0.0111 Threshold=0.04490 OOF macro_f1=0.6472
# C=3 | CV MACRO-F1=0.6464 +/- 0.0133 CV ACC=0.7464 +/- 0.0112 Threshold=0.04472 OOF macro_f1=0.6471
# C=10 | CV MACRO-F1=0.6464 +/- 0.0133 CV ACC=0.7464 +/- 0.0112 Threshold=0.04476 OOF macro_f1=0.6471
# Best C selected by OOF macro_f1: 0.01 (threshold=0.05999, score=0.6483)
# CV ROC_AUC: 0.7252 +/- 0.0195
# CV PR_AUC: 0.8853 +/- 0.0124
# CV F1: 0.8360 +/- 0.0080
# CV PRECISION: 0.8401 +/- 0.0061
# CV RECALL: 0.8320 +/- 0.0121
# CV BALANCED_ACCURACY: 0.6490 +/- 0.0132
# CV MACRO_F1: 0.6473 +/- 0.0134
# CV MCC: 0.2949 +/- 0.0269
# CV ACCURACY: 0.7483 +/- 0.0113
# [SECTION] Running holdout predictions and metric evaluation
# Use StandardScaler: False
# Decision threshold: 0.05999
# ROC-AUC: 0.7221
# PR-AUC: 0.8814
# F1: 0.8205
# Precision: 0.8468
# Recall: 0.7958
# Balanced-Accuracy: 0.6553
# Macro-F1: 0.6439
# MCC: 0.2921
# Accuracy: 0.7315
# Predicted positive rate: 0.7248