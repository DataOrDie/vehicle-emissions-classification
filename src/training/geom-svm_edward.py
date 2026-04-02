from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
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
newexist_option: str = "B"
createjob_option: str = "C" 
retainedjob_option: str = "B" 
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
svc_c: float = 1.0

# Hyperparameter search space for maximizing CV macro F1.
enable_hyperparameter_sweep: bool = True
sweep_reject_class_weights = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
sweep_c_values = [0.1, 0.3, 1.0, 3.0, 10.0]

# Optimize decisions for better class balance, not approval volume.
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
use_scaler = False
# use_scaler = True

if enable_hyperparameter_sweep and balance_strategy != "class_weight":
    print("[WARN] Disabling hyperparameter sweep because balance_strategy is not class_weight")
    enable_hyperparameter_sweep = False

run = wandb.init(
    project="MS Geometric - SVM",
    config={
        "model_name": "LinearSVC",
        "random_state": 42,
        "max_iter": 10000,
        "use_scaler": use_scaler,
        "balance_strategy": balance_strategy,
        "reject_class_weight": reject_class_weight,
        "svc_c": svc_c,
        "enable_hyperparameter_sweep": enable_hyperparameter_sweep,
        "sweep_reject_class_weights": sweep_reject_class_weights,
        "sweep_c_values": sweep_c_values,
        "optimize_metric": optimize_metric,
        "enable_threshold_tuning": enable_threshold_tuning,
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
print("[SECTION] Running cross-validation with hyperparameter search")
candidate_settings = []
if enable_hyperparameter_sweep:
    for candidate_reject_class_weight in sweep_reject_class_weights:
        for candidate_svc_c in sweep_c_values:
            candidate_settings.append(
                {
                    "reject_class_weight": float(candidate_reject_class_weight),
                    "svc_c": float(candidate_svc_c),
                }
            )
else:
    candidate_settings.append(
        {
            "reject_class_weight": float(reject_class_weight),
            "svc_c": float(svc_c),
        }
    )

candidate_results = []
for candidate_idx, candidate in enumerate(candidate_settings, 1):
    candidate_reject_class_weight = candidate["reject_class_weight"]
    candidate_svc_c = candidate["svc_c"]
    candidate_class_weight = (
        {0: candidate_reject_class_weight, 1: 1.0}
        if balance_strategy == "class_weight"
        else None
    )

    print(
        f"[CANDIDATE {candidate_idx}/{len(candidate_settings)}] "
        f"reject_class_weight={candidate_reject_class_weight:.2f}, C={candidate_svc_c:.2f}"
    )

    cv_fold_metrics_candidate = []
    oof_true_candidate = []
    oof_scores_candidate = []

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

        fold_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler() if use_scaler else "passthrough"),
                (
                    "model",
                    LinearSVC(
                        C=candidate_svc_c,
                        random_state=42,
                        max_iter=10000,
                        class_weight=candidate_class_weight,
                    ),
                ),
            ]
        )
        fold_pipeline.fit(X_fold_train, y_fold_train)

        y_fold_pred = fold_pipeline.predict(X_fold_val)
        y_fold_score = fold_pipeline.decision_function(X_fold_val)
        oof_true_candidate.append(y_fold_val)
        oof_scores_candidate.append(y_fold_score)

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
        cv_fold_metrics_candidate.append(fold_metrics)

    cv_summary_candidate = {
        "cv_mean_roc_auc": float(np.mean([m["roc_auc"] for m in cv_fold_metrics_candidate])),
        "cv_std_roc_auc": float(np.std([m["roc_auc"] for m in cv_fold_metrics_candidate])),
        "cv_mean_pr_auc": float(np.mean([m["pr_auc"] for m in cv_fold_metrics_candidate])),
        "cv_std_pr_auc": float(np.std([m["pr_auc"] for m in cv_fold_metrics_candidate])),
        "cv_mean_f1": float(np.mean([m["f1"] for m in cv_fold_metrics_candidate])),
        "cv_std_f1": float(np.std([m["f1"] for m in cv_fold_metrics_candidate])),
        "cv_mean_precision": float(np.mean([m["precision"] for m in cv_fold_metrics_candidate])),
        "cv_std_precision": float(np.std([m["precision"] for m in cv_fold_metrics_candidate])),
        "cv_mean_recall": float(np.mean([m["recall"] for m in cv_fold_metrics_candidate])),
        "cv_std_recall": float(np.std([m["recall"] for m in cv_fold_metrics_candidate])),
        "cv_mean_balanced_accuracy": float(
            np.mean([m["balanced_accuracy"] for m in cv_fold_metrics_candidate])
        ),
        "cv_std_balanced_accuracy": float(
            np.std([m["balanced_accuracy"] for m in cv_fold_metrics_candidate])
        ),
        "cv_mean_macro_f1": float(np.mean([m["macro_f1"] for m in cv_fold_metrics_candidate])),
        "cv_std_macro_f1": float(np.std([m["macro_f1"] for m in cv_fold_metrics_candidate])),
        "cv_mean_mcc": float(np.mean([m["mcc"] for m in cv_fold_metrics_candidate])),
        "cv_std_mcc": float(np.std([m["mcc"] for m in cv_fold_metrics_candidate])),
        "cv_mean_accuracy": float(np.mean([m["accuracy"] for m in cv_fold_metrics_candidate])),
        "cv_std_accuracy": float(np.std([m["accuracy"] for m in cv_fold_metrics_candidate])),
    }

    decision_threshold_candidate = 0.0
    threshold_objective_candidate = float("nan")
    if enable_threshold_tuning:
        y_oof_true_candidate = pd.concat(oof_true_candidate).reset_index(drop=True)
        y_oof_scores_candidate = np.concatenate(oof_scores_candidate)
        threshold_grid = np.quantile(y_oof_scores_candidate, np.linspace(0.05, 0.95, 121))

        best_threshold = 0.0
        best_threshold_score = -np.inf
        for threshold in threshold_grid:
            y_oof_pred = predict_with_threshold(y_oof_scores_candidate, float(threshold))
            metric_value = score_threshold(y_oof_true_candidate, y_oof_pred, optimize_metric)
            if metric_value > best_threshold_score:
                best_threshold_score = metric_value
                best_threshold = float(threshold)

        decision_threshold_candidate = best_threshold
        threshold_objective_candidate = float(best_threshold_score)

    candidate_results.append(
        {
            "candidate_idx": candidate_idx,
            "reject_class_weight": candidate_reject_class_weight,
            "svc_c": candidate_svc_c,
            "class_weight": candidate_class_weight,
            "cv_fold_metrics": cv_fold_metrics_candidate,
            "cv_summary": cv_summary_candidate,
            "decision_threshold": decision_threshold_candidate,
            "threshold_objective": threshold_objective_candidate,
        }
    )

    print(
        f"  CV Macro-F1={cv_summary_candidate['cv_mean_macro_f1']:.4f} +/- "
        f"{cv_summary_candidate['cv_std_macro_f1']:.4f} | "
        f"Threshold={decision_threshold_candidate:.5f}"
    )

    wandb.log(
        {
            "sweep/candidate_idx": candidate_idx,
            "sweep/reject_class_weight": candidate_reject_class_weight,
            "sweep/svc_c": candidate_svc_c,
            "sweep/decision_threshold": decision_threshold_candidate,
            "sweep/threshold_objective": threshold_objective_candidate,
            "sweep/cv_mean_macro_f1": cv_summary_candidate["cv_mean_macro_f1"],
            "sweep/cv_std_macro_f1": cv_summary_candidate["cv_std_macro_f1"],
            "sweep/cv_mean_balanced_accuracy": cv_summary_candidate["cv_mean_balanced_accuracy"],
            "sweep/cv_mean_mcc": cv_summary_candidate["cv_mean_mcc"],
            "sweep/cv_mean_roc_auc": cv_summary_candidate["cv_mean_roc_auc"],
            "sweep/cv_mean_pr_auc": cv_summary_candidate["cv_mean_pr_auc"],
        }
    )

best_candidate = max(candidate_results, key=lambda c: c["cv_summary"]["cv_mean_macro_f1"])
selected_reject_class_weight = best_candidate["reject_class_weight"]
selected_svc_c = best_candidate["svc_c"]
selected_class_weight = best_candidate["class_weight"]
cv_fold_metrics = best_candidate["cv_fold_metrics"]
cv_summary = best_candidate["cv_summary"]
decision_threshold = best_candidate["decision_threshold"]

print("[SECTION] Selected best hyperparameters by CV Macro-F1")
print(f"Selected reject_class_weight: {selected_reject_class_weight:.2f}")
print(f"Selected C: {selected_svc_c:.2f}")
print(f"Selected threshold: {decision_threshold:.5f}")
print(f"Best CV Macro-F1: {cv_summary['cv_mean_macro_f1']:.4f} +/- {cv_summary['cv_std_macro_f1']:.4f}")

wandb.log(
    {
        "selection/reject_class_weight": selected_reject_class_weight,
        "selection/svc_c": selected_svc_c,
        "selection/decision_threshold": decision_threshold,
        "selection/cv_mean_macro_f1": cv_summary["cv_mean_macro_f1"],
        "selection/cv_std_macro_f1": cv_summary["cv_std_macro_f1"],
        "selection/candidate_idx": int(best_candidate["candidate_idx"]),
    }
)
run.config.update(
    {
        "selected_reject_class_weight": selected_reject_class_weight,
        "selected_svc_c": selected_svc_c,
        "selected_decision_threshold": decision_threshold,
    },
    allow_val_change=True,
)

print("[SECTION] Cross-validation summary")
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
    columns=[
        "fold",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "accuracy",
    ],
    data=[
        [
            int(m["fold"]),
            float(m["roc_auc"]),
            float(m["pr_auc"]),
            float(m["f1"]),
            float(m["precision"]),
            float(m["recall"]),
            float(m["balanced_accuracy"]),
            float(m["macro_f1"]),
            float(m["mcc"]),
            float(m["accuracy"]),
        ]
        for m in cv_fold_metrics
    ],
)
wandb.log({"cv/folds_table": cv_table, **cv_summary})

print(
    f"[SECTION] Using selected threshold from best candidate: {decision_threshold:.5f} "
    f"(optimized for {optimize_metric})"
)

svm_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler() if use_scaler else "passthrough"),
        (
            "model",
            LinearSVC(
                C=selected_svc_c,
                random_state=42,
                max_iter=10000,
                class_weight=selected_class_weight,
            ),
        ),
    ]
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
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "balanced_accuracy": metrics["Balanced-Accuracy"],
        "macro_f1": metrics["Macro-F1"],
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
