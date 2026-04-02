from pathlib import Path
import sys
import time

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
from sklearn.svm import SVC


# -----------------------------------------------------------------------------
# W&B authentication
# -----------------------------------------------------------------------------
print("[SECTION] W&B authentication")
wandb.login()

# Start total execution timer
script_start_time = time.perf_counter()


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
newexist_option: str = "B" # do not change
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
use_scaler = False

balance_strategy: str = "class_weight"
# balance_strategy: str = "oversample_reject"
# balance_strategy: str = "undersample_approve"
# Options:
#   - "none"
#   - "class_weight"
#   - "oversample_reject"
#   - "undersample_approve"
reject_class_weight: float = 2.0


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


def find_best_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    optimize_metric: str,
    n_candidates: int = 101,
) -> tuple[float, float]:
    """Select the threshold that maximizes the chosen validation metric."""

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0, float("nan")

    low = float(np.quantile(scores, 0.01))
    high = float(np.quantile(scores, 0.99))
    if low == high:
        threshold = 0.0
        preds = predict_with_threshold(scores, threshold)
        return threshold, score_threshold(y_true, preds, optimize_metric)

    candidate_thresholds = np.unique(
        np.concatenate([np.linspace(low, high, n_candidates), np.array([0.0])])
    )

    best_threshold = 0.0
    best_metric = -np.inf
    for threshold in candidate_thresholds:
        preds = predict_with_threshold(scores, float(threshold))
        metric_value = score_threshold(y_true, preds, optimize_metric)
        if (metric_value > best_metric) or (
            metric_value == best_metric and abs(float(threshold)) < abs(best_threshold)
        ):
            best_metric = float(metric_value)
            best_threshold = float(threshold)

    return best_threshold, best_metric


# -----------------------------------------------------------------------------
# Model config and W&B run
# -----------------------------------------------------------------------------
print("[SECTION] Initializing model config and W&B run")
class_weight = {0: reject_class_weight, 1: 1.0} if balance_strategy == "class_weight" else None

# Use a high but bounded cap to avoid very long runs.
svc_max_iter: int = 50000
svc_cache_size_mb: int = 1024

# Keep this intentionally small to control kernel SVM training cost.
c_candidates = [0.5, 1.0, 2.0]
gamma_candidates = ["scale", 0.01, 0.001]
degree_candidates = [2, 3, 4]
coef0_candidates = [0.0, 1.0]

optimize_metric: str = "macro_f1"
threshold_candidates_per_fold: int = 101

# # Expanded grid now that runtime is acceptable.
# c_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
# gamma_candidates = ["scale", 0.03, 0.01, 0.003, 0.001]

run = wandb.init(
    project="MS Geometric - SVM",
    config={
        "model_name": "SVC_POLY",
        "kernel": "poly",
        "random_state": 42,
        "max_iter": svc_max_iter,
        "cache_size_mb": svc_cache_size_mb,
        "use_scaler": use_scaler,
        "balance_strategy": balance_strategy,
        "reject_class_weight": reject_class_weight,
        "c_candidates": c_candidates,
        "gamma_candidates": gamma_candidates,
        "degree_candidates": degree_candidates,
        "coef0_candidates": coef0_candidates,
        "optimize_metric": optimize_metric,
        "threshold_candidates_per_fold": threshold_candidates_per_fold,
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
print("[SECTION] Sweeping polynomial kernel hyperparameters (C, gamma, degree, coef0)")
sweep_start_time = time.perf_counter()
sweep_results = []

for c_value in c_candidates:
    for gamma_value in gamma_candidates:
        for degree_value in degree_candidates:
            for coef0_value in coef0_candidates:
                combo_macro_f1_scores = []

                for train_idx, val_idx in skf.split(X_trainval, y_trainval):
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

                    sweep_pipeline = Pipeline(
                        steps=[
                            ("scaler", StandardScaler() if use_scaler else "passthrough"),
                            (
                                "model",
                                SVC(
                                    kernel="poly",
                                    C=c_value,
                                    gamma=gamma_value,
                                    degree=degree_value,
                                    coef0=coef0_value,
                                    random_state=42,
                                    max_iter=svc_max_iter,
                                    cache_size=svc_cache_size_mb,
                                    class_weight=class_weight,
                                ),
                            ),
                        ]
                    )
                    sweep_pipeline.fit(X_fold_train, y_fold_train)

                    y_fold_pred = sweep_pipeline.predict(X_fold_val)
                    combo_macro_f1_scores.append(
                        f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0)
                    )

                combo_mean_macro_f1 = float(np.mean(combo_macro_f1_scores))
                sweep_result = {
                    "C": c_value,
                    "gamma": gamma_value,
                    "degree": degree_value,
                    "coef0": coef0_value,
                    "cv_mean_macro_f1": combo_mean_macro_f1,
                }
                sweep_results.append(sweep_result)
                print(
                    f"Sweep C={c_value}, gamma={gamma_value}, degree={degree_value}, coef0={coef0_value} | "
                    f"mean macro-F1={combo_mean_macro_f1:.4f}"
                )
                wandb.log(
                    {
                        "sweep/C": c_value,
                        "sweep/gamma": str(gamma_value),
                        "sweep/degree": degree_value,
                        "sweep/coef0": coef0_value,
                        "sweep/cv_mean_macro_f1": combo_mean_macro_f1,
                    }
                )

best_sweep = max(sweep_results, key=lambda row: row["cv_mean_macro_f1"])
best_c = best_sweep["C"]
best_gamma = best_sweep["gamma"]
best_degree = best_sweep["degree"]
best_coef0 = best_sweep["coef0"]
sweep_elapsed = time.perf_counter() - sweep_start_time

print(
    f"Best kernel params: C={best_c}, gamma={best_gamma}, degree={best_degree}, coef0={best_coef0} "
    f"(CV mean macro-F1={best_sweep['cv_mean_macro_f1']:.4f})"
)
print(f"[TIMING] Sweep completed in {sweep_elapsed:.2f} seconds ({sweep_elapsed/60:.2f} minutes)")

wandb.log(
    {
        "best_params/C": best_c,
        "best_params/gamma": str(best_gamma),
        "best_params/degree": best_degree,
        "best_params/coef0": best_coef0,
        "best_params/cv_mean_macro_f1": best_sweep["cv_mean_macro_f1"],
    }
)
run.summary["best_c"] = float(best_c)
run.summary["best_gamma"] = str(best_gamma)
run.summary["best_degree"] = int(best_degree)
run.summary["best_coef0"] = float(best_coef0)
run.summary["best_cv_mean_macro_f1"] = float(best_sweep["cv_mean_macro_f1"])

print("[SECTION] Training/evaluating polynomial SVC with best kernel params")
svm_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler() if use_scaler else "passthrough"),
        (
            "model",
            SVC(
                kernel="poly",
                C=best_c,
                gamma=best_gamma,
                degree=best_degree,
                coef0=best_coef0,
                random_state=42,
                max_iter=svc_max_iter,
                cache_size=svc_cache_size_mb,
                class_weight=class_weight,
            ),
        ),
    ]
)

# Use StratifiedKFold for fold-level metrics with the selected hyperparameters.
print("[SECTION] Running cross-validation on train/val split")
cv_start_time = time.perf_counter()
cv_fold_metrics = []
cv_best_thresholds = []

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
                SVC(
                    kernel="poly",
                    C=best_c,
                    gamma=best_gamma,
                    degree=best_degree,
                    coef0=best_coef0,
                    random_state=42,
                    max_iter=svc_max_iter,
                    cache_size=svc_cache_size_mb,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    fold_pipeline.fit(X_fold_train, y_fold_train)

    y_fold_score = fold_pipeline.decision_function(X_fold_val)
    y_fold_pred_default = fold_pipeline.predict(X_fold_val)
    fold_best_threshold, fold_best_metric = find_best_threshold(
        y_true=y_fold_val,
        scores=y_fold_score,
        optimize_metric=optimize_metric,
        n_candidates=threshold_candidates_per_fold,
    )
    cv_best_thresholds.append(fold_best_threshold)
    y_fold_pred_tuned = predict_with_threshold(y_fold_score, fold_best_threshold)

    fold_metrics = {
        "fold": fold_idx,
        "roc_auc": roc_auc_score(y_fold_val, y_fold_score),
        "pr_auc": average_precision_score(y_fold_val, y_fold_score),
        "f1": f1_score(y_fold_val, y_fold_pred_default, zero_division=0),
        "macro_f1": f1_score(y_fold_val, y_fold_pred_default, average="macro", zero_division=0),
        "precision": precision_score(y_fold_val, y_fold_pred_default, zero_division=0),
        "recall": recall_score(y_fold_val, y_fold_pred_default, zero_division=0),
        "accuracy": accuracy_score(y_fold_val, y_fold_pred_default),
        "best_threshold": fold_best_threshold,
        "best_threshold_metric": fold_best_metric,
        "tuned_f1": f1_score(y_fold_val, y_fold_pred_tuned, zero_division=0),
        "tuned_macro_f1": f1_score(y_fold_val, y_fold_pred_tuned, average="macro", zero_division=0),
        "tuned_precision": precision_score(y_fold_val, y_fold_pred_tuned, zero_division=0),
        "tuned_recall": recall_score(y_fold_val, y_fold_pred_tuned, zero_division=0),
        "tuned_accuracy": accuracy_score(y_fold_val, y_fold_pred_tuned),
    }
    cv_fold_metrics.append(fold_metrics)

    print(
        f"Fold {fold_idx} | "
        f"ROC-AUC={fold_metrics['roc_auc']:.4f} "
        f"PR-AUC={fold_metrics['pr_auc']:.4f} "
        f"F1={fold_metrics['f1']:.4f} "
        f"Macro-F1={fold_metrics['macro_f1']:.4f} "
        f"| tuned threshold={fold_metrics['best_threshold']:.4f} "
        f"tuned Macro-F1={fold_metrics['tuned_macro_f1']:.4f}"
    )

    wandb.log(
        {
            "cv/fold": fold_idx,
            "cv/roc_auc": fold_metrics["roc_auc"],
            "cv/pr_auc": fold_metrics["pr_auc"],
            "cv/f1": fold_metrics["f1"],
            "cv/macro_f1": fold_metrics["macro_f1"],
            "cv/precision": fold_metrics["precision"],
            "cv/recall": fold_metrics["recall"],
            "cv/accuracy": fold_metrics["accuracy"],
            "cv/best_threshold": fold_metrics["best_threshold"],
            "cv/best_threshold_metric": fold_metrics["best_threshold_metric"],
            "cv/tuned_f1": fold_metrics["tuned_f1"],
            "cv/tuned_macro_f1": fold_metrics["tuned_macro_f1"],
            "cv/tuned_precision": fold_metrics["tuned_precision"],
            "cv/tuned_recall": fold_metrics["tuned_recall"],
            "cv/tuned_accuracy": fold_metrics["tuned_accuracy"],
        }
    )

selected_threshold = float(np.median(cv_best_thresholds))
run.summary["selected_threshold"] = selected_threshold

cv_summary = {
    "cv_mean_roc_auc": float(np.mean([m["roc_auc"] for m in cv_fold_metrics])),
    "cv_std_roc_auc": float(np.std([m["roc_auc"] for m in cv_fold_metrics])),
    "cv_mean_pr_auc": float(np.mean([m["pr_auc"] for m in cv_fold_metrics])),
    "cv_std_pr_auc": float(np.std([m["pr_auc"] for m in cv_fold_metrics])),
    "cv_mean_f1": float(np.mean([m["f1"] for m in cv_fold_metrics])),
    "cv_std_f1": float(np.std([m["f1"] for m in cv_fold_metrics])),
    "cv_mean_macro_f1": float(np.mean([m["macro_f1"] for m in cv_fold_metrics])),
    "cv_std_macro_f1": float(np.std([m["macro_f1"] for m in cv_fold_metrics])),
    "cv_mean_precision": float(np.mean([m["precision"] for m in cv_fold_metrics])),
    "cv_std_precision": float(np.std([m["precision"] for m in cv_fold_metrics])),
    "cv_mean_recall": float(np.mean([m["recall"] for m in cv_fold_metrics])),
    "cv_std_recall": float(np.std([m["recall"] for m in cv_fold_metrics])),
    "cv_mean_accuracy": float(np.mean([m["accuracy"] for m in cv_fold_metrics])),
    "cv_std_accuracy": float(np.std([m["accuracy"] for m in cv_fold_metrics])),
    "cv_mean_threshold": float(np.mean(cv_best_thresholds)),
    "cv_std_threshold": float(np.std(cv_best_thresholds)),
    "cv_selected_threshold": selected_threshold,
    "cv_mean_tuned_f1": float(np.mean([m["tuned_f1"] for m in cv_fold_metrics])),
    "cv_std_tuned_f1": float(np.std([m["tuned_f1"] for m in cv_fold_metrics])),
    "cv_mean_tuned_macro_f1": float(np.mean([m["tuned_macro_f1"] for m in cv_fold_metrics])),
    "cv_std_tuned_macro_f1": float(np.std([m["tuned_macro_f1"] for m in cv_fold_metrics])),
    "cv_mean_tuned_precision": float(np.mean([m["tuned_precision"] for m in cv_fold_metrics])),
    "cv_std_tuned_precision": float(np.std([m["tuned_precision"] for m in cv_fold_metrics])),
    "cv_mean_tuned_recall": float(np.mean([m["tuned_recall"] for m in cv_fold_metrics])),
    "cv_std_tuned_recall": float(np.std([m["tuned_recall"] for m in cv_fold_metrics])),
    "cv_mean_tuned_accuracy": float(np.mean([m["tuned_accuracy"] for m in cv_fold_metrics])),
    "cv_std_tuned_accuracy": float(np.std([m["tuned_accuracy"] for m in cv_fold_metrics])),
}

cv_elapsed = time.perf_counter() - cv_start_time

print("[SECTION] Cross-validation summary")
for metric_name in ["roc_auc", "pr_auc", "f1", "macro_f1", "precision", "recall", "accuracy"]:
    print(
        f"CV {metric_name.upper()}: "
        f"{cv_summary[f'cv_mean_{metric_name}']:.4f} +/- "
        f"{cv_summary[f'cv_std_{metric_name}']:.4f}"
    )
for metric_name in ["f1", "macro_f1", "precision", "recall", "accuracy"]:
    print(
        f"CV TUNED_{metric_name.upper()}: "
        f"{cv_summary[f'cv_mean_tuned_{metric_name}']:.4f} +/- "
        f"{cv_summary[f'cv_std_tuned_{metric_name}']:.4f}"
    )
print(
    f"CV threshold ({optimize_metric}) mean={cv_summary['cv_mean_threshold']:.4f} "
    f"std={cv_summary['cv_std_threshold']:.4f} "
    f"selected(median)={cv_summary['cv_selected_threshold']:.4f}"
)
print(f"[TIMING] CV completed in {cv_elapsed:.2f} seconds ({cv_elapsed/60:.2f} minutes)")

# Log fold-level table and CV aggregate summary to W&B.
cv_table = wandb.Table(
    columns=[
        "fold",
        "roc_auc",
        "pr_auc",
        "f1",
        "macro_f1",
        "precision",
        "recall",
        "accuracy",
        "best_threshold",
        "best_threshold_metric",
        "tuned_f1",
        "tuned_macro_f1",
        "tuned_precision",
        "tuned_recall",
        "tuned_accuracy",
    ],
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
            float(m["best_threshold"]),
            float(m["best_threshold_metric"]),
            float(m["tuned_f1"]),
            float(m["tuned_macro_f1"]),
            float(m["tuned_precision"]),
            float(m["tuned_recall"]),
            float(m["tuned_accuracy"]),
        ]
        for m in cv_fold_metrics
    ],
)
wandb.log({"cv/folds_table": cv_table, **cv_summary})

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
y_pred_default = svm_pipeline.predict(X_holdout)
y_score = svm_pipeline.decision_function(X_holdout)
y_pred_tuned = predict_with_threshold(y_score, selected_threshold)

metrics_default = {
    "ROC-AUC": roc_auc_score(y_holdout, y_score),
    "PR-AUC": average_precision_score(y_holdout, y_score),
    "F1": f1_score(y_holdout, y_pred_default, zero_division=0),
    "Macro-F1": f1_score(y_holdout, y_pred_default, average="macro", zero_division=0),
    "Precision": precision_score(y_holdout, y_pred_default, zero_division=0),
    "Recall": recall_score(y_holdout, y_pred_default, zero_division=0),
    "Accuracy": accuracy_score(y_holdout, y_pred_default),
}

metrics_tuned = {
    "ROC-AUC": roc_auc_score(y_holdout, y_score),
    "PR-AUC": average_precision_score(y_holdout, y_score),
    "F1": f1_score(y_holdout, y_pred_tuned, zero_division=0),
    "Macro-F1": f1_score(y_holdout, y_pred_tuned, average="macro", zero_division=0),
    "Precision": precision_score(y_holdout, y_pred_tuned, zero_division=0),
    "Recall": recall_score(y_holdout, y_pred_tuned, zero_division=0),
    "Accuracy": accuracy_score(y_holdout, y_pred_tuned),
}

cm_default = confusion_matrix(y_holdout, y_pred_default)
cm_tuned = confusion_matrix(y_holdout, y_pred_tuned)
report = classification_report(y_holdout, y_pred_tuned, output_dict=True)
positive_rate_default = float((y_pred_default == 1).mean())
positive_rate_tuned = float((y_pred_tuned == 1).mean())
score_mean = float(y_score.mean())
score_std = float(y_score.std())

print(f"Use StandardScaler: {use_scaler}")
print(f"Selected decision threshold ({optimize_metric}): {selected_threshold:.4f}")
print("Default threshold metrics (decision_function >= 0):")
for name, value in metrics_default.items():
    print(f"  {name}: {value:.4f}")
print(f"  Predicted positive rate: {positive_rate_default:.4f}")

print("Tuned threshold metrics (CV-derived threshold):")
for name, value in metrics_tuned.items():
    print(f"  {name}: {value:.4f}")
print(f"  Predicted positive rate: {positive_rate_tuned:.4f}")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
print("[SECTION] Generating ROC and PR plots")
fpr, tpr, _ = roc_curve(y_holdout, y_score)
precision, recall, _ = precision_recall_curve(y_holdout, y_score)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, label=f"ROC-AUC = {metrics_tuned['ROC-AUC']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.7)
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].plot(recall, precision, label=f"PR-AUC = {metrics_tuned['PR-AUC']:.4f}")
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()

plt.tight_layout()
# plt.show()


# -----------------------------------------------------------------------------
# W&B logging
# -----------------------------------------------------------------------------
print("[SECTION] Logging metrics and artifacts to W&B")
wandb.log(
    {
        "holdout/threshold": selected_threshold,
        "holdout/default_roc_auc": metrics_default["ROC-AUC"],
        "holdout/default_pr_auc": metrics_default["PR-AUC"],
        "holdout/default_f1": metrics_default["F1"],
        "holdout/default_macro_f1": metrics_default["Macro-F1"],
        "holdout/default_precision": metrics_default["Precision"],
        "holdout/default_recall": metrics_default["Recall"],
        "holdout/default_accuracy": metrics_default["Accuracy"],
        "holdout/default_predicted_positive_rate": positive_rate_default,
        "holdout/tuned_roc_auc": metrics_tuned["ROC-AUC"],
        "holdout/tuned_pr_auc": metrics_tuned["PR-AUC"],
        "holdout/tuned_f1": metrics_tuned["F1"],
        "holdout/tuned_macro_f1": metrics_tuned["Macro-F1"],
        "holdout/tuned_precision": metrics_tuned["Precision"],
        "holdout/tuned_recall": metrics_tuned["Recall"],
        "holdout/tuned_accuracy": metrics_tuned["Accuracy"],
        "holdout/tuned_predicted_positive_rate": positive_rate_tuned,
        "roc_auc": metrics_tuned["ROC-AUC"],
        "pr_auc": metrics_tuned["PR-AUC"],
        "f1": metrics_tuned["F1"],
        "macro_f1": metrics_tuned["Macro-F1"],
        "precision": metrics_tuned["Precision"],
        "recall": metrics_tuned["Recall"],
        "accuracy": metrics_tuned["Accuracy"],
        "predicted_positive_rate": positive_rate_tuned,
        "decision_score_mean": score_mean,
        "decision_score_std": score_std,
        "holdout/default_tn": int(cm_default[0, 0]),
        "holdout/default_fp": int(cm_default[0, 1]),
        "holdout/default_fn": int(cm_default[1, 0]),
        "holdout/default_tp": int(cm_default[1, 1]),
        "tn": int(cm_tuned[0, 0]),
        "fp": int(cm_tuned[0, 1]),
        "fn": int(cm_tuned[1, 0]),
        "tp": int(cm_tuned[1, 1]),
    }
)

run.summary["macro_f1"] = metrics_tuned["Macro-F1"]
run.summary["holdout/macro_f1"] = metrics_tuned["Macro-F1"]
run.summary["holdout/default_macro_f1"] = metrics_default["Macro-F1"]
run.summary["holdout/selected_threshold"] = selected_threshold

wandb.log(
    {
        "confusion_matrix_default": wandb.plot.confusion_matrix(
            y_true=y_holdout.tolist(),
            preds=y_pred_default.tolist(),
            class_names=["Reject", "Accept"],
        ),
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_holdout.tolist(),
            preds=y_pred_tuned.tolist(),
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

script_elapsed = time.perf_counter() - script_start_time
print(f"[TIMING] Total script execution: {script_elapsed:.2f} seconds ({script_elapsed/60:.2f} minutes)")

print("[SECTION] Finishing W&B run")
run.finish()
