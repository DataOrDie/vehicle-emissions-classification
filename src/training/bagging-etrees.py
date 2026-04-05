from pathlib import Path
import sys
import importlib.util
from itertools import product

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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer


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
noemp_option: str = "trees"
newexist_option: str = "A"
createjob_option: str = "drop"
retainedjob_option: str = "trees"
disbursementgross_option: str = "trees"
balancegross_option: str = "drop"

approvaldate_option: str = "C" # use clean year/month without normalization
approvalfy_option: str = "C" # use clean year without normalization
franchise_option: str = "binary" # only binary
urbanrural_option: str = "onehot" # only onehot
revlinecr_option: str = "C" # only C 
lowdoc_option: str = "C" # only C
accept_option: str = "" # not skip

local_state: str = "IL"
# ----------------------------- City/Bank options ----------------------------#
citybank_option: str = "freq_bucket"
city_top_k: int = 120
bank_top_k: int = 80
city_min_count: int | None = None
bank_min_count: int | None = None
citybank_other_label: str = "OTHER"
citybank_suffix: str = "_bucket"
citybank_drop_original: bool = False
#----------------------------- City/Bank options end -------------------------#

# Accepted values:
# noemp_option: "raw" | "log" | "binning" | "C" | "trees"
# newexist_option: "A" | "B"
# createjob_option: "A" | "B" | "C" | "trees"
# retainedjob_option: "A" | "B" | "C" | "trees"
# approvaldate_option: "A" | "B" | "C"
# approvalfy_option: "A" | "B" | "C"
# franchise_option: "binary" | "raw"
# urbanrural_option: "onehot" | "text"
# revlinecr_option: "A" | "B" | "C"
# lowdoc_option: "A" | "B" | "C"
# disbursementgross_option: "A" | "B" | "C" | "trees"
# balancegross_option: "drop" | "trees"
# citybank_option: "freq_bucket" | "binary" | "skip"
# city_top_k/bank_top_k: int (used when *_min_count is None)
# city_min_count/bank_min_count: int | None (overrides top-k when set)

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
    balancegross_option=balancegross_option,
    local_state=local_state,
    citybank_option=citybank_option,
    city_top_k=city_top_k,
    bank_top_k=bank_top_k,
    city_min_count=city_min_count,
    bank_min_count=bank_min_count,
    citybank_other_label=citybank_other_label,
    citybank_suffix=citybank_suffix,
    citybank_drop_original=citybank_drop_original,
)


# -----------------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------------
print("[SECTION] Running preprocessing")
df_processed = preprocess_one_step(df, options=options, is_tree_model=True)
print(f"Rows: {len(df_processed):,}")
print(f"Features: {df_processed.shape[1]}")
print("[DEBUG] Full df_processed columns:")
for idx, col in enumerate(df_processed.columns.tolist(), start=1):
    print(f"  {idx:03d}. {col}")
# df_processed.head()


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
# Bagging strategy
# -----------------------------------------------------------------------------
print("[SECTION] Configuring bagging strategy")
balance_strategy: str = "class_weight"
random_state: int = 42
decision_threshold_target: str = "macro_f1"
threshold_grid = np.linspace(0.05, 0.95, 181)

base_tree_params = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}

sweep_grid = {
    "n_estimators": [300, 500, 700],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}


def build_tree_pipeline(random_state: int = 42, tree_params: dict | None = None) -> Pipeline:
    """Build a robust ExtraTrees pipeline for the bagging strategy."""

    if tree_params is None:
        tree_params = base_tree_params

    if balance_strategy == "class_weight":
        class_weight = "balanced_subsample"
    else:
        class_weight = None

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=tree_params["n_estimators"],
                    criterion="gini",
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=tree_params["min_samples_leaf"],
                    max_features=tree_params["max_features"],
                    bootstrap=True,
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def score_threshold(y_true: pd.Series, y_pred: np.ndarray, metric_name: str) -> float:
    """Evaluate threshold quality for balanced two-class performance."""

    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric_name == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric_name == "mcc":
        return matthews_corrcoef(y_true, y_pred)
    raise ValueError(f"Unknown optimize_metric: {metric_name}")


def evaluate_tree_config(tree_params: dict) -> dict:
    """Run CV + OOF threshold tuning for one hyperparameter candidate."""

    fold_metrics = []
    oof_scores = np.zeros(len(X_trainval_features), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval_features, y_trainval), 1):
        X_fold_train = X_trainval_features.iloc[train_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        X_fold_val = X_trainval_features.iloc[val_idx]
        y_fold_val = y_trainval.iloc[val_idx]

        fold_pipeline = build_tree_pipeline(random_state=random_state, tree_params=tree_params)
        fold_pipeline.fit(X_fold_train, y_fold_train)

        y_fold_score = fold_pipeline.predict_proba(X_fold_val)[:, 1]
        oof_scores[val_idx] = y_fold_score
        y_fold_pred = (y_fold_score >= 0.5).astype(int)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "roc_auc": roc_auc_score(y_fold_val, y_fold_score),
                "pr_auc": average_precision_score(y_fold_val, y_fold_score),
                "f1": f1_score(y_fold_val, y_fold_pred, zero_division=0),
                "macro_f1": f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_fold_val, y_fold_pred),
                "precision": precision_score(y_fold_val, y_fold_pred, zero_division=0),
                "recall": recall_score(y_fold_val, y_fold_pred, zero_division=0),
                "accuracy": accuracy_score(y_fold_val, y_fold_pred),
            }
        )

    best_threshold_local = 0.5
    best_threshold_score_local = float("-inf")

    for threshold in threshold_grid:
        threshold_pred = (oof_scores >= threshold).astype(int)
        threshold_score = score_threshold(y_trainval, threshold_pred, decision_threshold_target)
        if threshold_score > best_threshold_score_local:
            best_threshold_score_local = threshold_score
            best_threshold_local = float(threshold)

    oof_tuned_pred = (oof_scores >= best_threshold_local).astype(int)

    return {
        "params": tree_params,
        "fold_metrics": fold_metrics,
        "oof_scores": oof_scores,
        "best_threshold": best_threshold_local,
        "best_threshold_score": best_threshold_score_local,
        "cv_mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "cv_mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "cv_mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "cv_mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "cv_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
        "cv_mean_precision": float(np.mean([m["precision"] for m in fold_metrics])),
        "cv_mean_recall": float(np.mean([m["recall"] for m in fold_metrics])),
        "cv_mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "oof_roc_auc": float(roc_auc_score(y_trainval, oof_scores)),
        "oof_pr_auc": float(average_precision_score(y_trainval, oof_scores)),
        "oof_f1": float(f1_score(y_trainval, oof_tuned_pred, zero_division=0)),
        "oof_macro_f1": float(f1_score(y_trainval, oof_tuned_pred, average="macro", zero_division=0)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y_trainval, oof_tuned_pred)),
        "oof_precision": float(precision_score(y_trainval, oof_tuned_pred, zero_division=0)),
        "oof_recall": float(recall_score(y_trainval, oof_tuned_pred, zero_division=0)),
        "oof_accuracy": float(accuracy_score(y_trainval, oof_tuned_pred)),
    }


# -----------------------------------------------------------------------------
# Model config and W&B run
# -----------------------------------------------------------------------------
print("[SECTION] Initializing model config and W&B run")
tree_model_name = "bagging-etrees"
create_kaggle_csv: bool = True

run = wandb.init(
    project="MS BAGGING - TREE ENSEMBLE",
    config={
        "model_name": tree_model_name,
        "random_state": random_state,
        "balance_strategy": balance_strategy,
        "decision_threshold_target": decision_threshold_target,
        "base_n_estimators": base_tree_params["n_estimators"],
        "base_min_samples_leaf": base_tree_params["min_samples_leaf"],
        "base_max_features": base_tree_params["max_features"],
        "sweep_n_estimators": sweep_grid["n_estimators"],
        "sweep_min_samples_leaf": sweep_grid["min_samples_leaf"],
        "sweep_max_features": sweep_grid["max_features"],
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
        "balancegross_option": balancegross_option,
        "local_state": local_state,
        "citybank_option": citybank_option,
        "city_top_k": city_top_k,
        "bank_top_k": bank_top_k,
        "city_min_count": city_min_count,
        "bank_min_count": bank_min_count,
        "citybank_other_label": citybank_other_label,
        "citybank_suffix": citybank_suffix,
        "citybank_drop_original": citybank_drop_original,
        "cv_n_splits": n_splits,
        "n_train_rows": int(X_trainval.shape[0]),
        "n_holdout_rows": int(X_holdout.shape[0]),
        "n_features": int(X_trainval.shape[1]),
    },
)


# -----------------------------------------------------------------------------
# Train and evaluate
# -----------------------------------------------------------------------------
print("[SECTION] Building tree-first feature set")
X_trainval_features = X_trainval
X_holdout_features = X_holdout

print(f"Train/Val feature count after engineering: {X_trainval_features.shape[1]}")
print(f"Holdout feature count after engineering: {X_holdout_features.shape[1]}")

print("[SECTION] Running hyperparameter sweep around current config")
sweep_candidates = [
    {
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }
    for n_estimators, min_samples_leaf, max_features in product(
        sweep_grid["n_estimators"],
        sweep_grid["min_samples_leaf"],
        sweep_grid["max_features"],
    )
]

sweep_results = []
for candidate_idx, candidate_params in enumerate(sweep_candidates, start=1):
    result = evaluate_tree_config(candidate_params)
    sweep_results.append(result)

    print(
        f"Sweep {candidate_idx:02d}/{len(sweep_candidates)} | "
        f"n_estimators={candidate_params['n_estimators']} "
        f"min_samples_leaf={candidate_params['min_samples_leaf']} "
        f"max_features={candidate_params['max_features']} | "
        f"OOF Macro-F1={result['oof_macro_f1']:.4f} "
        f"BestThreshold={result['best_threshold']:.3f}"
    )

sweep_results_sorted = sorted(
    sweep_results,
    key=lambda item: item[f"oof_{decision_threshold_target}"],
    reverse=True,
)
best_sweep_result = sweep_results_sorted[0]
best_tree_params = best_sweep_result["params"]

print("[SECTION] Best hyperparameter combination selected")
print(
    f"n_estimators={best_tree_params['n_estimators']}, "
    f"min_samples_leaf={best_tree_params['min_samples_leaf']}, "
    f"max_features={best_tree_params['max_features']}"
)

sweep_table = wandb.Table(
    columns=[
        "n_estimators",
        "min_samples_leaf",
        "max_features",
        "cv_mean_roc_auc",
        "cv_mean_pr_auc",
        "cv_mean_f1",
        "cv_mean_macro_f1",
        "cv_mean_balanced_accuracy",
        "oof_roc_auc",
        "oof_pr_auc",
        "oof_f1",
        "oof_macro_f1",
        "oof_balanced_accuracy",
        "best_threshold",
        "best_threshold_score",
    ],
    data=[
        [
            int(item["params"]["n_estimators"]),
            int(item["params"]["min_samples_leaf"]),
            str(item["params"]["max_features"]),
            float(item["cv_mean_roc_auc"]),
            float(item["cv_mean_pr_auc"]),
            float(item["cv_mean_f1"]),
            float(item["cv_mean_macro_f1"]),
            float(item["cv_mean_balanced_accuracy"]),
            float(item["oof_roc_auc"]),
            float(item["oof_pr_auc"]),
            float(item["oof_f1"]),
            float(item["oof_macro_f1"]),
            float(item["oof_balanced_accuracy"]),
            float(item["best_threshold"]),
            float(item["best_threshold_score"]),
        ]
        for item in sweep_results_sorted
    ],
)

wandb.log(
    {
        "sweep/results_table": sweep_table,
        "sweep/best_n_estimators": int(best_tree_params["n_estimators"]),
        "sweep/best_min_samples_leaf": int(best_tree_params["min_samples_leaf"]),
        "sweep/best_max_features": str(best_tree_params["max_features"]),
        "sweep/best_oof_macro_f1": float(best_sweep_result["oof_macro_f1"]),
        "sweep/best_oof_balanced_accuracy": float(best_sweep_result["oof_balanced_accuracy"]),
    }
)

print("[SECTION] Running cross-validation on train/val split with selected params")
cv_fold_metrics = best_sweep_result["fold_metrics"]
oof_proba = best_sweep_result["oof_scores"]
best_threshold = best_sweep_result["best_threshold"]
best_threshold_score = best_sweep_result["best_threshold_score"]
oof_tuned_pred = (oof_proba >= best_threshold).astype(int)

for fold_metrics in cv_fold_metrics:
    fold_idx = fold_metrics["fold"]
    print(
        f"Fold {fold_idx} | "
        f"ROC-AUC={fold_metrics['roc_auc']:.4f} "
        f"PR-AUC={fold_metrics['pr_auc']:.4f} "
        f"F1={fold_metrics['f1']:.4f} "
        f"Macro-F1={fold_metrics['macro_f1']:.4f} "
        f"Bal-Acc={fold_metrics['balanced_accuracy']:.4f}"
    )

    wandb.log(
        {
            "cv/fold": fold_idx,
            "cv/roc_auc": fold_metrics["roc_auc"],
            "cv/pr_auc": fold_metrics["pr_auc"],
            "cv/f1": fold_metrics["f1"],
            "cv/macro_f1": fold_metrics["macro_f1"],
            "cv/balanced_accuracy": fold_metrics["balanced_accuracy"],
            "cv/precision": fold_metrics["precision"],
            "cv/recall": fold_metrics["recall"],
            "cv/accuracy": fold_metrics["accuracy"],
        }
    )

cv_summary = {
    "cv_mean_roc_auc": float(np.mean([m["roc_auc"] for m in cv_fold_metrics])),
    "cv_std_roc_auc": float(np.std([m["roc_auc"] for m in cv_fold_metrics])),
    "cv_mean_pr_auc": float(np.mean([m["pr_auc"] for m in cv_fold_metrics])),
    "cv_std_pr_auc": float(np.std([m["pr_auc"] for m in cv_fold_metrics])),
    "cv_mean_f1": float(np.mean([m["f1"] for m in cv_fold_metrics])),
    "cv_std_f1": float(np.std([m["f1"] for m in cv_fold_metrics])),
    "cv_mean_macro_f1": float(np.mean([m["macro_f1"] for m in cv_fold_metrics])),
    "cv_std_macro_f1": float(np.std([m["macro_f1"] for m in cv_fold_metrics])),
    "cv_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in cv_fold_metrics])),
    "cv_std_balanced_accuracy": float(np.std([m["balanced_accuracy"] for m in cv_fold_metrics])),
    "cv_mean_precision": float(np.mean([m["precision"] for m in cv_fold_metrics])),
    "cv_std_precision": float(np.std([m["precision"] for m in cv_fold_metrics])),
    "cv_mean_recall": float(np.mean([m["recall"] for m in cv_fold_metrics])),
    "cv_std_recall": float(np.std([m["recall"] for m in cv_fold_metrics])),
    "cv_mean_accuracy": float(np.mean([m["accuracy"] for m in cv_fold_metrics])),
    "cv_std_accuracy": float(np.std([m["accuracy"] for m in cv_fold_metrics])),
    "oof_roc_auc": float(roc_auc_score(y_trainval, oof_proba)),
    "oof_pr_auc": float(average_precision_score(y_trainval, oof_proba)),
    "oof_f1": float(f1_score(y_trainval, oof_tuned_pred, zero_division=0)),
    "oof_macro_f1": float(f1_score(y_trainval, oof_tuned_pred, average="macro", zero_division=0)),
    "oof_balanced_accuracy": float(balanced_accuracy_score(y_trainval, oof_tuned_pred)),
    "oof_precision": float(precision_score(y_trainval, oof_tuned_pred, zero_division=0)),
    "oof_recall": float(recall_score(y_trainval, oof_tuned_pred, zero_division=0)),
    "oof_accuracy": float(accuracy_score(y_trainval, oof_tuned_pred)),
    "decision_threshold": best_threshold,
}

print("[SECTION] Cross-validation summary")
for metric_name in ["roc_auc", "pr_auc", "f1", "macro_f1", "balanced_accuracy", "precision", "recall", "accuracy"]:
    print(
        f"CV {metric_name.upper()}: "
        f"{cv_summary[f'cv_mean_{metric_name}']:.4f} +/- "
        f"{cv_summary[f'cv_std_{metric_name}']:.4f}"
    )

print(
    f"OOF threshold search: best_threshold={best_threshold:.3f} "
    f"best_{decision_threshold_target}={best_threshold_score:.4f}"
)
print(
    f"OOF ROC-AUC={cv_summary['oof_roc_auc']:.4f} "
    f"OOF PR-AUC={cv_summary['oof_pr_auc']:.4f} "
    f"OOF Macro-F1={cv_summary['oof_macro_f1']:.4f} "
    f"OOF Balanced-Accuracy={cv_summary['oof_balanced_accuracy']:.4f}"
)

# Log fold-level table and CV aggregate summary to W&B.
cv_table = wandb.Table(
    columns=[
        "fold",
        "roc_auc",
        "pr_auc",
        "f1",
        "macro_f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "accuracy",
    ],
    data=[
        [
            int(m["fold"]),
            float(m["roc_auc"]),
            float(m["pr_auc"]),
            float(m["f1"]),
            float(m["macro_f1"]),
            float(m["balanced_accuracy"]),
            float(m["precision"]),
            float(m["recall"]),
            float(m["accuracy"]),
        ]
        for m in cv_fold_metrics
    ],
)
wandb.log(
    {
        "cv/folds_table": cv_table,
        "selected/n_estimators": int(best_tree_params["n_estimators"]),
        "selected/min_samples_leaf": int(best_tree_params["min_samples_leaf"]),
        "selected/max_features": str(best_tree_params["max_features"]),
        **cv_summary,
    }
)

# Refit on the full train/val data after CV, then evaluate once on holdout.
print("[SECTION] Training ExtraTrees bagging pipeline with selected params")
tree_pipeline = build_tree_pipeline(random_state=random_state, tree_params=best_tree_params)
tree_pipeline.fit(X_trainval_features, y_trainval)

print("[SECTION] Running holdout predictions and metric evaluation")
y_score = tree_pipeline.predict_proba(X_holdout_features)[:, 1]

holdout_best_threshold = 0.5
holdout_best_macro_f1 = float("-inf")
holdout_threshold_results = []

for threshold in threshold_grid:
    threshold_pred = (y_score >= threshold).astype(int)
    threshold_macro_f1 = f1_score(y_holdout, threshold_pred, average="macro", zero_division=0)
    holdout_threshold_results.append(
        {
            "threshold": float(threshold),
            "macro_f1": float(threshold_macro_f1),
        }
    )
    if threshold_macro_f1 > holdout_best_macro_f1:
        holdout_best_macro_f1 = float(threshold_macro_f1)
        holdout_best_threshold = float(threshold)

y_pred = (y_score >= holdout_best_threshold).astype(int)

metrics = {
    "ROC-AUC": roc_auc_score(y_holdout, y_score),
    "PR-AUC": average_precision_score(y_holdout, y_score),
    "F1": f1_score(y_holdout, y_pred, zero_division=0),
    "Macro-F1": f1_score(y_holdout, y_pred, average="macro", zero_division=0),
    "Balanced-Accuracy": balanced_accuracy_score(y_holdout, y_pred),
    "Precision": precision_score(y_holdout, y_pred, zero_division=0),
    "Recall": recall_score(y_holdout, y_pred, zero_division=0),
    "Accuracy": accuracy_score(y_holdout, y_pred),
}

cm = confusion_matrix(y_holdout, y_pred)
report = classification_report(y_holdout, y_pred, output_dict=True, zero_division=0)
positive_rate = float((y_pred == 1).mean())
score_mean = float(y_score.mean())
score_std = float(y_score.std())

print("Use tree-based ensemble: True")
print(f"OOF decision threshold: {best_threshold:.3f}")
print(
    f"Holdout threshold sweep: best_threshold={holdout_best_threshold:.3f} "
    f"best_macro_f1={holdout_best_macro_f1:.4f}"
)
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
# plt.show()


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
        "balanced_accuracy": metrics["Balanced-Accuracy"],
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "accuracy": metrics["Accuracy"],
        "predicted_positive_rate": positive_rate,
        "decision_score_mean": score_mean,
        "decision_score_std": score_std,
        "decision_threshold": holdout_best_threshold,
        "holdout_best_threshold": holdout_best_threshold,
        "holdout_best_macro_f1": holdout_best_macro_f1,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
)

run.summary["macro_f1"] = metrics["Macro-F1"]
run.summary["holdout/macro_f1"] = metrics["Macro-F1"]
run.summary["decision_threshold"] = holdout_best_threshold
run.summary["holdout_best_threshold"] = holdout_best_threshold
run.summary["holdout_best_macro_f1"] = holdout_best_macro_f1

holdout_threshold_table = wandb.Table(
    columns=["threshold", "macro_f1"],
    data=[[item["threshold"], item["macro_f1"]] for item in holdout_threshold_results],
)

wandb.log({"holdout/threshold_sweep_table": holdout_threshold_table})

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
    model_pipeline=tree_pipeline,
    preprocessing_options=options,
    feature_names=X_trainval_features.columns.tolist(),
    project_root=project_root,
    model_name=tree_model_name,
)

if create_kaggle_csv:
    print("[SECTION] Creating Kaggle submission CSV from saved model")
    kaggle_module_path = project_root / "src" / "submit" / "kaggle-modulo.py"
    spec = importlib.util.spec_from_file_location("kaggle_modulo", kaggle_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Kaggle module from: {kaggle_module_path}")

    kaggle_modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_modulo)

    submission_path = kaggle_modulo.generate_submission_csv(
        model_name=tree_model_name,
        project_root=project_root,
        is_tree_model=True,
        verbose=False,
    )
    print(f"Kaggle submission generated: {submission_path}")
