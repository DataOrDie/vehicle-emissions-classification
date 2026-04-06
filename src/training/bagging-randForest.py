from pathlib import Path
import sys
import importlib.util

import numpy as np
import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# W&B authentication
# -----------------------------------------------------------------------------
print("[SECTION] W&B authentication")
wandb.login()


# -----------------------------------------------------------------------------
# Project paths and imports
# -----------------------------------------------------------------------------
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
df_candidate = globals().get("df", None)
if not isinstance(df_candidate, pd.DataFrame):
    df = pd.read_csv(project_root / "data" / "train.csv")
else:
    df = df_candidate


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

approvaldate_option: str = "C"
approvalfy_option: str = "C"
franchise_option: str = "binary"
urbanrural_option: str = "onehot"
revlinecr_option: str = "C"
lowdoc_option: str = "C"
accept_option: str = ""

local_state: str = "IL"
citybank_option: str = "freq_bucket"
city_top_k: int = 120
bank_top_k: int = 80
city_min_count: int | None = None
bank_min_count: int | None = None
citybank_other_label: str = "OTHER"
citybank_suffix: str = "_bucket"
citybank_drop_original: bool = False

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


# -----------------------------------------------------------------------------
# Train/test split
# -----------------------------------------------------------------------------
print("[SECTION] Building train/holdout split")
target_col = "Accept"
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train set size: {X_train.shape[0]}")
print(f"Holdout set size: {X_holdout.shape[0]}")


# -----------------------------------------------------------------------------
# RandomForest bagging strategy
# -----------------------------------------------------------------------------
print("[SECTION] Configuring RandomForest bagging strategy")
random_state: int = 42
decision_threshold: float = 0.5
balance_strategy: str = "class_weight"

rf_params = {
    "n_estimators": 600,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
}


def build_rf_pipeline(random_state: int = 42, model_params: dict | None = None) -> Pipeline:
    if model_params is None:
        model_params = rf_params

    class_weight = "balanced_subsample" if balance_strategy == "class_weight" else None

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=model_params["n_estimators"],
                    criterion=model_params["criterion"],
                    max_depth=model_params["max_depth"],
                    min_samples_split=model_params["min_samples_split"],
                    min_samples_leaf=model_params["min_samples_leaf"],
                    max_features=model_params["max_features"],
                    bootstrap=model_params["bootstrap"],
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


# -----------------------------------------------------------------------------
# W&B run
# -----------------------------------------------------------------------------
print("[SECTION] Initializing W&B run")
model_name = "bagging-randforest"
create_kaggle_csv: bool = True

run = wandb.init(
    project="MS BAGGING - RANDOM FOREST",
    config={
        "model_name": model_name,
        "random_state": random_state,
        "decision_threshold": decision_threshold,
        "balance_strategy": balance_strategy,
        "rf_params": rf_params,
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
        "n_train_rows": int(X_train.shape[0]),
        "n_holdout_rows": int(X_holdout.shape[0]),
        "n_features": int(X_train.shape[1]),
    },
)


# -----------------------------------------------------------------------------
# Train and evaluate
# -----------------------------------------------------------------------------
print("[SECTION] Training RandomForest pipeline")
rf_pipeline = build_rf_pipeline(random_state=random_state, model_params=rf_params)
rf_pipeline.fit(X_train, y_train)

print("[SECTION] Running holdout predictions")
y_score = rf_pipeline.predict_proba(X_holdout)[:, 1]
y_pred = (y_score >= decision_threshold).astype(int)

metrics = {
    "roc_auc": roc_auc_score(y_holdout, y_score),
    "pr_auc": average_precision_score(y_holdout, y_score),
    "f1": f1_score(y_holdout, y_pred, zero_division=0),
    "macro_f1": f1_score(y_holdout, y_pred, average="macro", zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_holdout, y_pred),
    "precision": precision_score(y_holdout, y_pred, zero_division=0),
    "recall": recall_score(y_holdout, y_pred, zero_division=0),
    "accuracy": accuracy_score(y_holdout, y_pred),
}

cm = confusion_matrix(y_holdout, y_pred)
report = classification_report(y_holdout, y_pred, output_dict=True, zero_division=0)
positive_rate = float((y_pred == 1).mean())

print(f"Holdout ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Holdout PR-AUC: {metrics['pr_auc']:.4f}")
print(f"Holdout F1: {metrics['f1']:.4f}")
print(f"Holdout Macro-F1: {metrics['macro_f1']:.4f}")
print(f"Holdout Balanced-Accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"Holdout Precision: {metrics['precision']:.4f}")
print(f"Holdout Recall: {metrics['recall']:.4f}")
print(f"Holdout Accuracy: {metrics['accuracy']:.4f}")
print(f"Predicted positive rate: {positive_rate:.4f}")


# -----------------------------------------------------------------------------
# W&B logging
# -----------------------------------------------------------------------------
print("[SECTION] Logging to W&B")
wandb.log(
    {
        **metrics,
        "predicted_positive_rate": positive_rate,
        "decision_threshold": decision_threshold,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
)

run.summary["macro_f1"] = metrics["macro_f1"]
run.summary["holdout/macro_f1"] = metrics["macro_f1"]
run.summary["decision_threshold"] = decision_threshold

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

wandb.log(
    {
        "classification_report": report_table,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_holdout.tolist(),
            preds=y_pred.tolist(),
            class_names=["Reject", "Accept"],
        ),
    }
)

print("[SECTION] Finishing W&B run")
run.finish()


# Save model for reuse/submission
saved_paths = save_model(
    model_pipeline=rf_pipeline,
    preprocessing_options=options,
    feature_names=X_train.columns.tolist(),
    project_root=project_root,
    model_name=model_name,
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
        model_name=model_name,
        project_root=project_root,
        is_tree_model=True,
        verbose=False,
    )
    print(f"Kaggle submission generated: {submission_path}")
