# Loan Approval Challenge - One-Page Plan

## Objective

Build the best possible classifier for Kaggle to decide whether a loan for an SME in Illinois should be approved or denied.

- Guiding question: As a bank representative, should I grant this loan? Why?
- Success metric: maximize Kaggle score while keeping a reproducible workflow and clear final report.

## Core Strategy

- Prioritize feature engineering over constant model switching.
- Keep one shared data pipeline for all teams.
- Compare models under the same validation protocol.
- Treat the secret algorithm as the main optimization track.

## Shared Pipeline

```text
raw data -> cleaning -> feature engineering -> final dataset -> model -> Kaggle submission
```

All model families must use the same base feature set before branch-specific experiments.

## Workstreams

1. Geometric models

- SVM
- KNN
- Logistic Regression
- Linear classifier

2. Tree-based models

- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

3. Secret algorithm (all team)

- Baseline implementation
- Hyperparameter tuning
- Feature tuning
- Optional ensembling

## Timeline Snapshot

- Phase 0: project setup and baseline submission
- Phase 1: domain understanding + hypotheses
- Phase 2: EDA and feature decisions
- Phase 3: feature engineering sprint
- Phase 4: model development by subteams
- Phase 5: optimization and ensembles
- Phase 6: Kaggle final submissions
- Phase 7: final report and delivery

## Deliverables by End

- Reproducible training pipeline
- Valid Kaggle submission workflow
- Model comparison table (local + Kaggle scores)
- Final optimized solution (especially secret algorithm)
- Final report: coordination, issues, improvements, and optimization evidence

## Team Operating Rules

- Track every experiment (MLflow/W&B/CSV logs).
- Use short, frequent sync meetings.
- Maintain one live table: model, features, local score, Kaggle score.
- Keep documentation updated in parallel with experiments.
