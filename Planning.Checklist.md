# Loan Approval Challenge - Execution Checklist

Use this checklist as a day-to-day control board.

## Phase 0 - Setup

- [ ] Create/verify repository structure
- [ ] Define coding and experiment naming conventions
- [ ] Build baseline data loading script
- [ ] Implement basic cleaning pipeline
- [ ] Implement train/validation split
- [ ] Build first baseline model
- [ ] Generate valid Kaggle CSV format
- [ ] Submit first baseline to Kaggle
- [ ] Register baseline score (local + Kaggle)

## Phase 1 - Domain Understanding

- [ ] Review all variables and data types
- [ ] Identify missing values and suspicious fields
- [ ] Draft initial risk hypotheses
- [ ] List potential noisy or leakage-prone variables
- [ ] Write initial notebook documentation
- [ ] Team brainstorming meeting completed

## Phase 2 - EDA

- [ ] Compute target distribution
- [ ] Compute feature summary statistics
- [ ] Run correlation analysis
- [ ] Build key visualizations (histograms, pairplots, PCA)
- [ ] Identify outliers and collinearity
- [ ] Document EDA conclusions
- [ ] Freeze common feature set v1

## Phase 3 - Feature Engineering

- [ ] Apply numeric transformations (scaling/log)
- [ ] Build ratio and interaction features
- [ ] Test dimensionality reduction (PCA/selection)
- [ ] Evaluate each feature batch with same validation setup
- [ ] Keep only features that improve performance
- [ ] Freeze common feature set v2

## Phase 4 - Model Development

### Geometric Team

- [ ] Train Logistic Regression baseline
- [ ] Train KNN variants
- [ ] Train SVM (linear + kernel options)
- [ ] Tune key hyperparameters
- [ ] Record CV metrics and best configs

### Tree Team

- [ ] Train Decision Tree baseline
- [ ] Train Random Forest variants
- [ ] Train Gradient Boosting/XGBoost variants
- [ ] Tune depth, estimators, subsampling
- [ ] Record CV metrics and best configs

### Secret Algorithm (All Team)

- [ ] Implement baseline version
- [ ] Define tuning search space
- [ ] Run hyperparameter tuning
- [ ] Run feature tuning
- [ ] Evaluate ensemble with best candidates
- [ ] Document all gains and trade-offs

## Phase 5 - Global Optimization

- [ ] Compare top candidates on same validation protocol
- [ ] Run focused hyperparameter search (Grid/Random/Bayesian)
- [ ] Re-check class imbalance handling
- [ ] Validate calibration/threshold decisions if needed
- [ ] Build and test final ensemble candidates
- [ ] Select final model by robust performance

## Phase 6 - Kaggle Competition

- [ ] Retrain final model on agreed training setup
- [ ] Generate final predictions on test set
- [ ] Validate submission schema before upload
- [ ] Upload and log each Kaggle submission
- [ ] Track leaderboard changes and notes

## Phase 7 - Final Documentation

- [ ] Team coordination section complete
- [ ] Problems encountered section complete
- [ ] Improvement techniques section complete
- [ ] Secret algorithm optimization section complete (strongest section)
- [ ] Include experiment evidence (tables/figures)
- [ ] Final report reviewed and delivered

## Tracking Tables

### Model Tracking

| Date | Team | Model | Feature Set | Local Metric | Kaggle Score | Notes |
| ---- | ---- | ----- | ----------- | -----------: | -----------: | ----- |

### Submission Log

| Submission # | Date | Model | Feature Set | Public Score | Change vs Previous | Notes |
| -----------: | ---- | ----- | ----------- | -----------: | -----------------: | ----- |

### Risks and Blockers

| Date | Risk/Blocker | Impact | Owner | Mitigation | Status |
| ---- | ------------ | ------ | ----- | ---------- | ------ |
