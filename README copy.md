# Loan Approval Risk Classification

## Repository Description

This repository documents an automatic learning competition where students apply techniques learned in class to build the best-performing model.

The challenge provides a dataset of small and medium-sized enterprises (SMEs) that have applied for a loan. The objective is to train a classifier that predicts whether a loan should be approved or denied.

Students assume the role of a bank representative and answer the core decision question:

**Should I grant a loan to a particular small business (Company X)? Why or why not?**

This decision is made by assessing the risk profile of each loan application through data preprocessing, feature engineering, model training, and evaluation.

## Challenge Brief

These notes are working assumptions extracted from the professor's instructions and the Kaggle brief. They are intentionally exploratory and should be treated as hypotheses while analyzing the dataset.

- The task is loan approval classification, not a generic classification benchmark.
- Each team must test at least one geometric algorithm, one decision tree, and the assigned secret algorithm.
- The final report should emphasize modeling choices, improvement strategies, and validation results.
- Generic theory should be kept brief; the focus should be on what was tried, what worked, and why.

## Training and Evaluation

The dataset is split as follows:

- Training set: [data/train.csv](data/train.csv)
- Unlabeled test set: [data/test_nolabel.csv](data/test_nolabel.csv)

Evaluation on Kaggle uses a hidden subset of the test data. The exact evaluation split is not revealed, so preprocessing must be learned on the training set and then applied consistently to the test/submission data.

The competition metric is Macro F1-score. That means model selection should be based on validation performance that reflects both classes, not on accuracy alone.

Practical implication:

- Fit imputers, encoders, scalers, and feature selectors only on the training folds.
- Apply the same transformations to the unlabeled test data before generating the CSV submission.
- Keep any city-specific or geography-based encoding stable for unseen categories in the hidden evaluation set.

## Submission Notes

- Final Kaggle predictions must be written to a CSV file matching the provided sample submission format.
- The final Moodle deliverable should include the PDF report, the code or notebooks used for the submission, and any supporting artifacts required by the class instructions.