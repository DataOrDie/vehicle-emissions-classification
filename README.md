# CDAW Loan Approval Prediction in Illinois

This repository contains the team workflow, code, models, notebooks, and submissions for the CDAW loan approval challenge.

## Challenge summary

The competition goal is to predict whether a loan application should be accepted or rejected for SME companies.

Core decision question:
As a bank representative, should we grant a loan to company X, and why?

The challenge setup (from professor notes + Kaggle brief):

- Teams build and compare machine learning models for loan risk classification.
- Each team must use at least:
  - One geometric algorithm.
  - One decision tree model.
  - The team-assigned secret algorithm.
- Kaggle is used for iterative submissions and leaderboard feedback.
- Main competition metric is Macro F1.
- Final ranking uses selected best submissions before deadline.

## Course deliverables (from professor notes)

Moodle delivery includes:

- Final notebook(s) and/or source code used to generate the final submission.
- Team report in PDF:
  - Common section (methodology, coordination, model-improvement techniques, problems and solutions).
  - Strong focus on model optimization details, especially for the assigned secret algorithm.
  - Individual section per team member with contributions and peer-evaluation table.
  - Notebook authorship table mapping notebooks to members.
- Packaged delivery (PDF presentation/report + memory + code) in ZIP format.

## Data and evaluation

Dataset files:

- Training: [data/train.csv](data/train.csv)
- Test (no labels): [data/test_nolabel.csv](data/test_nolabel.csv)
- Submission template: [data/sample_submission.csv](data/sample_submission.csv)

Evaluation notes:

- Kaggle evaluates on hidden labels / hidden split.
- Preprocessing must be learned on training data and applied consistently to test data.
- Macro F1 is the main selection metric, so class balance behavior matters.

## Current repository organization

Top-level:

- [README.md](README.md)
- [requirements.txt](requirements.txt)
- [Planning.md](Planning.md)
- [about-data.md](about-data.md)
- [feature-engineering.md](feature-engineering.md)
- [Proffessor Notes and indications.md](Proffessor%20Notes%20and%20indications.md)
- [data](data)
- [src](src)
- [notebooks](notebooks)
- [models](models)
- [submissions](submissions)
- [experiments](experiments)
- [wandb](wandb)

Source code:

- Preprocessing modules: [src/preprocessing](src/preprocessing)
  - Includes feature transforms and one-step pipeline pieces such as noemp, createjob, retainedjob, revlinecr, lowdoc, date and amount processing.
- Training scripts: [src/training](src/training)
  - Geometric family:
    - [src/training/geom-svm.py](src/training/geom-svm.py)
    - [src/training/geom-svm-kernel-rbf.py](src/training/geom-svm-kernel-rbf.py)
    - [src/training/geom-svm-kernel-poly.py](src/training/geom-svm-kernel-poly.py)
    - [src/training/geom-svm-kernel-sigmoid.py](src/training/geom-svm-kernel-sigmoid.py)
    - [src/training/geom-svm-thresholdTuneClassifier.py](src/training/geom-svm-thresholdTuneClassifier.py)
    - [src/training/geom-svm-thresholdTuneClassifier-csweep.py](src/training/geom-svm-thresholdTuneClassifier-csweep.py)
  - Tree / bagging family:
    - [src/training/bagging-extratrees.py](src/training/bagging-extratrees.py)
    - [src/training/bagging-randomForest_MS.py](src/training/bagging-randomForest_MS.py)
- Submission helpers: [src/submit](src/submit)
  - [src/submit/kaggle.py](src/submit/kaggle.py)
  - [src/submit/save_model.py](src/submit/save_model.py)
  - [src/submit/SUBMISSION_GUIDE.md](src/submit/SUBMISSION_GUIDE.md)

Experiment artifacts:

- Saved model bundles under [models](models), grouped by model name.
- CSV outputs under [submissions](submissions).
- W&B run logs and artifacts under [wandb](wandb) and [src/training/wandb](src/training/wandb).

## Environment setup

1. Create / activate your Python environment (example: conda env sitc).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training examples

Run from repository root.

Geometric SVM baseline:

```bash
conda run --no-capture-output -n sitc python -u ./src/training/geom-svm.py
```

Threshold tuning + C sweep (recent workflow):

```bash
conda run --no-capture-output -n sitc python -u ./src/training/geom-svm-thresholdTuneClassifier-csweep.py
```

Bagging ExtraTrees:

```bash
conda run --no-capture-output -n sitc python -u ./src/training/bagging-extratrees.py
```

## Generating Kaggle submissions

After training, generate submission with the saved model artifacts:

```bash
python src/submit/kaggle.py <model_name>
```

For tree-specific feature path:

```bash
python src/submit/kaggle.py <model_name> --is-tree-model
```

Examples:

```bash
python src/submit/kaggle.py geom-svm
python src/submit/kaggle.py bagging-extratrees --is-tree-model
```

Output file format:

- Path: [submissions](submissions)
- Naming: submission-<model_name>.csv
- Columns: id, Accept

## Kaggle competition reference

Competition slug used in team notes and scripts:

- cdaw-loan-approval-prediction-in-illinois

CLI example:

```bash
kaggle competitions submit -c cdaw-loan-approval-prediction-in-illinois -f submissions/submission-geom-svm.csv -m "Geom-SVM model submission"
```

## Practical checklist

- Keep preprocessing and feature ordering identical between train and test.
- Validate with stratified folds and monitor Macro F1.
- Save model + options + feature names before submission generation.
- Track leaderboard changes and annotate what changed between submissions.
