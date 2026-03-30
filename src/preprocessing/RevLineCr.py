"""Preprocessing + Cleaning utilities for RevLineCr column.

This script includes:
1. Cleaning: grouping non-standard and missing values
2. Preprocessing: feature engineering options A and B
"""

from __future__ import annotations

import pandas as pd


DEFAULT_REVLINECR_OPTION = "A"

def clean_revlinecr(series: pd.Series) -> pd.Series:
    """
    Clean RevLineCr values:
    - Standard values: Y, N
    - Non-standard → UNKNOWN
    - Missing → MISSING
    """

    cleaned = series.astype("string").str.strip()

    cleaned = cleaned.fillna("MISSING")

    valid_values = ["Y", "N", "MISSING"]

    cleaned = cleaned.apply(
        lambda x: x if x in valid_values else "UNKNOWN"
    )

    return cleaned


def preprocess_revlinecr_option_a(
    df: pd.DataFrame,
    source_col: str = "RevLineCr",
) -> pd.DataFrame:
    """Option A: clean + indicators + one-hot"""

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()

    # CLEANING
    clean_col = clean_revlinecr(result[source_col])
    result["RevLineCr_clean"] = clean_col

    # INDICATORS
    result["revlinecr_is_nonstandard"] = (clean_col == "UNKNOWN").astype(int)
    result["revlinecr_is_missing"] = (clean_col == "MISSING").astype(int)

    # ONE HOT (y elimina la columna original limpia)
    result = pd.get_dummies(result, columns=["RevLineCr_clean"], prefix="revlinecr")

    return result


def preprocess_revlinecr_option_b(
    df: pd.DataFrame,
    source_col: str = "RevLineCr",
) -> pd.DataFrame:
    """Option B: clean + one-hot only"""

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()

    clean_col = clean_revlinecr(result[source_col])
    result["RevLineCr_clean"] = clean_col

    result = pd.get_dummies(result, columns=["RevLineCr_clean"], prefix="revlinecr")

    return result

def preprocess_revlinecr(
    df: pd.DataFrame,
    option: str = DEFAULT_REVLINECR_OPTION,
) -> pd.DataFrame:

    if option.upper() == "A":
        return preprocess_revlinecr_option_a(df)

    if option.upper() == "B":
        return preprocess_revlinecr_option_b(df)

    raise ValueError("option must be 'A' or 'B'")
