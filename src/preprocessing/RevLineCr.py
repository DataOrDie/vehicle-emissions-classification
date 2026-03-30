"""Preprocessing + Cleaning utilities for RevLineCr column.

This script includes:
1. Cleaning: grouping non-standard and missing values
2. Preprocessing: feature engineering options A and B
"""

from __future__ import annotations

import pandas as pd


DEFAULT_REVLINECR_OPTION = "A"


# =========================
# 1. CLEANING
# =========================
def clean_revlinecr(series: pd.Series) -> pd.Series:
    """
    Clean RevLineCr values:
    - Standard values: Y, N
    - Non-standard → UNKNOWN
    - Missing → MISSING
    """

    cleaned = series.astype("string").str.strip()

    # Missing values
    cleaned = cleaned.fillna("MISSING")

    # Define valid values
    valid_values = ["Y", "N", "MISSING"]

    # Anything not valid → UNKNOWN
    cleaned = cleaned.apply(
        lambda x: x if x in valid_values else "UNKNOWN"
    )

    return cleaned


# =========================
# 2. PREPROCESSING
# =========================
def preprocess_revlinecr_option_a(
    df: pd.DataFrame,
    source_col: str = "RevLineCr",
) -> pd.DataFrame:
    """
    Option A:
    - Clean values
    - Add indicator variables
    """

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()

    # CLEANING
    result["RevLineCr_clean"] = clean_revlinecr(result[source_col])

    # INDICATORS
    result["revlinecr_is_nonstandard"] = (result["RevLineCr_clean"] == "UNKNOWN").astype(int)
    result["revlinecr_is_missing"] = (result["RevLineCr_clean"] == "MISSING").astype(int)

    # ONE HOT
    result = pd.get_dummies(result, columns=["RevLineCr_clean"], prefix="revlinecr")

    return result


def preprocess_revlinecr_option_b(
    df: pd.DataFrame,
    source_col: str = "RevLineCr",
) -> pd.DataFrame:
    """
    Option B:
    - Clean values
    - Only one-hot encoding
    """

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()

    # CLEANING
    result["RevLineCr_clean"] = clean_revlinecr(result[source_col])

    # ONE HOT
    result = pd.get_dummies(result, columns=["RevLineCr_clean"], prefix="revlinecr")

    return result


# =========================
# 3. DISPATCH FUNCTION
# =========================
def preprocess_revlinecr(
    df: pd.DataFrame,
    option: str = DEFAULT_REVLINECR_OPTION,
) -> pd.DataFrame:

    if option.upper() == "A":
        return preprocess_revlinecr_option_a(df)

    if option.upper() == "B":
        return preprocess_revlinecr_option_b(df)

    raise ValueError("option must be 'A' or 'B'")
