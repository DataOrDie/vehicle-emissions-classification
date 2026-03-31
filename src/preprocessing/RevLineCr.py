"""Preprocessing + Cleaning utilities for RevLineCr column"""

from __future__ import annotations
import pandas as pd

DEFAULT_REVLINECR_OPTION = "A"

def clean_revlinecr(series: pd.Series) -> pd.Series:
    """
    Clean RevLineCr values:
    - Y, N → valid
    - Others → UNKNOWN
    - NaN → MISSING
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.fillna("MISSING")

    valid_values = ["Y", "N", "MISSING"]

    cleaned = cleaned.apply(
        lambda x: x if x in valid_values else "UNKNOWN"
    )

    return cleaned

def preprocess_revlinecr(
    df: pd.DataFrame,
    option: str = DEFAULT_REVLINECR_OPTION,
    source_col: str = "RevLineCr",
) -> pd.DataFrame:

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    df = df.copy()

    # CLEAN
    clean_col = clean_revlinecr(df[source_col])
    df["RevLineCr_clean"] = clean_col

    # FLAGS (opción A)
    if option.upper() == "A":
        df["revlinecr_is_nonstandard"] = (clean_col == "UNKNOWN").astype(int)
        df["revlinecr_is_missing"] = (clean_col == "MISSING").astype(int)

    return df
