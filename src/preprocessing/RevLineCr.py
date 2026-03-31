"""Preprocessing + Cleaning utilities for RevLineCr column"""

from __future__ import annotations
import pandas as pd

DEFAULT_REVLINECR_OPTION = "A"
ALLOWED_REVLINECR_OPTIONS = {"A", "B", "C"}

def clean_revlinecr(series: pd.Series) -> pd.Series:
    """
    Clean RevLineCr values:
    - Y, N → valid
    - Others → UNKNOWN
    - NaN → MISSING
    """
    cleaned = series.astype("string").str.strip().str.upper()
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
    """
    Option A: creates RevLineCr_clean + quality flags.
    Option B: creates only RevLineCr_clean.
    Option C: Option A + has_RevLineCr one-hot (Y=1, N=0).
    """

    option = str(option).strip().upper()
    if option not in ALLOWED_REVLINECR_OPTIONS:
        raise ValueError(
            f"Invalid option '{option}'. Use one of {sorted(ALLOWED_REVLINECR_OPTIONS)}"
        )

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    df = df.copy()

    # CLEAN
    clean_col = clean_revlinecr(df[source_col])
    df["RevLineCr_clean"] = clean_col

    # FLAGS (opción A)
    if option in {"A", "C"}:
        df["revlinecr_is_nonstandard"] = (clean_col == "UNKNOWN").astype(int)
        df["revlinecr_is_missing"] = (clean_col == "MISSING").astype(int)

    # One-hot style target for revolving credit line presence
    # Built from cleaned values; only Y/N are treated as valid binary values.
    if option == "C":
        df["has_RevLineCr"] = clean_col.map({"Y": 1, "N": 0}).fillna(0).astype(int)
        df = df.drop(columns=["RevLineCr_clean", source_col], errors="ignore")

    return df
