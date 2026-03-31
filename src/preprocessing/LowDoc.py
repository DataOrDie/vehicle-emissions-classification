"""Preprocessing + Cleaning utilities for LowDoc column"""

from __future__ import annotations
import pandas as pd

DEFAULT_LOWDOC_OPTION = "A"
ALLOWED_LOWDOC_OPTIONS = {"A", "B", "C"}


def clean_lowdoc(series: pd.Series) -> pd.Series:
    """
    Clean LowDoc values:
    - Y, N → valid (case-insensitive)
    - Blanks and NaN → MISSING
    - Others (0, S, C, A, R) → UNKNOWN
    """
    # Normalize: strip whitespace and uppercase for case-insensitive matching
    cleaned = series.astype("string").str.strip().str.upper()
    
    # Map empty strings (blanks after stripping) to MISSING
    cleaned = cleaned.replace("", "MISSING")
    
    # Map NaN to MISSING
    cleaned = cleaned.fillna("MISSING")

    valid_values = ["Y", "N", "MISSING"]

    cleaned = cleaned.apply(
        lambda x: x if x in valid_values else "UNKNOWN"
    )

    return cleaned


def preprocess_lowdoc(
    df: pd.DataFrame,
    option: str = DEFAULT_LOWDOC_OPTION,
    source_col: str = "LowDoc",
) -> pd.DataFrame:
    """
    Preprocess LowDoc (low-documentation loan program flag).
    
    Option A: creates LowDoc_clean + quality flags (is_nonstandard, is_missing).
    Option B: creates LowDoc_clean only.
    Option C: Option A + is_LowDoc one-hot (Y=1, N=0), drops LowDoc_clean and source column.
    """
    # Validate option
    option = str(option).strip().upper()
    if option not in ALLOWED_LOWDOC_OPTIONS:
        raise ValueError(
            f"Invalid option '{option}'. Use one of {sorted(ALLOWED_LOWDOC_OPTIONS)}"
        )

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    df = df.copy()

    # CLEAN: Normalize Y/N/missing/nonstandard values
    clean_col = clean_lowdoc(df[source_col])
    df["LowDoc_clean"] = clean_col

    # Quality flags: Option A and C include these
    # These flags help models capture "data quality signal" for rare/nonstandard values
    if option in {"A", "C"}:
        df["lowdoc_is_nonstandard"] = (clean_col == "UNKNOWN").astype(int)
        df["lowdoc_is_missing"] = (clean_col == "MISSING").astype(int)

    # Binary encoding for LowDoc presence: Option C only
    # Maps only Y/N to binary; UNKNOWN and MISSING become 0 (safe default for missing data)
    if option == "C":
        df["is_LowDoc"] = clean_col.map({"Y": 1, "N": 0}).fillna(0).astype(int)
        # Drop intermediate cleaned column and original source column
        df = df.drop(columns=["LowDoc_clean", source_col], errors="ignore")

    return df
