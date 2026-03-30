"""Preprocessing utilities for the DisbursementGross column.

Paths:
- Option A: normalize values (min-max)
- Option B: standardize values (z-score)
"""

from __future__ import annotations

import pandas as pd


# Choose the default preprocessing path.
# Allowed values: "A" or "B".
DEFAULT_DISBURSEMENTGROSS_OPTION = "A"


def _to_numeric_disbursementgross(series: pd.Series) -> pd.Series:
	"""Convert DisbursementGross values to numeric.

	- Trims whitespace
	- Converts blank strings to missing
	- Coerces non-numeric values to missing
	"""
	cleaned = series.astype("string").str.strip()
	cleaned = cleaned.mask(cleaned.fillna("").eq(""), pd.NA)
	return pd.to_numeric(cleaned, errors="coerce")


def _min_max_normalize(series: pd.Series) -> pd.Series:
	"""Apply min-max normalization to a numeric series."""
	min_val = series.min(skipna=True)
	max_val = series.max(skipna=True)

	if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
		return pd.Series(pd.NA, index=series.index, dtype="Float64")

	return ((series - min_val) / (max_val - min_val)).astype("Float64")


def _zscore_standardize(series: pd.Series) -> pd.Series:
	"""Apply z-score standardization to a numeric series."""
	mean_val = series.mean(skipna=True)
	std_val = series.std(skipna=True, ddof=0)

	if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
		return pd.Series(pd.NA, index=series.index, dtype="Float64")

	return ((series - mean_val) / std_val).astype("Float64")


def preprocess_disbursementgross_option_a(
	df: pd.DataFrame,
	source_col: str = "DisbursementGross",
) -> pd.DataFrame:
	"""Option A preprocessing for DisbursementGross (normalize values)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	disbursementgross_num = _to_numeric_disbursementgross(result[source_col])

	result[source_col] = disbursementgross_num
	result["disbursementgross_normalized"] = _min_max_normalize(disbursementgross_num)

	return result


def preprocess_disbursementgross_option_b(
	df: pd.DataFrame,
	source_col: str = "DisbursementGross",
) -> pd.DataFrame:
	"""Option B preprocessing for DisbursementGross (standardize values)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	disbursementgross_num = _to_numeric_disbursementgross(result[source_col])

	result[source_col] = disbursementgross_num
	result["disbursementgross_standardized"] = _zscore_standardize(disbursementgross_num)

	return result


def preprocess_disbursementgross(
	df: pd.DataFrame,
	option: str = DEFAULT_DISBURSEMENTGROSS_OPTION,
	source_col: str = "DisbursementGross",
) -> pd.DataFrame:
	"""Dispatch DisbursementGross preprocessing based on selected option.

	Parameters
	----------
	df : pd.DataFrame
		Input dataset.
	option : str
		"A" for Option A (normalize), "B" for Option B (standardize).
	source_col : str
		Column name for DisbursementGross.
	"""
	option_upper = option.upper()

	if option_upper == "A":
		return preprocess_disbursementgross_option_a(df=df, source_col=source_col)
	if option_upper == "B":
		return preprocess_disbursementgross_option_b(df=df, source_col=source_col)

	raise ValueError("option must be 'A' or 'B'")