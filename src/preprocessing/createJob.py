"""Preprocessing utilities for the CreateJob column.

Paths:
- Option A: normalize values (min-max)
- Option B: standardize values (z-score)
"""

from __future__ import annotations

import pandas as pd


# Choose the default preprocessing path.
# Allowed values: "A" or "B".
DEFAULT_CREATEJOB_OPTION = "A"


def _to_numeric_createjob(series: pd.Series) -> pd.Series:
	"""Convert CreateJob values to numeric.

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

	# Avoid division by zero when all non-missing values are equal.
	if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
		return pd.Series(pd.NA, index=series.index, dtype="Float64")

	return ((series - min_val) / (max_val - min_val)).astype("Float64")


def _zscore_standardize(series: pd.Series) -> pd.Series:
	"""Apply z-score standardization to a numeric series."""
	mean_val = series.mean(skipna=True)
	std_val = series.std(skipna=True, ddof=0)

	# Avoid division by zero when standard deviation is 0.
	if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
		return pd.Series(pd.NA, index=series.index, dtype="Float64")

	return ((series - mean_val) / std_val).astype("Float64")


def preprocess_createjob_option_a(
	df: pd.DataFrame,
	source_col: str = "CreateJob",
) -> pd.DataFrame:
	"""Option A preprocessing for CreateJob (normalize values)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	createjob_num = _to_numeric_createjob(result[source_col])

	result[source_col] = createjob_num
	result["createjob_normalized"] = _min_max_normalize(createjob_num)
	result = result.drop(columns=[source_col])
	return result


def preprocess_createjob_option_b(
	df: pd.DataFrame,
	source_col: str = "CreateJob",
) -> pd.DataFrame:
	"""Option B preprocessing for CreateJob (standardize values)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	createjob_num = _to_numeric_createjob(result[source_col])

	result[source_col] = createjob_num
	result["createjob_standardized"] = _zscore_standardize(createjob_num)
	result = result.drop(columns=[source_col])
	return result


def preprocess_createjob(
	df: pd.DataFrame,
	option: str = DEFAULT_CREATEJOB_OPTION,
	source_col: str = "CreateJob",
) -> pd.DataFrame:
	"""Dispatch CreateJob preprocessing based on selected option.

	Parameters
	----------
	df : pd.DataFrame
		Input dataset.
	option : str
		"A" for Option A (normalize), "B" for Option B (standardize).
	source_col : str
		Column name for CreateJob.
	"""
	option_upper = option.upper()

	if option_upper == "A":
		return preprocess_createjob_option_a(df=df, source_col=source_col)
	if option_upper == "B":
		return preprocess_createjob_option_b(df=df, source_col=source_col)

	raise ValueError("option must be 'A' or 'B'")
