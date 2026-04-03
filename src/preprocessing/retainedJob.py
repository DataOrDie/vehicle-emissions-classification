"""Preprocessing utilities for the RetainedJob column.

Paths:
- Option A: normalize values (min-max)
- Option B: standardize values (z-score)
- Option C: log1p + standardize values (z-score)
- Option trees: clean numeric values and keep natural units
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Choose the default preprocessing path.
# Allowed values: "A", "B", or "C".
DEFAULT_RETAINEDJOB_OPTION = "A"


def _to_numeric_retainedjob(series: pd.Series) -> pd.Series:
	"""Convert RetainedJob values to numeric.

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


def _log1p_then_zscore(series: pd.Series) -> pd.Series:
	"""Apply log1p transform then z-score standardization.

	Negative values are clipped to 0 before log1p to keep the transform valid.
	"""
	logged = series.clip(lower=0).apply(lambda x: pd.NA if pd.isna(x) else np.log1p(x))
	logged = pd.to_numeric(logged, errors="coerce")
	return _zscore_standardize(logged)


def preprocess_retainedjob_option_a(
	df: pd.DataFrame,
	source_col: str = "RetainedJob",
) -> pd.DataFrame:
	"""Option A preprocessing for RetainedJob (normalize values)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	retainedjob_num = _to_numeric_retainedjob(result[source_col])

	result[source_col] = retainedjob_num
	result["retainedjob_normalized"] = _min_max_normalize(retainedjob_num)
	result = result.drop(columns=[source_col])
	return result


def preprocess_retainedjob_option_b(
	df: pd.DataFrame,
	source_col: str = "RetainedJob",
) -> pd.DataFrame:
	"""Option B preprocessing for RetainedJob (standardize values)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	retainedjob_num = _to_numeric_retainedjob(result[source_col])

	result[source_col] = retainedjob_num
	result["retainedjob_standardized"] = _zscore_standardize(retainedjob_num)
	result = result.drop(columns=[source_col])
	return result


def preprocess_retainedjob_option_c(
	df: pd.DataFrame,
	source_col: str = "RetainedJob",
) -> pd.DataFrame:
	"""Option C preprocessing for RetainedJob (log1p + standardize)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	retainedjob_num = _to_numeric_retainedjob(result[source_col])

	result[source_col] = retainedjob_num
	result["retainedjob_log1p_standardized"] = _log1p_then_zscore(retainedjob_num)
	result = result.drop(columns=[source_col])
	return result


def preprocess_retainedjob_option_trees(
	df: pd.DataFrame,
	source_col: str = "RetainedJob",
) -> pd.DataFrame:
	"""Tree option for RetainedJob (clean numeric values, keep natural units)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	result[source_col] = _to_numeric_retainedjob(result[source_col])
	return result


def preprocess_retainedjob(
	df: pd.DataFrame,
	option: str = DEFAULT_RETAINEDJOB_OPTION,
	source_col: str = "RetainedJob",
) -> pd.DataFrame:
	"""Dispatch RetainedJob preprocessing based on selected option.

	Parameters
	----------
	df : pd.DataFrame
		Input dataset.
	option : str
		"A" for Option A (normalize), "B" for Option B (standardize),
		"C" for Option C (log1p + standardize), "trees" for natural units.
	source_col : str
		Column name for RetainedJob.
	"""
	option_upper = option.upper()

	if option_upper == "A":
		return preprocess_retainedjob_option_a(df=df, source_col=source_col)
	if option_upper == "B":
		return preprocess_retainedjob_option_b(df=df, source_col=source_col)
	if option_upper == "C":
		return preprocess_retainedjob_option_c(df=df, source_col=source_col)
	if option_upper == "TREES":
		return preprocess_retainedjob_option_trees(df=df, source_col=source_col)

	raise ValueError("option must be 'A', 'B', 'C', or 'trees'")
