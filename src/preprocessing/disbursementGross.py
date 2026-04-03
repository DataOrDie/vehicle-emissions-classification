"""Preprocessing utilities for the DisbursementGross column.

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
DEFAULT_DISBURSEMENTGROSS_OPTION = "A"


def _to_numeric_disbursementgross(series: pd.Series) -> pd.Series:
	"""Convert DisbursementGross values to numeric.

	- Trims whitespace
	- Converts blank strings to missing
	- Removes currency symbols and thousands separators
	- Coerces non-numeric values to missing
	"""
	cleaned = series.astype("string").str.strip()
	cleaned = cleaned.mask(cleaned.fillna("").eq(""), pd.NA)

	cleaned = (
		cleaned
		.str.replace("$", "", regex=False)
		.str.replace(",", "", regex=False)
	)

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


def _log1p_then_zscore(series: pd.Series) -> pd.Series:
	"""Apply log1p transform then z-score standardization.

	Negative values are clipped to 0 before log1p to keep the transform valid.
	"""
	logged = series.clip(lower=0).apply(lambda x: pd.NA if pd.isna(x) else np.log1p(x))
	logged = pd.to_numeric(logged, errors="coerce")
	return _zscore_standardize(logged)


def preprocess_disbursementgross_option_a(
	df: pd.DataFrame,
	source_col: str = "DisbursementGross",
) -> pd.DataFrame:
	"""Option A preprocessing for DisbursementGross (normalize values)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	disbursementgross_num = _to_numeric_disbursementgross(result[source_col])

	#Sobrescribir columna original
	result[source_col] = _min_max_normalize(disbursementgross_num)

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

	# Sobrescribir columna original
	result[source_col] = _zscore_standardize(disbursementgross_num)

	return result


def preprocess_disbursementgross_option_c(
	df: pd.DataFrame,
	source_col: str = "DisbursementGross",
) -> pd.DataFrame:
	"""Option C preprocessing for DisbursementGross (log1p + standardize)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	disbursementgross_num = _to_numeric_disbursementgross(result[source_col])

	result[source_col] = _log1p_then_zscore(disbursementgross_num)

	return result


def preprocess_disbursementgross_option_trees(
	df: pd.DataFrame,
	source_col: str = "DisbursementGross",
) -> pd.DataFrame:
	"""Tree option for DisbursementGross (clean numeric values, keep natural units)."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	disbursementgross_num = _to_numeric_disbursementgross(result[source_col])
	result[source_col] = disbursementgross_num
	return result


def preprocess_disbursementgross(
	df: pd.DataFrame,
	option: str = DEFAULT_DISBURSEMENTGROSS_OPTION,
	source_col: str = "DisbursementGross",
) -> pd.DataFrame:
	"""Dispatch DisbursementGross preprocessing based on selected option."""
	option_upper = option.upper()

	if option_upper == "A":
		return preprocess_disbursementgross_option_a(df=df, source_col=source_col)
	if option_upper == "B":
		return preprocess_disbursementgross_option_b(df=df, source_col=source_col)
	if option_upper == "C":
		return preprocess_disbursementgross_option_c(df=df, source_col=source_col)
	if option_upper == "TREES":
		return preprocess_disbursementgross_option_trees(df=df, source_col=source_col)

	raise ValueError("option must be 'A', 'B', 'C', or 'trees'")
