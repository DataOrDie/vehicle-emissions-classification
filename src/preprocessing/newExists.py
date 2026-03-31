"""Preprocessing utilities for the NewExist column.

NewExist accepted values:
- 1: Existing business
- 2: New business
"""

from __future__ import annotations

import pandas as pd


# Choose the default preprocessing path.
# Allowed values: "A" or "B".
DEFAULT_NEWEXIST_OPTION = "A"


def _standardize_and_convert_newexist(series: pd.Series) -> pd.Series:
	"""Convert NewExist values to numeric after basic text cleanup.

	- Trims whitespace
	- Converts blank strings to missing
	- Converts non-numeric text (letters/symbols) to missing via coercion
	"""
	cleaned = series.astype("string").str.strip()
	cleaned = cleaned.mask(cleaned.fillna("").eq(""), pd.NA)
	return pd.to_numeric(cleaned, errors="coerce")


def preprocess_newexist_option_a(
	df: pd.DataFrame,
	source_col: str = "NewExist",
) -> pd.DataFrame:
	"""Option A preprocessing for NewExist.

	Steps:
	1. Convert NewExist to numeric.
	2. Create `is_new_business` where value 2 -> 1, otherwise 0.
	3. Create `newexist_missing_or_invalid` where values are missing/invalid:
	   - 0
	   - negative values
	   - values > 2
	   - letters/non-numeric
	   - blanks
	"""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	newexist_num = _standardize_and_convert_newexist(result[source_col])

	invalid_mask = (
		newexist_num.isna()
		| (newexist_num == 0)
		| (newexist_num < 0)
		| (newexist_num > 2)
	)

	result[source_col] = newexist_num
	# Nullable boolean masks can contain <NA>; fill before integer casting.
	result["is_new_business"] = (newexist_num == 2).fillna(False).astype(int)
	result["newexist_missing_or_invalid"] = invalid_mask.fillna(True).astype(int)
	result = result.drop(columns=[source_col])

	return result


def preprocess_newexist_option_b(
	df: pd.DataFrame,
	source_col: str = "NewExist",
) -> pd.DataFrame:
	"""Option B preprocessing for NewExist.

	Steps:
	1. Convert NewExist to numeric.
	2. Remove rows with missing NewExist after conversion.
	3. Create `is_new_business` where value 2 -> 1, otherwise 0.
	"""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	newexist_num = _standardize_and_convert_newexist(result[source_col])
	result[source_col] = newexist_num

	result = result[result[source_col].notna()].copy()
	result["is_new_business"] = (result[source_col] == 2).fillna(False).astype(int)
	result = result.drop(columns=[source_col])

	return result


def preprocess_newexist(
	df: pd.DataFrame,
	option: str = DEFAULT_NEWEXIST_OPTION,
	source_col: str = "NewExist",
) -> pd.DataFrame:
	"""Dispatch NewExist preprocessing based on selected option.

	Parameters
	----------
	df : pd.DataFrame
		Input dataset.
	option : str
		"A" for Option A, "B" for Option B.
	source_col : str
		Column name for NewExist.
	"""
	option_upper = option.upper()

	if option_upper == "A":
		return preprocess_newexist_option_a(df=df, source_col=source_col)
	if option_upper == "B":
		return preprocess_newexist_option_b(df=df, source_col=source_col)

	
	raise ValueError("option must be 'A' or 'B'")
