"""Preprocessing utilities for the Accept column.

Path:
- Convert values to numeric
"""

from __future__ import annotations

import pandas as pd


def _to_numeric_accept(series: pd.Series) -> pd.Series:
	"""Convert Accept values to numeric.

	- Trims whitespace
	- Converts blank strings to missing
	- Coerces non-numeric values to missing
	"""
	cleaned = series.astype("string").str.strip()
	cleaned = cleaned.mask(cleaned.fillna("").eq(""), pd.NA)
	return pd.to_numeric(cleaned, errors="coerce")


def preprocess_accept(
	df: pd.DataFrame,
	source_col: str = "Accept",
) -> pd.DataFrame:
	"""Preprocess Accept by converting values to numeric."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	result[source_col] = _to_numeric_accept(result[source_col])

	return result
