"""Preprocessing utilities for the BalanceGross column.

Paths:
- drop: drop the original column
- trees: clean numeric values and keep natural units
"""

from __future__ import annotations

import pandas as pd


def _to_numeric_balancegross(series: pd.Series) -> pd.Series:
	"""Convert BalanceGross values to numeric currency amounts."""
	cleaned = series.astype("string").str.strip()
	cleaned = cleaned.mask(cleaned.fillna("").eq(""), pd.NA)

	cleaned = (
		cleaned
		.str.replace("$", "", regex=False)
		.str.replace(",", "", regex=False)
	)

	return pd.to_numeric(cleaned, errors="coerce")


def preprocess_balancegross(
	df: pd.DataFrame,
	option: str = "drop",
	source_col: str = "BalanceGross",
) -> pd.DataFrame:
	"""Preprocess BalanceGross with either drop or tree-friendly natural units."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	option_lower = option.lower()

	if option_lower in {"drop", "a"}:
		result = result.drop(columns=[source_col])
		return result

	if option_lower == "trees":
		result[source_col] = _to_numeric_balancegross(result[source_col])
		return result

	raise ValueError("option must be 'drop' or 'trees'")
