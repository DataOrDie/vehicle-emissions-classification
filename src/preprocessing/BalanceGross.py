"""Preprocessing utilities for the BalanceGross column.

Path:
- Drop the original column
"""

from __future__ import annotations

import pandas as pd


def preprocess_balancegross(
	df: pd.DataFrame,
	source_col: str = "BalanceGross",
) -> pd.DataFrame:
	"""Preprocess BalanceGross by dropping the column."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	result = result.drop(columns=[source_col])

	return result