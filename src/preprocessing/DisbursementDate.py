"""Preprocessing utilities for the DisbursementDate column.

Path:
- Drop the original column
"""

from __future__ import annotations

import pandas as pd


def preprocess_disbursementdate(
	df: pd.DataFrame,
	source_col: str = "DisbursementDate",
) -> pd.DataFrame:
	"""Preprocess DisbursementDate by dropping the column."""
	if source_col not in df.columns:
		raise KeyError(f"Column '{source_col}' not found in DataFrame")

	result = df.copy()
	result = result.drop(columns=[source_col])

	return result