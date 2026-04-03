"""One-step preprocessing pipeline.

This module centralizes the full preprocessing flow currently used in
`notebooks/feature-enginnering.ipynb` so it can be imported and reused from
scripts, training code, or other notebooks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from preprocessing import LowDoc
from preprocessing import RevLineCr
from preprocessing import accept
from preprocessing import approvalDate
from preprocessing import approvalFY
from preprocessing import balanceGross
from preprocessing import base_cleaning
from preprocessing import city_bank
from preprocessing import createJob
from preprocessing import disbursementDate
from preprocessing import disbursementGross
from preprocessing import franchise_code
from preprocessing import newExists
from preprocessing import noemp
from preprocessing import retainedJob
from preprocessing import urban_rural


@dataclass
class OneStepOptions:
	"""Default options aligned with notebooks/feature-enginnering.ipynb."""

	noemp_option: str = "log"
	newexist_option: str = "A"
	createjob_option: str = "A"
	retainedjob_option: str = "A"
	approvaldate_option: str = "A"
	approvalfy_option: str = "A"
	franchise_option: str = "binary"
	urbanrural_option: str = "onehot"
	revlinecr_option: str = "C"
	lowdoc_option: str = "C"
	disbursementgross_option: str = "A"
	accept_option: str = "skip"
	local_state: str = "IL"


def get_default_options() -> dict[str, Any]:
	"""Return notebook-aligned defaults for all preprocessing steps."""
	return asdict(OneStepOptions())


def preprocess_one_step(
	df: pd.DataFrame,
	options: OneStepOptions | None = None,
) -> pd.DataFrame:
	"""Run all preprocessing modules in notebook order.

	Parameters
	----------
	df : pd.DataFrame
		Input raw dataframe.
	options : OneStepOptions | None
		Pipeline options. If None, notebook defaults are used.
	"""
	opts = options or OneStepOptions()

	df_out = df.copy()

	# 1) Base cleanup
	df_out = base_cleaning.clean_base_columns(df_out, local_state=opts.local_state)

	# 2) NoEmp
	df_out = noemp.preprocess_noemp(df_out, option=opts.noemp_option, source_col="NoEmp")

	# 3) City/Bank binary encoding
	df_out = city_bank.get_city_bank_encoder(df_out)

	# 4) NewExist
	df_out = newExists.preprocess_newexist(
		df=df_out,
		option=opts.newexist_option,
		source_col="NewExist",
	)

	# 5) CreateJob
	df_out = createJob.preprocess_createjob(
		df=df_out,
		option=opts.createjob_option,
		source_col="CreateJob",
	)

	# 6) RetainedJob
	df_out = retainedJob.preprocess_retainedjob(
		df=df_out,
		option=opts.retainedjob_option,
		source_col="RetainedJob",
	)

	# 7) ApprovalDate
	df_out = approvalDate.preprocess_approvaldate(
		df=df_out,
		option=opts.approvaldate_option,
		source_col="ApprovalDate",
	)

	# 8) ApprovalFY
	df_out = approvalFY.preprocess_approvalfy(
		df=df_out,
		option=opts.approvalfy_option,
		source_col="ApprovalFY",
	)

	# 9) FranchiseCode
	df_out = franchise_code.preprocess_franchise_code(
		df=df_out,
		option=opts.franchise_option,
		source_col="FranchiseCode",
	)

	# 10) UrbanRural
	df_out = urban_rural.preprocess_urban_rural(
		df=df_out,
		option=opts.urbanrural_option,
		source_col="UrbanRural",
	)

	# 11) RevLineCr
	df_out = RevLineCr.preprocess_revlinecr(
		df=df_out,
		option=opts.revlinecr_option,
		source_col="RevLineCr",
	)

	# 12) LowDoc
	df_out = LowDoc.preprocess_lowdoc(
		df=df_out,
		option=opts.lowdoc_option,
		source_col="LowDoc",
	)

	# 13) DisbursementDate
	df_out = disbursementDate.preprocess_disbursementdate(
		df=df_out,
		source_col="DisbursementDate",
	)

	# 14) BalanceGross
	df_out = balanceGross.preprocess_balancegross(
		df=df_out,
		source_col="BalanceGross",
	)

	# 15) Accept
	df_out = accept.preprocess_accept(
		df=df_out,
		option=opts.accept_option,
		source_col="Accept",
	)

	# 16) DisbursementGross
	df_out = disbursementGross.preprocess_disbursementgross(
		df=df_out,
		option=opts.disbursementgross_option,
		source_col="DisbursementGross",
	)

	return df_out


__all__ = ["OneStepOptions", "get_default_options", "preprocess_one_step"]
