# About the Data - FOR BAGGING & RANDOM FOREST

## Column Type Classification (Simplified for Bagging)

| Field             | Type                             | Category             | Bagging Notes                                                  |
| ----------------- | -------------------------------- | -------------------- | -------------------------------------------------------------- |
| id                | Text (identifier)                | Categorical_Nominal  | Drop from features; unique identifiers add no signal.          |
| LoanNr_ChkDgt     | Text (identifier)                | Categorical_Nominal  | Use for dedup checks only; do not use as feature.              |
| Name              | Text                             | Categorical_Nominal  | High cardinality; avoid direct use.                             |
| City              | Text                             | Categorical_Nominal  | Frequency-bucket or group by region.                           |
| State             | Text (2-letter code)             | Categorical_Nominal  | Good regional feature; treat as categorical.                   |
| Bank              | Text                             | Categorical_Nominal  | High cardinality; use frequency bucketing (top-k + OTHER).     |
| BankState         | Text (2-letter code)             | Categorical_Nominal  | Can differ from borrower state; useful for lender patterns.    |
| ApprovalDate      | Date/Time                        | Categorical_Ordinal  | Extract year/month instead of raw date.                        |
| ApprovalFY        | Year                             | Categorical_Ordinal  | Macro conditions and policy context.                           |
| NoEmp             | Integer count                    | Numerical_Discrete   | Handle zeros carefully; consider banding or imputation flags.  |
| NewExist          | Encoded category (1/2)           | Categorical_Ordinal  | Recode 0 as Unknown; strong predictor of business survival.    |
| CreateJob         | Integer count                    | Numerical_Discrete   | Program impact; often zero; keep as-is or use with caution.    |
| RetainedJob       | Integer count                    | Numerical_Discrete   | Program impact; often zero; combine with CreateJob if needed.  |
| FranchiseCode     | Text/code                        | Categorical_Nominal  | Recode to Is_Franchise (binary); franchise = stability signal. |
| UrbanRural        | Encoded category (0/1/2)         | Categorical_Ordinal  | Keep as discrete buckets; macro/geographic context.            |
| RevLineCr         | Text flag                        | Categorical_Nominal  | Normalize to Y/N/UNKNOWN/MISSING; 24.3% non-standard values.  |
| LowDoc            | Text flag                        | Categorical_Nominal  | Normalize to Y/N/UnknownOrMissing; policy & risk signal.       |
| DisbursementDate  | Date/Time                        | Categorical_Ordinal  | Extract date components; check time-to-disbursement vs approval. |
| DisbursementGross | Currency → Numeric               | Numerical_Continuous | Excellent quality (99.99%; 1 zero value). Strong risk signal. |
| BalanceGross      | Currency → Numeric               | Numerical_Continuous | Often zero or sparse; keep as-is for signal.                   |
| Accept            | Binary label (0/1)               | Categorical_Ordinal  | Target: 0 = not approved, 1 = approved.                        |

---

## Risk Assessment View (Bank Perspective)

Based on the data dictionary and train sample, these are the most critical columns to analyze for risk.

### 1. Business Maturity (NewExist)

Business age is one of the strongest predictors of survival. Startups (coded as 2) have significantly higher failure rates than established businesses (1).

### 2. Capacity and Scale (NoEmp & DisbursementGross)

Debt capacity matters. A 1-employee firm borrowing $500K faces much higher risk than a 50-employee firm with the same loan. Analyze loan-to-employee ratio as proxy for repayment capacity.

### 3. Business Model Stability (FranchiseCode)

Franchises have proven operating models; independent firms are riskier. Codes 00000/00001 = non-franchise; other codes indicate franchise relationship.

### 4. Economic Impact (CreateJob & RetainedJob)

In SBA lending, jobs created/retained matter beyond pure credit risk. Loans creating jobs are more attractive under government-guaranteed programs, even at similar risk.

### 5. Loan Type & Process (RevLineCr & LowDoc)

- **RevLineCr**: Revolving lines signal cash-flow struggles if heavily used.
- **LowDoc**: Lower-documentation loans imply different risk and policy behavior, often smaller amounts with different eligibility rules.

### 6. Macro Conditions (ApprovalFY & UrbanRural)

Macro context affects approval patterns: a 2006 loan (pre-2008 crash) differs from 1996. Urban vs Rural areas have different customer bases and economic conditions.

---

## Modeling Notes for Tree-Based Ensembles

### Current Direction

We are using tree-based ensembles (Bagging & Random Forests) for this problem. Prioritize:
- Split-friendly, leakage-safe features
- Simple, robust preprocessing (trees benefit from clean buckets, not aggressive scaling)
- Clean categorical normalization over complex transformations

### Target

- **Accept**: Binary target (0 = not approved, 1 = approved)
- Verify class imbalance on full dataset before modeling

### Identifier & Leakage-Prone Fields

- **id**: Unique row identifier; drop entirely
- **LoanNr_ChkDgt**: Loan identifier; useful for dedup only, not a feature
- **ChgOffDate**: Post-outcome information; likely leakage risk for approval prediction

### Field Grouping for Trees

**Borrower Profile Fields:**
- Name, City, State, NoEmp, NewExist, FranchiseCode, UrbanRural
- Requires category cleaning; handle non-standard values carefully

**Loan & Process Fields:**
- Bank, BankState, ApprovalDate, ApprovalFY, DisbursementDate, DisbursementGross, BalanceGross
- Requires currency parsing and date extraction; clean non-standard values

**Program & Policy Flags:**
- RevLineCr, LowDoc
- Requires normalization and explicit UNKNOWN/MISSING buckets

### Strongest Risk Signals

Priority ranking for trees:
1. Business size and maturity (NoEmp, NewExist)
2. Requested amount (DisbursementGross)
3. Lender and program characteristics (Bank, RevLineCr, LowDoc)
4. Geography and time context (State/City, ApprovalFY)

### Essential Preprocessing for Trees

- Parse currency fields to numeric (DisbursementGross, BalanceGross)
- Parse dates into components: year, month, quarter
- Normalize categorical text (case, typos, encoding)
- Explicit UNKNOWN/MISSING buckets for policy flags
- Drop pure identifiers (id, LoanNr_ChkDgt)
- Avoid standardization; trees work on natural units
- Avoid target encoding unless strictly out-of-fold (use in CV only)

### Feature Engineering Defaults (Bagging/Random Forest)

Keep numeric signals in natural units after cleaning. Add robust interaction features trees can split naturally:
- `loan_per_employee` = DisbursementGross / max(NoEmp, 1)
- `jobs_total` = CreateJob + RetainedJob
- `bank_borrower_same_state` = 1 if State == BankState else 0
- `approval_year`, `approval_month` (from ApprovalDate)
- `disbursement_year`, `disbursement_month` (from DisbursementDate)
- `days_approval_to_disbursement` (with outlier clipping)

For high-cardinality categoricals (Bank, City), prefer frequency bucketing (top-k + OTHER) over wide one-hot encoding.

### Validation Rules for Bagging

- Fit all cleaning, imputing, bucketing, encoding inside each CV fold (no leakage)
- Use stratified CV; track macro F1 as primary metric
- Compare engineered feature set against minimal baseline (raw cleaned features)
- Prefer simplest stable feature set with low fold-to-fold variance
- If comparing multiple preprocessing strategies, choose by CV stability (mean + variance across folds), not single best fold

---

## Column Details (Bagging-Specific Guidance)

### NewExist

**Preprocessing:**
- Recode NewExist == 0.0 to Missing (NaN); docs define only 1 and 2
- Create explicit Unknown category
- Final categories for trees: Existing (1), New (2), Unknown

**Bagging advantage:** Trees naturally handle ordinal structure (new = higher risk)

### NoEmp

**Preprocessing options:**
1. **Option A (Aggressive)**: Drop NoEmp = 0 (lose ~1% data for cleaner signal)
2. **Option B (Balanced)**: Replace 0 → NaN, impute median, add NoEmp_is_zero flag
3. **Option C (Preserve)**: Keep 0 as meaningful bucket if pre-operational/unknown staffing is valid

**Feature engineering:**
- For skewed distribution: add NoEmp_band (0, 1, 2-5, 6-10, 11-50, >50)
- loan_per_employee = DisbursementGross / max(NoEmp, 1)
- Add NoEmp_is_zero flag when using Option B/C

**Bagging recommendation:** Start with Option B (impute + flag); compare against A/C via CV

**Strong baseline:** NoEmp_is_zero flag + NoEmp_clean (imputed) + NoEmp_band

### FranchiseCode

**Preprocessing:**
- Values: 00000 or 00001 = non-franchise; others = franchise codes
- Recode to binary: `Is_Franchise` (0 = non-franchise, 1 = franchise)
- Or keep detailed franchise codes if cardinality is manageable

**Bagging insight:** Franchises = proven models = lower risk; trees can exploit this split naturally

### UrbanRural

**Preprocessing:**
- Keep discrete buckets: 0 = Undefined, 1 = Urban, 2 = Rural
- Treat as categorical (ordered but small cardinality)
- No imputation needed if no missing values

**Bagging insight:** Urban/Rural = macro/geographic context; trees can learn separate risk profiles

### RevLineCr

**Data quality issue:** 24.3% non-standard values (0, T, Q, blanks); cannot ignore

**Preprocessing:**
- Create RevLineCr_clean with buckets: Y, N, UNKNOWN (for 0/T/Q), MISSING
- Add flags: RevLineCr_is_nonstandard, RevLineCr_is_missing
- Avoid arbitrary ordinal encoding

**Sensitivity testing:** Try Policy A (0→UNKNOWN), Policy B (0→N, T/Q→UNKNOWN), Policy C (keep 0 separate)
- Choose most stable across CV folds (mean + variance), not single best fold

**Bagging recommendation:** Start with Policy A; ablate to Policy B if stable improvement observed

**Default starting point:**
- RevLineCr_clean: Y/N/UNKNOWN/MISSING
- Include nonstandard and missing flags
- Trees naturally handle categorical buckets

### LowDoc

**Data quality issue:** ~0.42% non-standard values (0, S, C, A, R, blanks); small but keep to avoid bias

**Preprocessing:**
- Option A: LowDoc_clean with Y, N, UNKNOWN, MISSING buckets
- Option B (Compact): LowDoc_clean with Y, N, UnknownOrMissing (merge rare buckets)
- Add flag: LowDoc_was_nonstandard_or_missing

**Bagging recommendation:** Use Option B (compact) for simplicity; avoid removing rows

**Encoding:** Categorical buckets passed to tree models directly

### DisbursementGross

**Data quality:** Excellent (99.99% valid); 1 zero-value outlier (row 11414, IL)

**Preprocessing options:**
- Keep as-is (let model learn zero = special case)
- Investigate row 11414 for actual approval status
- Consider zero as potential data error if needed

**Features:**
- Use in ratio features (loan_per_employee)
- May add log-transformed version for nonlinearity: log1p(DisbursementGross)

**Bagging insight:** Strong risk signal; raw values work well for trees

### ApprovalDate & ApprovalFY

**Preprocessing:**
- Extract year, month from ApprovalDate (not raw date)
- ApprovalFY already provides fiscal year context
- Create date components inside CV pipeline to avoid leakage

**Bagging note:** Macro conditions (pre-2008 vs post-2008) affect approval patterns; trees can split on these

### CreateJob & RetainedJob

**Bagging insight:** Program impact features; often zero
- Keep as-is or combine: jobs_total = CreateJob + RetainedJob
- Investigate correlation with approval before use

### Bank & BankState

**Preprocessing:**
- Clean non-standard bank names (case, typos)
- For high-cardinality Bank: use frequency bucketing (top-k + OTHER)
- BankState can be used as-is (state codes)
- Optional feature: bank_borrower_same_state (1 if Bank state == Borrower state)

**Bagging insight:** Lender behavior varies; trees can split on lender identity after bucketing

---

## Data Quality Checks & Open Questions

**DisbursementGross zero value (row 11414):**
- Investigate if loan was actually approved despite $0 disbursement
- If data error: drop or mark as suspicious
- If legitimate (pre-approved but not yet drawn): keep and let model decide

**Empty Bank / BankState values:**
- Count approved vs not approved in these rows
- Decide: drop rows or impute with UNKNOWN category

**RevLineCr & LowDoc distribution:**
- Verify policy A/B/C stability via CV before final decision
- Non-standard buckets often capture "data quality signal"

   Compare AUC/F1/PR-AUC and calibration; choose simplest stable winner.

Strong default baseline:

- NoEmp_is_zero flag
- NoEmp_clean = NoEmp.replace(0, np.nan) with median imputation
- NoEmp_band feature for nonlinearity

## LowDoc column

Since LowDoc is essentially a binary program flag (Y/N), those rare values should be treated as data quality artifacts, not real stable categories. Strategies for modeling:

Option A: Define a canonical mapping policy

- Keep Y and N as valid.
- Map 0, S, C, A, R, and blanks to Unknown.


- Create LowDoc_clean with categories: Y, N, Unknown, Missing.
- Add LowDoc_is_nonstandard as a binary flag (1 if original value in 0/S/C/A/R).
- Add LowDoc_is_missing as a binary flag.
- Avoid dropping rows. These rows are only 0.42% total, but removing them can still bias data and reduce robustness.
  Usually better to keep them and let model learn if Unknown/Missing carries risk signal.

Encoding by model type:

- Tree models (Bagging/ExtraTrees, XGBoost/LightGBM/CatBoost): keep Option A categorical buckets directly (ordinal/frequency/one-hot depending on implementation constraints).
- Linear/logistic models (secondary baseline only): one-hot encode LowDoc_clean and include indicator flags.

Validation and leakage control:

- Fit all preprocessing inside CV folds only (pipeline).
- If you use target encoding for LowDoc, do out-of-fold target encoding only.
- Because rare buckets are tiny, avoid treating each as separate category unless you have strong domain reason.
- Run an ablation check to compare:
  - Y/N only
  - Y/N/Unknown/Missing
  - Y/N + indicator flags
- Select the simplest version that gives stable CV performance and calibration.

Option C: Practical defaults

- LowDoc_clean: Y, N, UnknownOrMissing (single combined fallback bucket)
- One flag: LowDoc_was_nonstandard_or_missing

Then use compact categorical encoding for tree models, and one-hot only if required by the implementation.

## RevLineCr column

Non-standard values are not rare, mainly because 0 is 24.3% of the dataset. That means you should not just drop them.

Recommended approach:

1. do not force-map 0 to Y or N.
2. Use RevLineCr_clean buckets:
   - Y
   - N
   - UNKNOWN (for 0, T, Q)
   - MISSING
3. Add quality flags (These flags often help models capture “data quality signal.”):

- RevLineCr_is_nonstandard: 1 for 0/T/Q, else 0.
- RevLineCr_is_missing: 1 for missing, else 0.

Encoding:

- Tree models: pass as categorical buckets with compact encoding, and include quality flags.
- Avoid arbitrary ordinal encoding (Y=1, N=0, UNKNOWN=2) unless explicitly justified.

Sensitivity tests(Choose by cross-validation stability (mean and variance across folds), not one best fold.):

- Policy A: 0/T/Q -> UNKNOWN
- Policy B: 0 -> N, T/Q -> UNKNOWN
- Policy C: keep 0 as separate category, T/Q grouped

  Choose the most stable option (mean + variance across folds), not just best single score.
  Keep preprocessing inside CV pipeline
  Fit encoders/imputers within each fold only.
  If using target encoding, use out-of-fold encoding only to avoid leakage.

Default starting point:

- RevLineCr_clean: Y/N/UNKNOWN/MISSING
- Keep 0 inside UNKNOWN at first
- Add nonstandard and missing flags
- Validate with ablation; if mapping 0->N consistently improves and is stable, then adopt it

### Bank

some rows have empty banks (how many approved , non approved?)

## BankState

some rows are empty. verify if we can easily remove them, how may approved/ not approved

### ApprovalDate

what kind of data can we infer from the dates? are they important or they add bias to our model/training?
are the "test" data and kaggle data evaluating this datapoints?

### ApprovalFY

lets match and plot aprovaldate and ApprovalFY by year, and reforate approvaldate

### CreateJob and RetainedJob

review is 0 jobs has approved loans

### FranchiseCode

review what insights we can obtain from this field

### UrbanRural

get insights about this field

### ChgOffDate - no existe

### DisbursementDate

date = approved ?
does it represent anything at training time?

### DisbursementGross

Excellent data quality with 20,767 valid values (99.99%)
1 invalid value: a $0.00 disbursement (row 11414, IL) - what to do?? delete? is this "approved"?

### BalanceGross

are there any correlations between having a previous balance and new loan approvals? Can a model "see" this connections?

---

## Column Type Classification Table

| Field             | Type                             | Category             | Notes                                                                                |
| ----------------- | -------------------------------- | -------------------- | ------------------------------------------------------------------------------------ |
| id                | Text (identifier)                | Categorical_Nominal  | Unique row id. High cardinality; drop from modeling features.                        |
| LoanNr_ChkDgt     | Text (identifier)                | Categorical_Nominal  | Loan application id. Use for dedup/checks, not as predictive signal.                 |
| Name              | Text                             | Categorical_Nominal  | Borrower name. Very high cardinality; typically avoid direct use.                    |
| City              | Text                             | Categorical_Nominal  | Geographic category; can be grouped by frequency or encoded carefully.               |
| State             | Text (2-letter code)             | Categorical_Nominal  | No inherent order. Good regional feature.                                            |
| Bank              | Text                             | Categorical_Nominal  | Lender identity. High cardinality and potentially strong policy signal.              |
| BankState         | Text (2-letter code)             | Categorical_Nominal  | State of the bank; can differ from borrower state.                                   |
| ApprovalDate      | Date/Time                        | Categorical_Ordinal  | Time-ordered feature. Extract year/month/quarter instead of raw date text.           |
| ApprovalFY        | Year (integer/text in source)    | Categorical_Ordinal  | Ordered fiscal year; can also be treated as Numerical_Discrete after cleaning.       |
| NoEmp             | Integer count                    | Numerical_Discrete   | Employee count (0, 1, 2...). Consider handling suspicious zeros.                     |
| NewExist          | Encoded category (1/2)           | Categorical_Ordinal  | 1 = existing, 2 = new. Ordered coding but behaves like a small categorical variable. |
| CreateJob         | Integer count                    | Numerical_Discrete   | Number of jobs created. Non-negative count variable.                                 |
| RetainedJob       | Integer count                    | Numerical_Discrete   | Number of jobs retained. Non-negative count variable.                                |
| FranchiseCode     | Text/code                        | Categorical_Nominal  | Code values are labels, not magnitudes (even if numeric-looking).                    |
| UrbanRural        | Encoded category (0/1/2)         | Categorical_Ordinal  | 0 = undefined, 1 = urban, 2 = rural. Encoded levels, not continuous distance.        |
| RevLineCr         | Text flag                        | Categorical_Nominal  | Y/N plus non-standard values; normalize to stable buckets.                           |
| LowDoc            | Text flag                        | Categorical_Nominal  | Y/N plus non-standard values; keep unknown/missing bucket.                           |
| DisbursementDate  | Date/Time                        | Categorical_Ordinal  | Time-ordered event date; extract components for modeling.                            |
| DisbursementGross | Currency (numeric after parsing) | Numerical_Continuous | Parse currency symbols/commas to numeric float/decimal.                              |
| BalanceGross      | Currency (numeric after parsing) | Numerical_Continuous | Continuous monetary amount; often sparse/zero-heavy in practice.                     |
| Accept            | Binary label (0/1)               | Categorical_Ordinal  | Target variable: 0 = not approved, 1 = approved.                                     |
