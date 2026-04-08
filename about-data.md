# About the Data

## Column Type Classification Table

| Field | Type | Category |
| --- | --- | --- |
| id | Text (identifier) | Categorical_Nominal |
| LoanNr_ChkDgt | Text (identifier) | Categorical_Nominal |
| Name | Text | Categorical_Nominal |
| City | Text | Categorical_Nominal |
| State | Text (2-letter code) | Categorical_Nominal |
| Bank | Text | Categorical_Nominal |
| BankState | Text (2-letter code) | Categorical_Nominal |
| ApprovalDate | Date/Time | Categorical_Ordinal |
| ApprovalFY | Year (integer/text in source) | Categorical_Ordinal |
| NoEmp | Integer count | Numerical_Discrete |
| NewExist | Encoded category (1/2) | Categorical_Ordinal |
| CreateJob | Integer count | Numerical_Discrete |
| RetainedJob | Integer count | Numerical_Discrete |
| FranchiseCode | Text/code | Categorical_Nominal |
| UrbanRural | Encoded category (0/1/2) | Categorical_Ordinal |
| RevLineCr | Text flag | Categorical_Nominal |
| LowDoc | Text flag | Categorical_Nominal |
| ChgOffDate | Date/Time | Categorical_Ordinal |
| DisbursementDate | Date/Time | Categorical_Ordinal |
| DisbursementGross | Currency (numeric after parsing) | Numerical_Continuous |
| BalanceGross | Currency (numeric after parsing) | Numerical_Continuous |
| Accept | Binary label (0/1) | Categorical_Ordinal |

---

## Risk Assessment View (Bank Perspective)

Based on the data dictionary and train sample, these are the most critical columns to analyze for risk.

- Business maturity: NewExist
- Capacity and scale: NoEmp, DisbursementGross
- Business model stability: FranchiseCode
- Economic impact: CreateJob, RetainedJob
- Loan type/process complexity: RevLineCr, LowDoc
- Macro conditions: ApprovalFY, UrbanRural

---

## Column Details

### id

- Description: Identifier of the data instance.
- Common values and notes:
  - Unique identifier for each row in the dataset.
  - Usually used only to identify records, not as a model feature.
- Modeling guidance:
  - unique row identifier; do not use as predictive signal.

### LoanNr_ChkDgt

- Description: Identifier of the loan petition.
- Common values and notes:
  - Unique loan application identifier.
  - Useful for tracking or deduplication checks.
- Modeling guidance:
  - loan identifier; mostly useful for dedup checks.
  - do not use as predictive signal.

### Name

- Description: Borrower name.
- Common values and notes:
  - Name of the business owner or borrower.
  - High-cardinality text; often not ideal for direct modeling.
- Modeling guidance:
  - Very high cardinality; typically avoid direct use.

### City

- Description: Borrower city.
- Common values and notes:
  - Location feature.
- Modeling guidance:
  - Geographic category; can be grouped by frequency or encoded carefully.
  - Included in Borrower Profile Fields.

### State

- Description: Borrower state.
- Common values and notes:
  - Two-letter state code in many cases.
- Modeling guidance:
  - No inherent order.
  - Good regional feature.
  - Included in Borrower Profile Fields.
- Related feature engineering:
  - State_Default_Rate (target encoding): replace State with historical approval rate for that state.

### Bank

- Description: Bank name.
- Common values and notes:
  - Name of the bank that handled the loan.
  - Can capture lender-specific behavior.
- Modeling guidance:
  - High cardinality and potentially strong policy signal.
  - Included in Loan and Process Fields.
  - These include non-standard values in some cases, so category cleaning is required.
- Open checks:
  - some rows have empty banks (how many approved, non approved?).
- Related feature engineering:
  - Bank_Market_Share: replace Bank name with frequency count in dataset.

### BankState

- Description: Bank state.
- Common values and notes:
  - State where the bank is located.
  - May differ from borrower state.
- Modeling guidance:
  - Included in Loan and Process Fields.
  - These include non-standard values in some cases, so category cleaning is required.
- Open checks:
  - some rows are empty. verify if we can easily remove them, how many approved/not approved.
- Related feature engineering:
  - Is_Local_Lender / Back_To_Back: compare State and BankState.

### ApprovalDate

- Description: Date SBA commitment issued.
- Common values and notes:
  - Date when the SBA commitment was approved.
  - Can be converted into year/month/quarter features.
- Modeling guidance:
  - Time-ordered feature. Extract year/month/quarter instead of raw date text.
  - Included in Loan and Process Fields.
  - Parse dates into features (year, month, age).
- Open checks:
  - what kind of data can we infer from the dates?
  - are they important or they add bias to our model/training?
  - are the test data and kaggle data evaluating this datapoint?
- Related feature engineering:
  - Approval_Quarter.

### ApprovalFY

- Description: Fiscal year of commitment.
- Common values and notes:
  - Fiscal year when the loan was approved.
  - Time and macroeconomic context feature.
- Modeling guidance:
  - Ordered fiscal year; can also be treated as Numerical_Discrete after cleaning.
  - Macro conditions can affect approval behavior.
- Open checks:
  - lets match and plot ApprovalDate and ApprovalFY by year, and reformat ApprovalDate.
- Related feature engineering:
  - Is_Recession_Era based on ApprovalFY.

### NoEmp

- Description: Number of business employees.
- Common values and notes:
  - Number of employees in the business.
  - Proxy for business size.
- Risk insight:
  - Debt capacity matters. If a company with 1 employee asks for $500,000, risk is much higher than a 50-employee firm asking for the same amount.
  - Analyze ratio of loan size to employee count.
- Modeling guidance:
  - Consider handling suspicious zeros.
  - For NoEmp = 0, choose based on whether 0 is invalid or meaningful.
- Column-specific recommendations:
  - Option A: treat 0 as invalid and drop those rows.
    - Pro: cleaner signal.
    - Con: lose data (about 1.02%).
  - Option B: treat 0 as missing and impute.
    - Replace 0 with NaN, impute (median/KNN/model-based), add NoEmp_was_zero flag.
    - Pro: keeps rows and preserves suspicious-value signal.
    - Con: imputation can blur patterns and true risk patterns.
  - Option C: keep 0 as meaningful state.
    - Keep 0 if it could represent pre-operational firms/unknown staffing.
    - Add explicit NoEmp = 0 bucket.
    - Pro: model can learn if this group has different risk.
    - Con: only works if 0 is truly meaningful.
  - Option D: robust feature engineering.
    - Use log1p(NoEmp) and/or bands (0, 1, 2-5, 6-10, 11-50, >50).
    - For loan_per_employee, avoid division instability:
      - loan_per_employee = loan_amount / max(NoEmp, 1)
      - or set ratio missing when NoEmp = 0 and add NoEmp_is_zero flag.
- Model-family guidance:
  - Tree models (XGBoost/LightGBM/CatBoost): indicator + raw or banded feature often works well.
  - Linear models: imputation + indicator + transformation is usually better.
- Suggested workflow:
  1. Pipeline A: drop NoEmp = 0
  2. Pipeline B: 0 to NaN + impute + NoEmp_is_zero column
  3. Pipeline C: keep 0 + NoEmp_is_zero + NoEmp bands
  Compare AUC/F1/PR-AUC and calibration; choose simplest stable winner.
- Strong default baseline:
  - NoEmp_is_zero flag
  - NoEmp_clean = NoEmp.replace(0, np.nan) with median imputation
  - NoEmp_band feature for nonlinearity
- Related feature engineering:
  - Gross_per_Employee = DisbursementGross / (NoEmp + 1)
  - Loan_Size_Ratio = DisbursementGross / NoEmp (with safeguards)

### NewExist

- Description: 1 = Existing business, 2 = New business.
- Common values and notes:
  - Whether the business is new or existing.
  - 1 = Existing business, 2 = New business.
- Risk insight:
  - Business age is one of the strongest predictors of survival.
  - Startups (often coded as 2.0) have significantly higher failure rates than established businesses (1.0).
  - New businesses often require stronger collateral or a stronger plan.
- Modeling guidance:
  - Recode NewExist == 0.0 to missing (NaN), since docs define only 1 and 2.
  - Create explicit Unknown category for missing values.
  - One-hot encode NewExist (Existing, New, Unknown).
  - Validate with CV against baseline that drops those 13 rows.
- Practical choice:
  - Simplest: drop those rows.
  - More robust: keep rows and add Unknown category.
- Related feature engineering:
  - Is_New_Business: 1 if NewExist == 2, else 0.

### CreateJob

- Description: Number of jobs created.
- Common values and notes:
  - Number of jobs expected to be created by the loan.
  - Program impact indicator.
- Risk insight:
  - In SBA-like lending, impact can matter in addition to pure credit risk.
  - A loan that creates 10 jobs can be more attractive than one that creates zero, even with similar risk profile.
- Open checks:
  - review if 0 jobs still has approved loans.
- Related feature engineering:
  - Total_Job_Impact = CreateJob + RetainedJob
  - Job_Creation_Efficiency = DisbursementGross / (CreateJob + 1)

### RetainedJob

- Description: Number of jobs retained.
- Common values and notes:
  - Number of jobs expected to be retained by the loan.
  - Program impact indicator.
- Risk insight:
  - In SBA-like lending, jobs retained can influence attractiveness under government-guaranteed programs.
- Open checks:
  - review if 0 jobs still has approved loans.
- Related feature engineering:
  - Total_Job_Impact = CreateJob + RetainedJob

### FranchiseCode

- Description: Franchise code; 00000 or 00001 = no franchise.
- Common values and notes:
  - Code indicating franchise relationship.
  - 00000 or 00001 often means no franchise.
- Risk insight:
  - Franchises have proven operating models.
  - Independent firms are often riskier.
  - Specific franchise codes may indicate lower operational uncertainty.
- Modeling guidance:
  - Code values are labels, not magnitudes (even if numeric-looking).
- Open checks:
  - review what insights we can obtain from this field.
- Related feature engineering:
  - Is_Franchise: if FranchiseCode is 0 or 1 => Independent (0), otherwise Franchise (1).

### UrbanRural

- Description: 1 = Urban, 2 = Rural, 0 = Undefined.
- Common values and notes:
  - Area type where the business operates.
  - 1 = Urban, 2 = Rural, 0 = Undefined.
- Risk insight:
  - Economic conditions in Urban vs Rural areas affect customer base and risk context.
- Modeling guidance:
  - Encoded levels, not continuous distance.
- Open checks:
  - get insights about this field.

### RevLineCr

- Description: Revolving line of credit; Y = Yes, N = No.
- Common values and notes:
  - Indicates if loan is a revolving line of credit.
  - Y/N may include non-standard values.
- Risk insight:
  - These are like credit cards for businesses.
  - High usage can signal cash-flow struggles.
- Modeling guidance:
  - Non-standard values are not rare, mainly because 0 is 24.3% of the dataset.
  - Do not just drop them.
- Recommended approach:
  1. do not force-map 0 to Y or N.
  2. Use RevLineCr_clean buckets:
     - Y
     - N
     - UNKNOWN (for 0, T, Q)
     - MISSING
  3. Add quality flags:
     - RevLineCr_is_nonstandard: 1 for 0/T/Q, else 0.
     - RevLineCr_is_missing: 1 for missing, else 0.
- Encoding:
  - Tree models: pass as categorical buckets (or one-hot).
  - Linear/logistic models: one-hot RevLineCr_clean + include indicator flags.
  - Avoid ordinal encoding (Y=1, N=0, UNKNOWN=2) unless explicitly justified.
- Sensitivity tests (choose by CV mean and variance across folds, not one best fold):
  - Policy A: 0/T/Q -> UNKNOWN
  - Policy B: 0 -> N, T/Q -> UNKNOWN
  - Policy C: keep 0 as separate category, T/Q grouped
- CV leakage control:
  - Keep preprocessing inside CV pipeline.
  - Fit encoders/imputers within each fold only.
  - If using target encoding, use out-of-fold encoding only.
- Default starting point:
  - RevLineCr_clean: Y/N/UNKNOWN/MISSING
  - Keep 0 inside UNKNOWN at first
  - Add nonstandard and missing flags
  - Validate with ablation; if mapping 0->N consistently improves and is stable, then adopt it.

### LowDoc

- Description: LowDoc loan program; Y = Yes, N = No.
- Common values and notes:
  - Indicates if this is a low-documentation loan.
  - Y/N may include non-standard values.
- Risk insight:
  - Lower-documentation processing can imply different risk and policy behavior.
  - If LowDoc = Y, often means smaller loan with possibly different interest/eligibility profile.
- Modeling guidance:
  - Treat rare non-Y/N values as data-quality artifacts, not stable categories.
- Option A: canonical mapping policy
  - Keep Y and N as valid.
  - Map 0, S, C, A, R, and blanks to Unknown.
  - Keep true missing as Missing (or merge with Unknown).
- Option B: robust encoded feature set
  - LowDoc_clean categories: Y, N, Unknown, Missing.
  - LowDoc_is_nonstandard = 1 if original in 0/S/C/A/R.
  - LowDoc_is_missing = 1 for missing.
  - Avoid dropping rows (about 0.42% total) to reduce bias risk.
- Encoding by model type:
  - Tree models: keep categorical buckets directly (or one-hot).
  - Linear/logistic: one-hot LowDoc_clean + indicator flags.
- Validation and leakage control:
  - Fit all preprocessing inside CV folds only.
  - If using target encoding, do out-of-fold target encoding only.
  - Because rare buckets are tiny, avoid treating each as separate category unless strongly justified.
- Ablation checks:
  - Y/N only
  - Y/N/Unknown/Missing
  - Y/N + indicator flags
  - Select simplest version with stable CV performance and calibration.
- Option C practical default:
  - LowDoc_clean: Y, N, UnknownOrMissing
  - One flag: LowDoc_was_nonstandard_or_missing

### ChgOffDate

- Description: Date when a loan is declared in default.
- Common values and notes:
  - Date the loan was charged off (default/write-off date).
  - Missing values can mean no charge-off occurred.
- Open checks:
  - ChgOffDate - no existe.
- Modeling guardrail:
  - likely post-outcome information for approval prediction; evaluate leakage risk before use.

### DisbursementDate

- Description: Disbursement date.
- Common values and notes:
  - Date when funds were disbursed.
  - Useful for time-based features.
- Open checks:
  - date = approved?
  - does it represent anything at training time?
- Related feature engineering:
  - Disbursement_Delay / Time_To_Funds = DisbursementDate - ApprovalDate (days).

### DisbursementGross

- Description: Amount disbursed.
- Common values and notes:
  - Total loan amount disbursed.
  - Convert currency strings to numeric values before modeling.
- Risk insight:
  - Requested/disbursed amount is one of the strongest likely risk signals.
- Data quality note:
  - Excellent data quality with 20,767 valid values (99.99%).
  - 1 invalid value: a $0.00 disbursement (row 11414, IL).
- Open checks:
  - what to do with $0.00 row? delete? is this approved?
- Related feature engineering:
  - Gross_per_Employee
  - Loan_Size_Ratio
  - Job_Creation_Efficiency

### BalanceGross

- Description: Gross amount outstanding.
- Common values and notes:
  - Remaining gross balance outstanding.
  - Often zero in some subsets; still useful to inspect.
- Open checks:
  - are there correlations between having a previous balance and new loan approvals?
  - can a model see these connections?

### Accept

- Description: Loan approval status; 0 = Not approved, 1 = Approved.
- Common values and notes:
  - Target label for modeling.
- Modeling guidance:
  - Accept is the binary target label (0 or 1).
  - class 1 appears more frequent than class 0, so verify class imbalance on full dataset.

---

## Cross-Column Modeling Notes

### Borrower Profile Fields

- Name, City, State
- NoEmp
- NewExist
- FranchiseCode
- UrbanRural

These include non-standard values in some cases, so category cleaning is required.

### Loan and Process Fields

- Bank, BankState
- ApprovalDate, ApprovalFY
- DisbursementDate
- DisbursementGross
- BalanceGross

These include non-standard values in some cases, so category cleaning is required.

### Program and Policy Flags

- RevLineCr
- LowDoc

These include non-standard values in some cases, so category cleaning is required.

### Strongest Likely Risk Signals

- Business size and maturity (NoEmp, NewExist)
- Requested amount (DisbursementGross)
- Lender and program characteristics (Bank, RevLineCr, LowDoc)
- Geography and time (City/State, ApprovalFY)

### Essential Preprocessing

- Parse currency fields to numeric.
- Parse dates into features (year, month, age).
- Normalize categorical text (case, typos).
- Handle unusual categorical values and missing data.
- Drop pure identifiers.

---

## Feature Engineering (Consolidated)

### 1. Risk and Relationship Features

- Is_Local_Lender (Binary): compare State and BankState.
  - Logic: If State == BankState, it is a local loan. Local banks often have better qualitative insights into the business's community, which can correlate with different default rates compared to out-of-state big lenders.
- Is_Franchise (Binary): transform FranchiseCode.
  - Logic: If FranchiseCode is 0 or 1, set to 0 (Independent); otherwise 1 (Franchise). Franchises often have higher survival rates because they follow a proven business model.
- Gross_per_Employee (Numerical): DisbursementGross / (NoEmp + 1).
  - Logic: Measures loan intensity by business size. +1 avoids division by zero.

### 2. Time-Based Engineering

- Disbursement_Delay (Numerical): DisbursementDate - ApprovalDate in days.
  - Logic: Long delays might indicate administrative hurdles or changes in borrower condition.
- Is_Recession_Era (Binary): based on ApprovalFY.
  - Logic: Flag years within known downturn periods (example: 2008-2009).
- Approval_Quarter (Categorical): from ApprovalDate.
  - Logic: Banks may have varying lending appetite across fiscal periods.

### 3. Business Impact and Growth

- Total_Job_Impact (Numerical): CreateJob + RetainedJob.
  - Logic: Single metric for total economic footprint.
- Job_Creation_Efficiency (Numerical): DisbursementGross / (CreateJob + 1).
  - Logic: Capital required per created job.

### 4. Categorical Consolidation

- Bank_Market_Share (Numerical): frequency count of Bank in dataset.
  - Logic: Captures lender-size proxy and policy differences.
- State_Default_Rate (Numerical): target-encoding style feature on State.
  - Logic: Captures state-level approval differences and macro context.

### Additional Proposed Features

| New Feature | Sources | Type | Goal |
| --- | --- | --- | --- |
| Back_To_Back | State, BankState | Binary | Identify local vs. national lending dynamics. |
| Loan_Size_Ratio | DisbursementGross, NoEmp | Ratio | Normalize loan impact by business size. |
| Time_To_Funds | ApprovalDate, DisbursementDate | Days | Measure administrative/operational lag. |
| Is_New_Business | NewExist | Binary | Cleaned version: 1 if NewExist == 2, else 0. |
