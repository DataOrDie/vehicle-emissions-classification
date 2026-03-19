# About the Data

## Data Fields

| Field             | Type      | Description                                          |
| ----------------- | --------- | ---------------------------------------------------- |
| id                | Text      | Identifier of the data instance                      |
| LoanNr_ChkDgt     | Text      | Identifier of the loan petition                      |
| Name              | Text      | Borrower name                                        |
| City              | Text      | Borrower city                                        |
| State             | Text      | Borrower state                                       |
| Bank              | Text      | Bank name                                            |
| BankState         | Text      | Bank state                                           |
| ApprovalDate      | Date/Time | Date SBA commitment issued                           |
| ApprovalFY        | Text      | Fiscal year of commitment                            |
| NoEmp             | Number    | Number of business employees                         |
| NewExist          | Text      | 1 = Existing business, 2 = New business              |
| CreateJob         | Number    | Number of jobs created                               |
| RetainedJob       | Number    | Number of jobs retained                              |
| FranchiseCode     | Text      | Franchise code; 00000 or 00001 = no franchise        |
| UrbanRural        | Text      | 1 = Urban, 2 = Rural, 0 = Undefined                  |
| RevLineCr         | Text      | Revolving line of credit; Y = Yes, N = No            |
| LowDoc            | Text      | LowDoc loan program; Y = Yes, N = No                 |
| ChgOffDate        | Date/Time | Date when a loan is declared in default              |
| DisbursementDate  | Date/Time | Disbursement date                                    |
| DisbursementGross | Currency  | Amount disbursed                                     |
| BalanceGross      | Currency  | Gross amount outstanding                             |
| Accept            | Text      | Loan approval status; 0 = Not approved, 1 = Approved |

---

## Risk Assessment View (Bank Perspective)

Based on the data dictionary and train sample, these are the most critical columns to analyze for risk.

### 1. Business Maturity (NewExist)

Business age is one of the strongest predictors of survival. Startups (often coded as 2.0) have significantly higher failure rates than established businesses (1.0).

- Insight: New businesses often require stronger collateral or a stronger plan.

### 2. Capacity and Scale (NoEmp and DisbursementGross)

Debt capacity matters. Debt capacity matters. If a company with 1 employee (NoEmp) asks for $500,000 (DisbursementGross), the risk is astronomical compared to a 50-employee firm asking for the same amount.

- Insight: analyze the ratio of loan size to employee count as a proxy for the business's ability to generate the revenue needed for repayment.

### 3. Business Model Stability (FranchiseCode)

Franchises have proven operating models. Independent firms are often riskier.

- Insight: Codes like 00000 or 00001 usually mean non-franchise. Specific franchise codes may indicate lower operational uncertainty.

### 4. Economic Impact (CreateJob and RetainedJob)

In SBA-like lending, impact can matter in addition to pure credit risk. the bank isn't just looking at profit—they're looking at the mission.

- Insight: A loan that creates 10 jobs is more "attractive" to approve under certain government-guaranteed programs than one that creates zero, even if the risk profile is similar.

### 5. Loan Type and Process Complexity (RevLineCr and LowDoc)

- RevLineCr: (Revolving Line of Credit) These are like credit cards for businesses. High usage can signal cash-flow struggles..
- LowDoc: Lower-documentation processing can imply different risk and policy behavior.If a loan is "LowDoc" (Y) often means the loan is smaller but might have higher interest or specific eligibility rules.

### 6. Macro Conditions (ApprovalFY and UrbanRural)

A loan approved in 2006 (ApprovalFY) just before the 2008 crash has a very different context than one approved in 1996. Similarly, economic conditions in Urban vs. Rural areas (coded in UrbanRural) affect a business's customer base.

## Modeling Notes

### Target

- Accept is the binary target label (0 or 1).
- class 1 appears more frequent than class 0, so verify class imbalance on the full dataset.

### Identifier and Leakage-Prone Fields

- id: unique row identifier; do not use as predictive signal.
- LoanNr_ChkDgt: loan identifier; mostly useful for dedup checks.

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

## Data Dictionary (Simplified)

| Field             | Type      | Simple Description                                      | Common Values / Notes                                          |
| ----------------- | --------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| id                | Text      | Unique identifier for each row in the dataset.          | Usually used only to identify records, not as a model feature. |
| LoanNr_ChkDgt     | Text      | Unique loan application identifier.                     | Useful for tracking or deduplication checks.                   |
| Name              | Text      | Name of the business owner or borrower.                 | High-cardinality text; often not ideal for direct modeling.    |
| City              | Text      | Borrower's city.                                        | Location feature.                                              |
| State             | Text      | Borrower's state.                                       | Two-letter state code in many cases.                           |
| Bank              | Text      | Name of the bank that handled the loan.                 | Can capture lender-specific behavior.                          |
| BankState         | Text      | State where the bank is located.                        | May differ from borrower state.                                |
| ApprovalDate      | Date/Time | Date when the SBA commitment was approved.              | Can be converted into year/month/quarter features.             |
| ApprovalFY        | Text      | Fiscal year when the loan was approved.                 | Time and macroeconomic context feature.                        |
| NoEmp             | Number    | Number of employees in the business.                    | Proxy for business size.                                       |
| NewExist          | Text      | Whether the business is new or existing.                | `1 = Existing business`, `2 = New business`.                   |
| CreateJob         | Number    | Number of jobs expected to be created by the loan.      | Program impact indicator.                                      |
| RetainedJob       | Number    | Number of jobs expected to be retained.                 | Program impact indicator.                                      |
| FranchiseCode     | Text      | Code indicating franchise relationship.                 | `00000` or `00001` often means no franchise.                   |
| UrbanRural        | Text      | Area type where the business operates.                  | `1 = Urban`, `2 = Rural`, `0 = Undefined`.                     |
| RevLineCr         | Text      | Indicates if this loan is a revolving line of credit.   | `Y = Yes`, `N = No` (may include non-standard values).         |
| LowDoc            | Text      | Indicates if this is a low-documentation loan.          | `Y = Yes`, `N = No` (may include non-standard values).         |
| ChgOffDate        | Date/Time | Date the loan was charged off (default/write-off date). | Missing values can mean no charge-off occurred.                |
| DisbursementDate  | Date/Time | Date when funds were disbursed to the borrower.         | Useful for time-based features.                                |
| DisbursementGross | Currency  | Total loan amount disbursed.                            | Convert currency strings to numeric values before modeling.    |
| BalanceGross      | Currency  | Remaining gross balance outstanding.                    | Often zero in some subsets; still useful to inspect.           |
| Accept            | Text      | Target label: whether the loan was approved.            | `0 = Not approved`, `1 = Approved`.                            |

## Column-Specific Recommendations

## NewExist column

- Recode NewExist == 0.0 to missing (NaN), since docs define only 1 and 2.
- Create explicit "Unknown" category for missing values.
- One-hot encode NewExist (Existing, New, Unknown).
- Validate with cross-validation against a simpler baseline (drop those 13 rows) and keep whichever performs better.

Practical choice:

- Simplest: drop those rows.
- More robust: keep rows and add Unknown category.

## NoEmp column.

For NoEmp = 0, choose based on whether 0 is invalid or meaningful. depends on whether 0 means bad data or a real business state.

Option A: Treat 0 as invalid and drop those rows

- Use when 0 is clearly noise.
- Pro: cleaner signal.
- Con: you lose data (in your case only about 1.02%, so loss is small).

Option B: Treat 0 as missing and impute

- Replace 0 with NaN, impute (median/KNN/model-based), add NoEmp_was_zero flag.
- Pro: keeps rows and preserves suspicious-value signal.
- Con: imputation can blur patterns, blur true risk patterns

Option C: Keep 0 as meaningful state.

- Keep 0 If 0 could represent pre-operational firms/unknown staffing, keep it.
- Add explicit NoEmp = 0 bucket.
- Pro: model can learn if this group has different risk.
- Con: only works if 0 is truly meaningful.

Option D: Use Robust feature engineering:

- For skewed NoEmp, Use log1p(NoEmp) and/or bands (0, 1, 2-5, 6-10, 11-50, >50).
- For ratios like loan_per_employee, avoid division instability:
  - loan_per_employee = loan_amount / max(NoEmp, 1)
  - or set ratio missing when NoEmp = 0 and add NoEmp_is_zero flag.

### Model-family guidance:

- Tree models(XGBoost/LightGBM/CatBoost): indicator + raw or banded feature often works well.
- Linear models: imputation + indicator + transformation is usually better.

Suggested workflow:

1. Pipeline A: drop NoEmp = 0
2. Pipeline B: 0 to NaN + impute + NoEmp_is_zero column
3. Pipeline C: keep 0 + NoEmp_is_zero + NoEmp bands
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
- Keep true missing as Missing (or merge with Unknown if you prefer one bucket).

Option B: Use a robust encoded feature set

- Create LowDoc_clean with categories: Y, N, Unknown, Missing.
- Add LowDoc_is_nonstandard as a binary flag (1 if original value in 0/S/C/A/R).
- Add LowDoc_is_missing as a binary flag.
- Avoid dropping rows. These rows are only 0.42% total, but removing them can still bias data and reduce robustness.
  Usually better to keep them and let model learn if Unknown/Missing carries risk signal.

Encoding by model type:

- Tree models (XGBoost/LightGBM/CatBoost): keep Option A categorical buckets directly (or one-hot).
- Linear/logistic models: one-hot encode OptionB LowDoc_clean and include the indicator flags

Validation and leakage control:

- Fit all preprocessing inside CV folds only (pipeline).
- If you use target encoding for LowDoc, do out-of-fold target encoding only.
  Usually better to keep them and let model learn if Unknown/Missing carries risk signal.
  Choose encoding by model type
  Tree models (XGBoost/LightGBM/CatBoost): keep categorical buckets directly (or one-hot).
  Linear/logistic models: one-hot encode LowDoc_clean and include the indicator flags.
  Prevent leakage and instability
  Fit all preprocessing inside CV folds only (pipeline).
  If you use target encoding for LowDoc, do out-of-fold target encoding only.
- Because rare buckets are tiny, avoid treating each as separate category unless you have strong domain reason.
- Run an ablation check to compare:
  - Y/N only
  - Y/N/Unknown/Missing
  - Y/N + indicator flags
- Select the simplest version that gives stable CV performance and calibration.

Option C: Practical defaults

- LowDoc_clean: Y, N, UnknownOrMissing (single combined fallback bucket)
- One flag: LowDoc_was_nonstandard_or_missing

Then one-hot encode for linear models, or pass as category to tree models.

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

- Tree models: pass as categorical buckets (or one-hot).
- Linear/logistic models: one-hot RevLineCr_clean + include indicator flags. Avoid ordinal encoding (Y=1, N=0, UNKNOWN=2) unless explicitly justified.

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
