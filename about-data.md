What the data contains:

Data fields

    id # Text # Identifier of the data instance
    LoanNr_ChkDgt # Text # Identifier of the loan petition
    Name # Text # Borrower name
    City # Text # Borrower city
    State # Text # Borrower state
    Bank # Text # Bank name
    BankState # Text # Bank state
    ApprovalDate # Date/Time # Date SBA commitment issued
    ApprovalFY # Text # Fiscal year of commitment
    NoEmp # Number # Number of business employees
    NewExist # Text # 1 = Existing business, 2 = New business
    CreateJob # Number # Number of jobs created
    RetainedJob # Number # Number of jobs retained
    FranchiseCode # Text # Franchise code, (00000 or 00001) = No franchise
    UrbanRural # Text # 1 = Urban, 2 = rural, 0 = undefined
    RevLineCr # Text # Revolving line of credit: Y = Yes, N = No
    LowDoc # Text # LowDoc Loan Program: Y = Yes, N = No
    ChgOffDate # Date/Time # The date when a loan is declared to be in default
    DisbursementDate # Date/Time # Disbursement date
    DisbursementGross # Currency # Amount disbursed
    BalanceGross # Currency # Gross amount outstanding
    Accept # Text # Loan approval status. 0 = not approved, 1 = approved

---

To help you think like a bank representative, let’s break down the most critical columns from a risk assessment perspective, based on
the about-data.md and the 100-line sample of train.csv.

1. Business Maturity (NewExist)

- Why it matters: In banking, the "age" of a business is one of the strongest predictors of survival. Startups (often coded as 2.0)  
  have significantly higher failure rates than established businesses (1.0).
- Insight: If Company X is a new business, you'd look for higher collateral or a stronger business plan.

2. Capacity and Scale (NoEmp & DisbursementGross)

- Why it matters: This is about "Debt Capacity." If a company with 1 employee (NoEmp) asks for $500,000 (DisbursementGross), the risk
  is astronomical compared to a 50-employee firm asking for the same amount.
- Insight: You'll want to analyze the ratio of loan size to employee count as a proxy for the business's ability to generate the
  revenue needed for repayment.

3. Business Model Stability (FranchiseCode)

- Why it matters: Franchises have a "proven" success model. A FranchiseCode of 0 or 1 usually means it's an independent small
  business, while a specific code (like 10481 for Big Apple Bagels in your sample) indicates a franchise.
- Insight: Independent businesses are generally riskier than franchises, which have corporate support and standardized operations.

4. Economic Impact (CreateJob & RetainedJob)

- Why it matters: Since these are likely SBA-backed loans (implied by the columns and about-data.md), the bank isn't just looking at
  profit—they're looking at the mission.
- Insight: A loan that creates 10 jobs is more "attractive" to approve under certain government-guaranteed programs than one that
  creates zero, even if the risk profile is similar.

5. Loan Type and Complexity (RevLineCr & LowDoc)

- Why it matters:
  - RevLineCr (Revolving Line of Credit): These are like credit cards for businesses. High usage can signal cash-flow struggles.
  - LowDoc (Low Documentation): If a loan is "LowDoc" (Y), it was processed with less paperwork. This often means the loan is  
    smaller but might have higher interest or specific eligibility rules.

6. Macro-Conditions (ApprovalFY & UrbanRural)

- Why it matters: A loan approved in 2006 (ApprovalFY) just before the 2008 crash has a very different context than one approved in 1996. Similarly, economic conditions in Urban vs. Rural areas (coded in UrbanRural) affect a business's customer base.

Strategic Question for your Analysis:
When you start coding, I suggest looking at the Default Rate (where Accept = 0) across these categories:

1.  Is the default rate significantly higher for New Businesses (NewExist=2)?
2.  Does a high DisbursementGross correlate with more defaults for small NoEmp counts?
3.  Are LowDoc loans actually safer or riskier in this specific dataset?

---

Accept: This is the Target label. binary outcome (0/1).
In this preview, class 1 appears much more frequent than class 0, so you should check class imbalance on the full file.

Identifier / leakage-prone fields:

id: unique row id (do not use as predictive signal).

LoanNr_ChkDgt: loan identifier, usually not useful except maybe for dedup checks.

Borrower profile fields:
Name, City, State
NoEmp (number of employees)
NewExist (existing business vs new business)
FranchiseCode
UrbanRural
Loan and process fields:
Bank, BankState
ApprovalDate, ApprovalFY
DisbursementDate
DisbursementGross (currency string)
BalanceGross (currency string; often 0 in this sample)

Job impact fields:
CreateJob
RetainedJob
Program/credit policy flags:
RevLineCr
LowDoc
These are categorical flags with values like Y/N and some non-standard values (for example 0, T, S), so category cleaning is needed.

What this means for your challenge (grant or deny loan):

You can frame risk as P(Accept = 1) or P(deny), depending on how the competition defines positive class.

The strongest likely risk signals will come from:

- business size and maturity (NoEmp, NewExist),
- requested amount (DisbursementGross),
- lender/program characteristics (Bank, RevLineCr, LowDoc),
- geography and time (City/State, ApprovalFY).

Before modeling, do essential preprocessing:
parse currency columns into numeric,
parse dates into time features (year, month, loan age),
normalize categorical text (case/typos),
handle unusual categorical codes and missing values,
drop pure IDs.

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

# Questions

### NewExisting Column.

Recode NewExist == 0.0 to missing (NaN), since docs define only 1 and 2.
Create an explicit "Unknown" category for missing values.
One-hot encode NewExist (Existing, New, Unknown).
Validate with cross-validation against a simpler baseline (drop those 13 rows) and keep whichever performs better.
If you want the simplest pipeline: drop those 13 rows.
If you want a production-robust pipeline: keep rows and add Unknown category (recommended).

### NoEmp column.

For training, NoEmp = 0 can be handled in a few solid ways. Best choice depends on whether 0 means bad data or a real business state.

- Treat 0 as invalid and drop those rows
  Use when 0 is impossible/noise.
  Pro: cleanest signal.
  Con: you lose data (in your case only about 1.02%, so loss is small).
- Treat 0 as missing and impute
  Replace 0 with NaN, then impute (median, KNN, or model-based).
  Add a flag feature like NoEmp_was_zero.
  Pro: keeps rows and preserves “was suspicious” information.
  Con: imputation can blur true risk patterns.
- Keep 0 as a separate meaningful state
  If 0 could represent pre-operational firms/unknown staffing, keep it.
  Add engineered bands with explicit category NoEmp=0.
  Pro: model can learn if this group has different risk.
  Con: only works if 0 is truly meaningful.
- Use robust transforms/features
  For skewed NoEmp, use log1p(NoEmp) and/or bands (0, 1, 2-5, 6-10, 11-50, >50).
  If using ratios like loan per employee, protect division with:
  loan_per_employee = loan_amount / max(NoEmp, 1) or
  set ratio missing when NoEmp=0 and add NoEmp_is_zero flag.
  Pro: avoids exploding ratios and improves stability.

Note: Model-family-specific handling

- Tree models (XGBoost/LightGBM/CatBoost): often do well with indicator + raw/banded feature.
- Linear models: usually better with imputation + indicator + transformation.
  Recommended practical workflow:

* Build 3 pipelines and compare by cross-validation:
  A: drop NoEmp=0
  B: 0 -> NaN + impute + NoEmp_is_zero
  C: keep 0 + add NoEmp_is_zero + NoEmp bands
  Compare AUC/F1/PR-AUC and calibration.
  Pick the simplest strategy that performs best and is easiest to explain.
  Given your data (small share of zeros), a strong baseline is:

- NoEmp_is_zero flag
- NoEmp_clean = NoEmp.replace(0, np.nan) with median imputation
- NoEmp_band feature for nonlinearity

### LowDoc column

Since LowDoc is essentially a binary program flag (Y/N), those rare values should be treated as data quality artifacts, not real stable categories.

Recommended strategy for modeling:

Define a canonical mapping policy
Keep Y and N as valid.
Convert 0, S, C, A, R and blanks to Unknown.
Keep true missing as Missing (or merge with Unknown if you prefer one bucket).
Use a robust encoded feature set
Create LowDoc_clean with categories: Y, N, Unknown, Missing.
Add LowDoc_is_nonstandard as a binary flag (1 if original value in 0/S/C/A/R).
Add LowDoc_is_missing as a binary flag.
Avoid dropping rows
These rows are only 0.42% total, but removing them can still bias data and reduce robustness.
Usually better to keep them and let model learn if Unknown/Missing carries risk signal.
Choose encoding by model type
Tree models (XGBoost/LightGBM/CatBoost): keep categorical buckets directly (or one-hot).
Linear/logistic models: one-hot encode LowDoc_clean and include the indicator flags.
Prevent leakage and instability
Fit all preprocessing inside CV folds only (pipeline).
If you use target encoding for LowDoc, do out-of-fold target encoding only.
Because rare buckets are tiny, avoid treating each as separate category unless you have strong domain reason.
Run an ablation check
Compare:
Y/N only (non-standard -> missing)
Y/N/Unknown/Missing
Y/N + indicator flags
Select the simplest version that gives stable CV performance and calibration.
Practical default I’d use:

LowDoc_clean: Y, N, UnknownOrMissing (single combined fallback bucket)
Plus one flag: LowDoc_was_nonstandard_or_missing
Then one-hot encode for linear models, or pass as category to tree models.

### RevLineCr column

For RevLineCr, the key difference vs LowDoc is that non-standard values are not rare, mainly because 0 is 24.3% of the dataset. That means you should not just drop them.

Practical strategies for modeling:

Treat this as a data-definition problem first
Check data dictionary or source docs to confirm what 0 and T mean.
If documentation confirms 0 is equivalent to N (or unknown), map accordingly.
If no reliable documentation, do not force-map 0 to Y or N blindly.
Use a safe fallback encoding
Build RevLineCr_clean with categories:
Y
N
UNKNOWN (for 0, T, Q)
MISSING
Because 0 is large, keeping it visible (as UNKNOWN or as its own bucket) is safer than deleting.
Add quality indicator features
RevLineCr_is_nonstandard: 1 for 0/T/Q, else 0.
RevLineCr_is_missing: 1 for missing, else 0.
These flags often help models capture “data quality signal.”
Pick encoding by model family
Tree models: pass as categorical buckets (or one-hot).
Linear/logistic: one-hot RevLineCr_clean + include indicator flags.
Avoid ordinal encoding (Y=1, N=0, UNKNOWN=2) unless explicitly justified.
Run policy sensitivity tests
Compare CV performance for:
Policy A: 0/T/Q -> UNKNOWN
Policy B: 0 -> N, T/Q -> UNKNOWN
Policy C: keep 0 as separate category, T/Q grouped
Choose the most stable option (mean + variance across folds), not just best single score.
Keep preprocessing inside CV pipeline
Fit encoders/imputers within each fold only.
If using target encoding, use out-of-fold encoding only to avoid leakage.
Recommended default starting point:

RevLineCr_clean: Y/N/UNKNOWN/MISSING
Keep 0 inside UNKNOWN at first
Add two flags: nonstandard and missing
Validate with ablation; if mapping 0->N consistently improves and is stable, then adopt it
