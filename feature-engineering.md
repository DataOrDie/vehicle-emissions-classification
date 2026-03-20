# Feature Engineering

## 1. Risk & Relationship Features

These features help the model understand the "distance" and "trust" between the bank and the business.

- **Is_Local_Lender** (Binary): Compare `State` and `BankState`.
  - **Logic:** If `State == BankState`, it is a local loan. Local banks often have better qualitative insights into the business's community, which can correlate with different default rates compared to out-of-state "big box" lenders.

- **Is_Franchise** (Binary): Transform `FranchiseCode`.
  - **Logic:** If `FranchiseCode` is `0` or `1`, set to `0` (Independent); otherwise `1` (Franchise). Franchises often have higher survival rates because they follow a proven business model, making them lower risk.

- **Gross_per_Employee** (Numerical): `DisbursementGross / (NoEmp + 1)`.
  - **Logic:** This measures the "intensity" of the loan. A $500k loan for a 2-person shop is a very different risk profile than a $500k loan for a 50-person company. (`+1` avoids division by zero.)

## 2. Time-Based Engineering

The gap between approval and disbursement tells a story about the urgency and efficiency of the business and the bank.

- **Disbursement_Delay** (Numerical): `DisbursementDate - ApprovalDate` (in days).
  - **Logic:** Long delays might indicate administrative hurdles or a change in the borrower's financial health during the underwriting process.

- **Is_Recession_Era** (Binary): Based on `ApprovalFY`.
  - **Logic:** Create a flag if the fiscal year falls within a known economic downturn (for example, the 2008-2009 Great Recession). A business that survived or started during a recession shows significant resilience.

- **Approval_Quarter** (Categorical): Extract the quarter from `ApprovalDate`.
  - **Logic:** Banks often have quotas or different lending appetites depending on whether it is the start or the end of their fiscal year.

## 3. Business Impact & Growth

Merging the job-related columns provides a better picture of the loan's productivity.

- **Total_Job_Impact** (Numerical): `CreateJob + RetainedJob`.
  - **Logic:** Instead of looking at jobs created and retained separately, this gives a single metric for the total economic footprint of the loan.

- **Job_Creation_Efficiency** (Numerical): `DisbursementGross / (CreateJob + 1)`.
  - **Logic:** How much capital is required to create one job? High efficiency might indicate a more scalable business model.

## 4. Categorical Consolidation (Merging)

High-cardinality columns like `Bank` or `City` can overwhelm a model.

- **Bank_Market_Share** (Numerical): Replace the `Bank` name with the frequency count of that bank in your dataset.
  - **Logic:** Large banks (JPMorgan, Wells Fargo) have different risk appetites than small community banks. This transforms a text name into a lender-size proxy.

- **State_Default_Rate** (Numerical): A form of target encoding. Replace `State` with the historical approval rate for that state.
  - **Logic:** Certain states may have higher success rates due to local economic conditions or state-level small business support programs.

## Additional Proposed Features

| New Feature     | Sources                        | Type   | Goal                                               |
| --------------- | ------------------------------ | ------ | -------------------------------------------------- |
| Back_To_Back    | State, BankState               | Binary | Identify local vs. national lending dynamics.      |
| Loan_Size_Ratio | DisbursementGross, NoEmp       | Ratio  | Normalize loan impact by business size.            |
| Time_To_Funds   | ApprovalDate, DisbursementDate | Days   | Measure administrative/operational lag.            |
| Is_New_Business | NewExist                       | Binary | Cleaned version: `1` if `NewExist == 2`, else `0`. |
