# Loan Approval Explainable AI (XAI) App

## Project title
Explainable AI for Data-Driven Decisions: Explainability for Deep Learning in High-Stakes Domains via LIME Local Surrogate Model Explanations, Applied to Loan/Credit Approval Explainability for Regulatory Compliance.

## Features
- Streamlit web application that runs locally in a browser.
- MLP neural network classifier.
- Logistic Regression baseline for comparison.
- LIME explanation for an individual applicant.
- Accuracy, precision, recall, F1-score, confusion matrix, and classification report.
- Downloadable LIME explanation summary.

## Important dataset note
`loan_prediction.csv` is synthetic educational data. It is not real bank data, and its labels do not represent any lender's policy. This app is for academic demonstration only, not real lending decisions or proof of regulatory compliance.

## Run in VS Code on Windows
1. Extract the ZIP and open the `Loan_Approval_XAI` folder in VS Code.
2. Select **Terminal > New Terminal**.
3. Run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
4. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## macOS/Linux
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Dataset columns
Features: `age`, `annual_income`, `credit_score`, `loan_amount`, `employment_years`, `debt_to_income`, `existing_loans`, `previous_defaults`, `savings`.
Target: `Loan_Status` (`Approved` or `Rejected`).

For a real research project, use a lawful, documented dataset, verify its permitted use, update feature/target names and preprocessing as needed, and evaluate fairness and subgroup performance.
