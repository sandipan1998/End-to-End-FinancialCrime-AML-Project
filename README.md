# End-to-End-FinancialCrime-AML-Project
# US AML Transaction Monitoring & Financial Crime Analytics Platform

## Project Overview

This project demonstrates an end-to-end synthetic financial-crime transaction-monitoring platform designed around a US financial institution use case.

The platform combines customer profiling, transaction monitoring, AML red-flag detection, machine-learning risk scoring, SHAP explainability, alert prioritization, analyst investigation, interactive dashboarding and AI-assisted SAR narrative drafting.

## Architecture

Customer / KYC Data
-> Transaction Monitoring
-> AML Rules
-> Feature Engineering
-> Machine Learning
-> SHAP Explainability
-> Risk Scoring
-> Alert Prioritization
-> Investigation Case Management
-> AML Dashboard
-> SAR Narrative Assistant

## AML Typologies Covered

- Structuring
- Rapid movement
- Funnel-account behavior
- Layering
- Mule-account behavior
- Fraud-linked suspicious activity
- High-value transaction activity
- Customer-profile mismatch
- International transaction activity
- High transaction velocity
- Multiple-counterparty behavior

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- SHAP
- Streamlit
- Plotly
- Excel / CSV
- Machine Learning
- Explainable AI

## Key AML Capabilities

- Rule-based transaction monitoring
- Behavioral feature engineering
- Customer risk profiling
- Machine-learning risk scoring
- Alert prioritization
- False-positive analysis
- Customer-level investigation
- Red-flag identification
- SHAP explainability
- Interactive investigation dashboard
- Analyst case management
- AI-assisted SAR narrative drafting
- Human-in-the-loop review

## Important Limitation

This project uses synthetic data for educational and portfolio demonstration purposes. The rules, thresholds and model outputs should not be interpreted as production financial-crime detection performance or legal conclusions.

The SAR component produces analyst-reviewable drafts and does not automatically determine or file a SAR.

## Future Production Improvements

- Use institution-specific historical alert and transaction data
- Calibrate thresholds using historical investigation outcomes
- Implement champion/challenger model testing
- Add model drift monitoring
- Add data-quality monitoring
- Add investigator feedback loops
- Add graph-based transaction-network analytics
- Add sanctions screening integration
- Add adverse-media intelligence
- Add case-management integration
- Perform formal model validation
- Perform fairness and bias assessment
- Implement secure enterprise deployment
- Integrate approved LLM infrastructure for controlled SAR drafting

# Summary: 
## I developed an end-to-end synthetic US AML transaction-monitoring project to understand how financial institutions can identify and prioritize potentially suspicious customer activity. I generated customer, account, and transaction data, then implemented rule-based detection for patterns such as structuring, rapid movement, funnel accounts, layering, and mule behavior.
## I engineered customer-level and transaction-level behavioral features over 24-hour, 7-day, and 30-day periods. I compared Logistic Regression, XGBoost, and CatBoost models using a time-based split and used SHAP to explain the model's risk drivers.
## After generating risk scores, I developed an alert-prioritization and investigation workflow that includes customer profiles, transaction reviews, red flags, analyst disposition fields, and SAR narrative drafts. The SAR drafts are only assistance for human review; the project does not automatically file SARs or make legal determinations.
