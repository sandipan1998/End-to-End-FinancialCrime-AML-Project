# End-to-End-FinancialCrime-AML-Project
# US AML Transaction Monitoring & Financial Crime Analytics Platform

## Project Overview

This project demonstrates an end-to-end synthetic financial-crime transaction-monitoring platform designed around a US financial institution use case.

The platform combines customer profiling, transaction monitoring, AML red-flag detection, machine-learning risk scoring, SHAP explainability, alert prioritization, analyst investigation, interactive dashboarding and AI-assisted SAR narrative drafting.

┌──────────────────────────────────────────────────────────────┐
│             US AML / FINANCIAL CRIME PROJECT                 │
│        Synthetic Financial Institution Transaction Data      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 1 — DATA GENERATION                                    │
│                                                              │
│ Customers • Accounts • Transactions • KYC Risk Indicators    │
│ PEP • Sanctions • Adverse Media • Expected Activity          │
│ AML Typologies • Suspicious Transaction Scenarios            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 2 — EXPLORATORY DATA ANALYSIS                          │
│                                                              │
│ Data Quality • Customer Risk • Transaction Patterns          │
│ Suspicious Activity • Countries • Channels • Red Flags       │
│ Rule Alerts • Risk Distribution                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 3 — AML RULE ENGINE                                    │
│                                                              │
│ Structuring • High Value • Velocity • Rapid Movement         │
│ International • Cash • PEP • Sanctions • Adverse Media      │
│ Mule • Layering • Funnel • Multiple Counterparties           │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 4 — FEATURE ENGINEERING                                │
│                                                              │
│ Transaction Features • Customer Behavior • Velocity          │
│ 24H / 7D / 30D Activity • Country & Channel Features         │
│ Structuring • Rapid Movement • Risk & Red-Flag Features      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 5 — MACHINE LEARNING                                   │
│                                                              │
│ Logistic Regression • XGBoost • CatBoost                     │
│ Time-Based Train/Test Split                                  │
│ Precision • Recall • F1 • ROC-AUC • PR-AUC                  │
│ False Positive / False Negative Analysis                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 6 — SHAP EXPLAINABILITY                                │
│                                                              │
│ Global Feature Importance                                    │
│ Transaction-Level Explanations                               │
│ Positive / Negative Risk Drivers                              │
│ Model Interpretability                                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 7 — AML RISK SCORING & ALERT PRIORITIZATION            │
│                                                              │
│ ML Risk Score + Rule Score                                   │
│ Risk Level: LOW / MEDIUM / HIGH / CRITICAL                  │
│ Alert Priority: P1 / P2 / P3 / P4                            │
│ Analyst Investigation Queue                                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 8 — AML INVESTIGATION & CASE MANAGEMENT                │
│                                                              │
│ Customer Profile Review                                      │
│ Transaction Review                                           │
│ Red-Flag Identification                                       │
│ SHAP Drivers                                                 │
│ Analyst Disposition • Escalation • Case Status               │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 9 — INTERACTIVE AML INVESTIGATION DASHBOARD            │
│                                                              │
│ Case Search • Customer Profile • Risk Signals                │
│ Transaction Timeline • Red Flags • SHAP Drivers              │
│ Analyst Review • Disposition • Investigation Notes            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 10 — SAR NARRATIVE ASSISTANT                           │
│                                                              │
│ Who • What • When • Where • Why                              │
│ Chronological Suspicious Activity Narrative                   │
│ Evidence-Based Draft                                         │
│ Human Review Required                                        │
│ Automatic SAR Filing = DISABLED                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ PHASE 11 — MODEL & PROJECT VALIDATION                         │
│                                                              │
│ Dataset Integrity • ML Performance                            │
│ Error Analysis • Scenario Validation                          │
│ Rule Validation • Feature Leakage Review                      │
│ Investigation Workflow • SAR Validation                       │
│ SHAP Validation • Model Governance                            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   PORTFOLIO OUTPUT    │
                    │                       │
                    │ README                │
                    │ Validation Reports    │
                    │ AML Reports           │
                    │ Models & Explainability│
                    │ Dashboard             │
                    └───────────────────────┘

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


## I developed an AML transaction-monitoring pipeline that combines rule-based detection, machine-learning risk scoring and SHAP explainability, then converts the alerts into customer-level investigation cases with risk prioritization, red-flag analysis and analyst disposition workflows.

 ### I developed a human-in-the-loop AI-assisted SAR narrative generation workflow that converts structured AML investigation evidence into a chronological draft covering who, what, when, where and why, while preserving analyst control over the final SAR decision.

 ## I built an end-to-end AML transaction monitoring and investigation platform using synthetic US financial-institution data. The platform combines behavioral AML rules, machine-learning risk scoring, SHAP explainability, alert prioritization, customer-level investigations, and an AI-assisted SAR narrative workflow