# AI-Based Manufacturing Efficiency Classification Using Sensor, Production, and 6G Network Data

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-unittest-0A9EDC)
![Internship evaluation](https://img.shields.io/badge/Internship%20evaluation-8%2F10-brightgreen)

A reproducible manufacturing analytics and machine-learning portfolio project covering current-state efficiency classification, machine-level performance, sensor and production telemetry, network reliability, explainability, operational monitoring and model validation across a supplied 100,000-record manufacturing dataset.

## Project Context

This project was originally completed with Unified Mentor Pvt. Ltd. as part of a Data Analyst Intern placement. It was selected from the Manufacturing project domain and structured around a Thales Group smart-manufacturing business scenario.

The project is presented here as a portfolio case study, not as evidence that Thales Group was the employer or client or that this was an official Thales engagement.

The current repository is an improved post-evaluation version of the original internship submission. Its manufacturing scope, dashboard and core analytical workflow are preserved, while class-imbalance handling, target diagnostics, temporal validation, feature ablation, reproducibility, automated testing and methodology documentation have been strengthened.

## Internship Project Evaluation

| Field | Original evaluation record |
|---|---|
| Status | Evaluated |
| Rating | 8/10 |
| Evaluation | Excellent |
| Program | Unified Mentor Pvt. Ltd. |
| Domain | Data Analyst Intern |
| Project category | Manufacturing |
| Submitted | 09 July 2026 |
| Evaluated | 5 August 2026 |

The evaluator described the original submission as a comprehensive machine-learning dashboard for monitoring and predicting manufacturing efficiency using sensor, production, quality, maintenance and network-related data.

The original Streamlit application classified machine performance into `Low`, `Medium` and `High` efficiency levels and provided both real-time simulation and historical analysis. Across the complete 100,000-record view, approximately 77.8% of records were classified as Low efficiency, 19.2% as Medium and 3.0% as High.

The evaluator specifically highlighted:

- machine-level trend analysis and efficiency rankings;
- operation-mode comparisons;
- historical heatmaps;
- network-reliability analysis;
- configurable alert thresholds;
- at-risk record identification;
- real-time machine-reading simulation.

The main improvement recommendation was to strengthen model evaluation and validation because the target is highly imbalanced. With nearly 77.8% of observations in the Low-efficiency class and only around 3% in the High-efficiency class, overall agreement or very high prediction confidence can give an overly optimistic impression of model performance.

## Improvements After Evaluation

The repository demonstrates the feedback loop **submission -> evaluation -> methodology review -> remediation -> validation**.

The post-evaluation version:

- adds majority-class and transparent business-rule baselines;
- evaluates balanced accuracy, Macro F1, Weighted F1 and per-class precision/recall;
- replaces shuffled validation with chronological holdout and expanding-window temporal validation;
- audits how `Efficiency_Status` was constructed;
- identifies the near-deterministic relationship with `Error_Rate_%` and `Production_Speed_units_per_hr`;
- adds feature-ablation experiments to test independent signal;
- evaluates probability quality and calibration;
- tests future-target feasibility and temporal continuity;
- rejects unsupported future-risk forecasting claims;
- interprets SHAP as current-state model attribution rather than causal evidence;
- adds automated sanity tests and reproducible root-level execution.

The methodology review found that the supplied efficiency label is almost entirely reconstructable from contemporaneous Error Rate and Production Speed. The current project therefore preserves the original benchmark results while interpreting the near-perfect tree-model performance as **current-state target-rule reconstruction**, not independent evidence of future predictive capability.

## Original Unified Mentor Submission Links

These are the four links recorded in the original Unified Mentor evaluation:

- [Code](https://github.com/mimohnaik-git/Thales-manufacturing-efficiency-ai)
- [Report](https://drive.google.com/file/d/1xoBXn6-7_pVrOxk9Sez4-gN4Sz1fsvpN/view?usp=sharing)
- [Live Project](https://drive.google.com/file/d/1w2WI37w7rnpBeDK0bAUgrp1eWRFb7qE3/view?usp=sharing)
- [Live Dashboard](https://thales-manufacturing-efficiency-ai-lbp48cgi3vy7hhyvsh4txp.streamlit.app/)

The links above preserve the historical internship submission. The current repository contains the later post-evaluation remediation and validation work.

---
## Project Structure

```text
Thales_Manufacturing_Efficiency_Project/
|
+-- 01_Reports/
|   +-- Thales_Manufacturing_Efficiency_Executive_Summary.docx
|   +-- Thales_Manufacturing_Efficiency_Research_Paper.docx
|
+-- 02_Streamlit_Dashboard/
|   +-- app/
|   |   +-- app.py
|   +-- data/
|   +-- models/
|   +-- outputs/
|       +-- diagnostics/
|       +-- metrics/
|       +-- validation/
|
+-- 03_Data/
|   +-- Thales_Group_Manufacturing.csv
|   +-- manufacturing_features.parquet
|
+-- 04_Source_Code/
|   +-- 01_eda.py
|   +-- 02_feature_engineering.py
|   +-- 03_modeling.py
|   +-- 04_explainability.py
|   +-- 05_streamlit_app.py
|   +-- config.py
|   +-- validation/
|
+-- 05_Figures/
+-- tests/
|   +-- test_project_sanity.py
|
+-- README.md
+-- requirements.txt
+-- .gitignore
```

---

## Business Objective

The project evaluates whether manufacturing efficiency can be classified using:

- production speed
- error rate
- quality-control defect rate
- temperature
- vibration
- power consumption
- predictive-maintenance score
- operation mode
- network latency
- packet loss
- engineered sensor-stability and network-reliability features

It also tests whether the supplied sequence contains enough temporal signal to support a defensible **future-efficiency prediction** use case.

---

## Dataset Summary

| Measure | Value |
|---|---:|
| Records | 100,000 |
| Machines | 50 |
| Raw columns | 14 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Observation period | 1 Jan 2025 - 10 Mar 2025 |

### Efficiency-status distribution

| Status | Records | Share |
|---|---:|---:|
| Low | 77,825 | 77.82% |
| Medium | 19,189 | 19.19% |
| High | 2,986 | 2.99% |

The target is highly imbalanced. A majority-class classifier already reaches approximately **78% accuracy**, so accuracy alone is not an adequate evaluation metric.

---

## Key Analytical Finding

The post-evaluation target audit found that `Efficiency_Status` is almost completely reconstructable from two contemporaneous variables:

- `Error_Rate_%`
- `Production_Speed_units_per_hr`

The recovered rule is:

```text
Low:
    Error_Rate_% > 5
    OR Production_Speed_units_per_hr < 200

High:
    Error_Rate_% <= 2
    AND Production_Speed_units_per_hr >= 400

Medium:
    otherwise
```

Across the full dataset, this rule disagrees with only **2 out of 100,000 observations**.

On the final chronological holdout, the rule reproduces the supplied target perfectly.

### Interpretation

The near-perfect Random Forest and XGBoost results are therefore best interpreted as **current-state target-rule reconstruction**, not as evidence that the models independently discovered future manufacturing behavior.

---

## Final Model Benchmark

The final benchmark uses a chronological **80% development / 20% holdout** split.

| Method | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Majority Baseline | 0.77995 | 0.33333 | 0.29212 | 0.68353 |
| Transparent Business Rule | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| Logistic Regression | 0.88150 | 0.90009 | 0.82012 | 0.88866 |
| Random Forest | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| XGBoost | 0.99755 | 0.99756 | 0.99680 | 0.99755 |

### Final interpretation

- **Transparent Business Rule** - primary operational classification method for this dataset
- **Random Forest** - highest final-holdout Macro F1 among the three fixed ML benchmarks
- **XGBoost** - near-perfect benchmark with strong probability-quality metrics
- **Logistic Regression** - useful lower-complexity benchmark

The three ML models are treated as fixed benchmarks. The final holdout is used for descriptive evaluation, not to select a production model. Their scores are not presented as proof of independent future predictive capability.

---

## Imbalance-Aware Evaluation

The final modeling workflow includes:

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- Macro Precision
- Macro Recall
- Per-class Precision
- Per-class Recall
- High-class recall
- Confusion matrices
- Macro ROC-AUC
- Macro Average Precision
- Log Loss
- Multiclass Brier Score
- Expected Calibration Error

This directly addresses the evaluator's concern that headline accuracy could be misleading on the imbalanced target.

---

## Feature Ablation

Feature ablation was used to measure how strongly model performance depends on variables that directly or mathematically encode the target rule.

| Feature set | Macro F1 | Balanced Accuracy | High Recall |
|---|---:|---:|---:|
| All features | 1.0000 | 1.0000 | 1.0000 |
| Remove direct rule features | 0.8221 | 0.9230 | 0.9787 |
| Remove all rule-linked features | 0.3314 | 0.3320 | 0.0393 |
| Sensor + network only | 0.3314 | 0.3320 | 0.0393 |

Once the direct and derived rule-linked features are removed, balanced performance falls to approximately chance level.

This is the strongest evidence that the original near-perfect ML results are driven primarily by target-rule reconstruction.

---

## Temporal Validation

The original shuffled cross-validation approach was replaced with **expanding-window temporal validation**.

| Model | Mean Macro F1 | Mean Balanced Accuracy |
|---|---:|---:|
| Logistic Regression | 0.8168 | 0.8945 |
| Random Forest | 0.9991 | 0.9986 |
| XGBoost | 0.9964 | 0.9962 |

The benchmark remains stable across chronological folds, but the target audit is essential for interpreting why the scores are so high.

---

## Future-Prediction Feasibility

A separate temporal-signal diagnostic tested whether the current machine state contains meaningful information about the next observed efficiency state.

| Diagnostic | Result |
|---|---:|
| Observed same-status rate | 0.6427 |
| Expected agreement under independence | 0.6434 |
| Agreement above chance | -0.0006 |
| Cohen's kappa | -0.0018 |
| Mutual information | ~0 |

Within-machine lag-1 autocorrelation for the major sensor, production, maintenance, and network variables is also approximately zero.

### Conclusion

The supplied dataset does **not** support a defensible future-efficiency forecasting claim.

The final project therefore focuses on:

- current-state efficiency classification
- descriptive manufacturing analytics
- model validation
- explainability
- operational monitoring

rather than predictive maintenance or future-risk forecasting.

---

## Explainability

SHAP analysis of the XGBoost benchmark identifies the strongest model-attribution features as:

1. `Error_Rate_%`
2. `Production_Speed_units_per_hr`
3. `Error_to_Output_Ratio`
4. `Quality_Adjusted_Error`
5. `Energy_Efficiency_Ratio`

These findings are consistent with the target-rule and feature-ablation analyses.

SHAP outputs are interpreted as explanations of **current-state model behavior**, not causal effects and not future forecasts.

---

## Streamlit Dashboard

The Streamlit dashboard contains five modules.

### 1. Current-State Classification

- Transparent Business Rule
- Logistic Regression
- Random Forest
- XGBoost
- what-if machine input
- filtered-batch classification agreement

### 2. Machine Insights

- machine-level efficiency profiles
- historical efficiency trends
- machine rankings
- summary tables

### 3. Explainability

- SHAP global feature importance
- Random Forest feature importance
- XGBoost feature importance
- selected-record explanations

### 4. Operational Monitoring

- operation-mode comparison
- network-reliability analysis
- threshold-based operational flags
- production, sensor, and network summaries

### 5. Model Validation

- target imbalance
- majority baseline
- transparent-rule baseline
- benchmark comparison
- temporal validation
- feature ablation
- temporal-signal limitations

Run the dashboard from the project root:

```powershell
streamlit run "02_Streamlit_Dashboard\app\app.py"
```

---

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Reproduce the Analysis

Run the analytical pipeline from the project root:

```powershell
python "04_Source_Code\01_eda.py"
python "04_Source_Code\02_feature_engineering.py"
python "04_Source_Code\03_modeling.py"
python "04_Source_Code\04_explainability.py"
```

Run the validation diagnostics:

```powershell
python "04_Source_Code\validation\data_quality.py"
python "04_Source_Code\validation\target_diagnostics.py"
python "04_Source_Code\validation\temporal_validation.py"
python "04_Source_Code\validation\future_target_diagnostics.py"
python "04_Source_Code\validation\temporal_signal_diagnostics.py"
```

Run the automated sanity suite:

```powershell
python -m unittest discover -s tests -v
```

Expected result:

```text
Ran 12 tests in ...
OK
```

---


## Deliverables

1. **Research Paper**
   `01_Reports/Thales_Manufacturing_Efficiency_Research_Paper.docx`

2. **Executive Summary**
   `01_Reports/Thales_Manufacturing_Efficiency_Executive_Summary.docx`

3. **Interactive Streamlit Dashboard**
   `02_Streamlit_Dashboard/app/app.py`

4. **Reproducible Analysis Pipeline**
   `04_Source_Code/`

5. **Validation & Diagnostic Outputs**
   `02_Streamlit_Dashboard/outputs/`

6. **Automated Sanity Tests**
   `tests/test_project_sanity.py`

---

## Tech Stack

Python  |  pandas  |  NumPy  |  scikit-learn  |  XGBoost  |  SHAP  |  Streamlit  |  Plotly  |  matplotlib  |  seaborn  |  PyArrow  |  joblib

---

## Project Limitations

The final project does **not** claim that:

- sensor or network variables causally drive manufacturing efficiency
- benchmark model probabilities represent future operational risk
- the supplied observations contain sufficient temporal continuity for predictive-maintenance forecasting
- near-perfect Random Forest or XGBoost scores prove independent future predictive power

The strongest supported use cases are **current-state classification, descriptive manufacturing analytics, model validation, explainability, and operational monitoring**.

---

## Portfolio Takeaway

The strongest result in this project is not the 100% Random Forest score.

It is the validation process that explains **why** that score occurs.

By auditing the target, testing class imbalance, using chronological validation, removing rule-linked features, and checking temporal signal, the project demonstrates the difference between:

**a model that scores well**

and

**an analysis that supports a defensible business conclusion**.
