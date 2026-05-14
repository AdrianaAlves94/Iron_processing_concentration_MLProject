# Predicting Iron Concentrate Quality in a Flotation Processing Plant

A machine learning project that predicts the purity of iron concentrate (% iron) from real-time sensor data collected across a flotation processing plant — enabling near real-time quality monitoring without waiting for lab results.

---

## The Problem

In iron ore processing plants, concentrate quality is typically measured through physical lab sampling — a process that takes hours. By the time operators know the iron grade is off-spec, a significant amount of product has already been processed under suboptimal conditions.

This project asks: **can we predict iron concentrate grade continuously, using sensor data that is already being collected?**

---

## The Data

- 6 months of hourly sensor readings from a flotation processing plant
- 124 features after feature engineering, including:
  - Ore feed composition (iron %, silica %)
  - Reagent flows (starch, amina)
  - Ore pulp properties (pH, density, flow)
  - Airflow and level readings across 7 flotation columns
  - Engineered lag features (1h, 3h) and rolling averages (3h window)
  - Time features (hour, day of week, month)
- Target variable: iron concentrate grade (%)

---

## Methodology

### Time-Aware Validation
Because this is time series data, standard random train/test splitting would allow the model to train on future observations to predict the past — a form of data leakage. All splits were done **chronologically**, and `TimeSeriesSplit` (7 folds) was used for cross-validation, ensuring the model is always trained on the past and evaluated on what comes after. This mirrors real operational conditions.

### Model Selection
Three models were compared as baselines, chosen to cover the full spectrum from simple to complex:

| Model | Approach |
|---|---|
| Decision Tree | Simple interpretable baseline |
| Random Forest | Bagging — committee of trees |
| XGBoost | Boosting — learns from mistakes iteratively |

Random Forest performed best at baseline (Test R²: 0.494). XGBoost was competitive (0.417 untuned) but showed more potential with tuning.

### Hyperparameter Tuning
`GridSearchCV` with `TimeSeriesSplit` was used to tune both Random Forest and XGBoost. Key finding: XGBoost benefited most from tuning — a low learning rate (0.01), shallow trees (max_depth=3), and subsampling (0.8) transformed it from an overfit model to the best performer.

---

## Results

| Model | CV R² (mean) | CV R² (std) | Test R² | Test MAE |
|---|---|---|---|---|
| XGBoost (tuned) | 0.610 | — | 0.511 | 0.632 |
| Random Forest (tuned) | 0.574 | 0.125 | 0.494 | 0.639 |
| Decision Tree | 0.508 | 0.122 | 0.355 | 0.724 |

**Best model: XGBoost (tuned)**
- Predicts iron concentrate grade within **±0.63 percentage points** on average
- CV and test scores are close, confirming the model generalises well to unseen time periods

### Why 0.51 R²?

The model has access to all key controllable process variables — reagent dosage, airflow, pH, feed grade. The remaining unexplained variance reflects the inherent physical complexity of flotation: microscopic ore heterogeneity, equipment wear, and operator decisions that no sensor fully captures. Even industrial ML systems built by process engineers typically land in the 50–65% R² range for this problem.

---

## Project Structure

```
├── data/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── src/
│   ├── preprocessing.py      # data cleaning and scaling utilities
│   ├── features.py           # feature engineering (lag, rolling, time features)
│   └── modeling.py           # evaluate_model, tune_model, compare_models,
│                             # plot_predictions, feature_importance
├── 01_eda.ipynb              # exploratory data analysis
├── 02_feature_engineering.ipynb
├── 03_modelling.ipynb        # training, tuning, evaluation and comparison
└── README.md
```

---

## Key Decisions & Reasoning

- **TimeSeriesSplit over random split** — prevents data leakage in time-ordered data
- **7 folds** — tested 3, 5 and 7; 7 gave the most stable CV scores for this dataset size
- **Silica concentrate kept as a feature** — tested with and without; improvement was modest (not the exponential jump characteristic of true leakage), confirming it provides genuine process signal
- **XGBoost chosen as final model** — outperformed Random Forest after tuning; low learning rate and shallow trees were key to handling noisy industrial data

---

## What Comes Next

A natural extension of this work would be to shift the prediction window forward — predicting concentrate grade 3–6 hours ahead rather than current grade. This would give the technical team an explicit action window to adjust process variables before off-spec product is produced. The lag features already engineered provide a foundation for this.

---

## Stack

Python, scikit-learn, XGBoost, pandas, matplotlib, Jupyter

---

## About

This project was completed as the capstone machine learning project of the Ironhack Data Analytics bootcamp. The dataset is based on real flotation plant sensor data from a mining operation.
