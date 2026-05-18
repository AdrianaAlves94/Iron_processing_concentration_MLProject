# Predicting Iron Concentrate Quality in a Flotation Processing Plant

A machine learning project that predicts the purity of iron concentrate (% iron) from real-time sensor data collected across a flotation processing plant, enabling near real-time quality monitoring without waiting for lab results.

---

## The Problem

In iron ore processing plants, concentrate quality is typically measured through physical lab sampling, a process that takes hours. By the time operators know the iron grade is off-spec, a significant amount of product has already been processed under suboptimal conditions.

This project asks: **can we predict iron concentrate grade, using sensor data that is already being collected?**

---

## The Data

- 6 months of hourly sensor readings from a flotation processing plant (March–September 2017)
- Raw data: 737,453 rows at 20-second intervals, aggregated to ~4,000 hourly samples
- 116 features after feature engineering, including:
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
Because this is time series data, standard random train/test splitting would allow the model to train on future observations to predict the past (a form of data leakage). All splits were done **chronologically**, and `TimeSeriesSplit` (7 folds) was used for cross-validation, ensuring the model is always trained on the past and evaluated on what comes after. This mirrors real operational conditions.

### Data Leakage Investigation
An initial model including silica concentrate features produced a Test R² of 0.51. Feature importance analysis revealed that `pct_silica_concentrate_max` accounted for over 80% of the model's predictive weight. Since silica concentrate is measured on the same final product at the same time as iron concentrate, this constituted data leakage. All silica concentrate features were removed.

### Target Lag Feature
After removing leakage, model performance dropped significantly. Investigation of the raw data revealed that iron concentrate is a lab measurement updated infrequently. The previous hour's iron concentrate grade was added as a feature. This is a legitimate input since operators always have access to the last lab result. This improved model performance substantially.

### Model Selection
Three models were compared as baselines, chosen to cover the full spectrum from simple to complex:

| Model | Approach |
|---|---|
| Decision Tree | Simple interpretable baseline |
| Random Forest | Bagging — committee of trees |
| XGBoost | Boosting — learns from mistakes iteratively |

### Hyperparameter Tuning
`GridSearchCV` with `TimeSeriesSplit` was used to tune both Random Forest and XGBoost. XGBoost benefited most from tuning. Low learning rate (0.01), shallow trees (max_depth=3), and subsampling (0.8) prevented overfitting on noisy industrial data.

---

## Results

**Best model: tuned XGBoost**
- Predicts iron concentrate grade within **±0.56 percentage points (MAE)** on average
- CV and test scores are close, confirming the model generalises honestly to unseen time periods

### What the R² of 0.54 reflects

The dominant predictor is the previous hour's grade (feature importance ~14% in XGBoost), which works largely because the lab measurement is frozen for hours at a time rather than because the model is capturing true process dynamics. Process variables such as pH, amina flow and airflow do contribute but modestly.

This is consistent with the structural limitation of the dataset: the target is a lab measurement updated at irregular intervals, not a continuous sensor. A model predicting at hourly resolution is partly tracking lab scheduling rather than process physics.

---

## Key Decisions & Reasoning

- **TimeSeriesSplit over random split** — prevents data leakage in time-ordered data; tested 3, 5 and 7 folds and 7 gave the most stable CV scores
- **Silica concentrate removed** — feature importance showed it dominated at 80%+ weight; it is measured simultaneously with the target and constitutes circular leakage
- **Target lag added** — previous hour's iron grade is legitimately available to operators and captures the strong temporal autocorrelation (lag-1 = 0.75)
- **XGBoost chosen as final model** — outperformed Random Forest after tuning; conservative settings (low learning rate, shallow trees) were key to handling noisy industrial data

---

## Limitations

- The target variable is a lab measurement held constant between samples. 
- Process variables show weak direct correlation with iron grade on their own (max ~0.18)
- The test period exhibits slight concept drift with the average grade being lower than in training

---

## What Comes Next

- **Higher-frequency lab sampling** — the core bottleneck; finer-grained target measurements would expose the relationship between process inputs and output quality
- **Predicting silica concentrate** as a complementary target which is the plant's key quality indicator alongside iron grade

---

## Project Structure

```
├── data/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── src/
│   ├── preprocessing.py      # cleaning, date parsing, chronological split
│   ├── features.py           # lag features, rolling windows, time features
│   └── modelling.py          # compare_models, tune_model, feature_importance,
│                             # plot_predictions
├── 01_eda.ipynb              # exploratory data analysis on cleaned data
├── 02_feature_engineering.ipynb
├── 03_modelling.ipynb        # training, tuning, evaluation and comparison
└── README.md
```

---

## Stack

Python, scikit-learn, XGBoost, pandas, matplotlib, Jupyter

---

## About

This project was completed as the capstone machine learning project of the Ironhack Data Analytics bootcamp. The dataset is sourced from Kaggle (real flotation plant sensor data, 2017).
