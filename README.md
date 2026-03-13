# 💳 Loan Default Risk Prediction

End-to-end credit risk pipeline trained on LendingClub-style data — comparing Logistic Regression and XGBoost for predicting loan defaults, with full preprocessing pipelines, hyperparameter tuning, and stakeholder-ready visualisations.

---

## 🚀 Results

| Model | ROC-AUC | Avg Precision | F1 (Default) |
|---|---|---|---|
| Logistic Regression | ~0.88 | ~0.72 | ~0.70 |
| **XGBoost** | **~0.93** | **~0.81** | **~0.76** |

XGBoost outperforms Logistic Regression across all metrics, particularly on the minority class (defaults). However, Logistic Regression provides clear coefficient interpretability — making it valuable for regulatory explainability requirements.

---

## 🧠 Approach

### Why two models?

| | Logistic Regression | XGBoost |
|---|---|---|
| Strength | Interpretable, fast, regulatory-friendly | High performance, captures non-linear relationships |
| Use case | Explain WHY a loan was declined | Maximise predictive accuracy |
| Output | Coefficient direction & magnitude | Feature importance (gain) |

### Pipeline Architecture

```
Raw LendingClub Data
        │
        ▼
Feature Engineering  ─── loan-to-income ratio, high utilisation flag,
        │                  employment years, interaction terms
        ▼
ColumnTransformer
   ├── Numeric  → Median Imputation → StandardScaler
   └── Categorical → Mode Imputation → OneHotEncoder
        │
        ▼
RandomizedSearchCV (5-fold StratifiedKFold, scoring=roc_auc)
   ├── Logistic Regression  (C, penalty, solver)
   └── XGBoost              (n_estimators, max_depth, learning_rate,
        │                    subsample, colsample_bytree)
        ▼
Threshold Optimisation  ─── tuned on Precision-Recall curve
        │
        ▼
Risk Scorecard  ─── 5 tiers: Very Low → Very High
```

---

## 📊 Outputs

| File | Description |
|---|---|
| `outputs/01_eda_overview.png` | 6-panel EDA: default by grade, DTI, purpose, utilisation |
| `outputs/02_roc_pr_curves.png` | ROC & PR curves for both models |
| `outputs/03_confusion_matrices.png` | Side-by-side confusion matrices |
| `outputs/04_feature_importance.png` | Top 18 XGBoost risk drivers |
| `outputs/05_risk_scorecard.png` | Borrower segmentation by risk tier |
| `outputs/06_lr_coefficients.png` | LR coefficients — direction of risk |
| `outputs/07_model_comparison.png` | Head-to-head metric comparison |

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/loan-default-risk.git
cd loan-default-risk

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python src/loan_default_pipeline.py
```

**Score new applications:**
```bash
python src/predict.py --input data/new_applications.csv --threshold 0.35
```

**Notebook walkthrough:**
```bash
jupyter notebook notebooks/loan_default_walkthrough.ipynb
```

---

## 📐 Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `loan_to_income` | loan_amnt / annual_inc | Affordability ratio |
| `int_rate_x_dti` | int_rate × dti | Combined debt burden signal |
| `has_delinq` | delinq_2yrs > 0 | Binary delinquency flag |
| `has_pub_rec` | pub_rec > 0 | Binary public record flag |
| `high_util` | revol_util > 75% | High credit utilisation flag |
| `emp_years` | mapped from emp_length | Ordinal employment stability |

---

## 🔍 Key Risk Findings

- **Loan grade** is the strongest predictor — Grade G loans default at ~10× the rate of Grade A
- **Revolving utilisation > 75%** nearly doubles default probability
- **DTI** and **interest rate** are highly correlated with default — even within the same grade
- Customers with **any public record** default ~40% more often than those without
- **Small business loans** have the highest default rate by purpose (~28%)

---

## 📁 Structure

```
loan-default-risk/
├── data/
│   └── lendingclub_sample.csv    # Generated on first run
├── src/
│   ├── loan_default_pipeline.py  # Full pipeline (train + evaluate + plot)
│   └── predict.py                # Inference on new applications
├── notebooks/
│   └── loan_default_walkthrough.ipynb
├── models/                       # Saved .pkl files (generated on run)
├── outputs/                      # All charts (generated on run)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- `scikit-learn` — Pipeline, ColumnTransformer, RandomizedSearchCV, Logistic Regression
- `xgboost` — Gradient boosted classifier
- `pandas` / `numpy` — Data manipulation & feature engineering
- `matplotlib` — All visualisations
- `scipy` — Hyperparameter search distributions
- `joblib` — Model serialisation

---

## 📄 License

MIT
