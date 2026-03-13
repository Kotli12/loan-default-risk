"""
loan_default_pipeline.py
End-to-end credit risk pipeline:
  - Synthetic LendingClub-style data generation
  - Preprocessing pipeline (imputation, encoding, scaling)
  - Logistic Regression + XGBoost training
  - Hyperparameter tuning via RandomizedSearchCV
  - Feature importance + stakeholder visualisations
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import os, joblib

from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

PALETTE = {
    "blue":       "#2563EB",
    "light_blue": "#DBEAFE",
    "red":        "#DC2626",
    "light_red":  "#FEE2E2",
    "green":      "#16A34A",
    "light_green":"#DCFCE7",
    "amber":      "#D97706",
    "purple":     "#7C3AED",
    "gray":       "#6B7280",
    "dark":       "#1F2937",
}

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC LENDINGCLUB-STYLE DATA
# ─────────────────────────────────────────────

def generate_data(n=15000, seed=42):
    rng = np.random.default_rng(seed)

    grade      = rng.choice(["A","B","C","D","E","F","G"],
                             n, p=[.18,.22,.20,.17,.12,.07,.04])
    sub_grade  = [g + str(rng.integers(1,6)) for g in grade]

    # Credit score correlated with grade
    grade_score = {"A":750,"B":710,"C":680,"D":650,"E":620,"F":590,"G":560}
    credit_score = np.array([grade_score[g] + rng.integers(-30,31)
                              for g in grade]).clip(300, 850)

    loan_amnt  = rng.choice([5000,10000,15000,20000,25000,30000,35000],
                             n, p=[.10,.20,.20,.18,.15,.10,.07])
    int_rate   = np.array([
        {"A":6.5,"B":9.5,"C":13.0,"D":16.5,"E":19.5,"F":22.0,"G":25.0}[g]
        + rng.uniform(-1,1) for g in grade
    ]).clip(5, 30)

    annual_inc     = np.exp(rng.normal(10.9, 0.6, n)).clip(20000, 500000)
    dti            = rng.uniform(0, 45, n).round(2)
    emp_length     = rng.choice(
        ["< 1 year","1 year","2 years","3 years","4 years",
         "5 years","6 years","7 years","8 years","9 years","10+ years"],
        n, p=[.08,.07,.09,.09,.08,.09,.07,.07,.07,.06,.23])
    home_ownership = rng.choice(["RENT","OWN","MORTGAGE","OTHER"],
                                n, p=[.45,.10,.42,.03])
    purpose        = rng.choice(
        ["debt_consolidation","credit_card","home_improvement",
         "other","major_purchase","medical","small_business"],
        n, p=[.38,.24,.10,.12,.07,.05,.04])
    term           = rng.choice([36, 60], n, p=[.65,.35])
    open_acc       = rng.integers(2, 30, n)
    revol_util     = rng.uniform(0, 99.9, n).round(1)
    pub_rec        = rng.choice([0,1,2,3], n, p=[.80,.12,.05,.03])
    delinq_2yrs    = rng.choice([0,1,2,3,4], n, p=[.72,.15,.07,.04,.02])
    inq_last_6mths = rng.choice([0,1,2,3,4,5], n, p=[.40,.25,.17,.10,.05,.03])
    mort_acc       = rng.choice([0,1,2,3,4], n, p=[.38,.25,.18,.12,.07])
    total_acc      = open_acc + rng.integers(0, 15, n)

    # --- Default probability model (realistic signal) ---
    grade_risk = {"A":.03,"B":.07,"C":.13,"D":.20,"E":.28,"F":.36,"G":.44}
    base_prob  = np.array([grade_risk[g] for g in grade])
    base_prob += (dti / 100) * 0.25
    base_prob += (revol_util / 100) * 0.15
    base_prob -= (credit_score - 300) / 550 * 0.20
    base_prob += (delinq_2yrs > 0).astype(float) * 0.08
    base_prob += (pub_rec > 0).astype(float) * 0.05
    base_prob += (inq_last_6mths > 2).astype(float) * 0.04
    base_prob -= (annual_inc > 80000).astype(float) * 0.05
    base_prob  = base_prob.clip(0.01, 0.92)

    default    = (rng.uniform(0,1,n) < base_prob).astype(int)

    # Introduce ~5% missing values in a few columns
    for col_arr in [dti, revol_util, annual_inc]:
        mask = rng.random(n) < 0.04
        col_arr[mask] = np.nan

    df = pd.DataFrame({
        "loan_amnt":       loan_amnt,
        "term":            term,
        "int_rate":        int_rate,
        "grade":           grade,
        "sub_grade":       sub_grade,
        "emp_length":      emp_length,
        "home_ownership":  home_ownership,
        "annual_inc":      annual_inc.round(2),
        "purpose":         purpose,
        "dti":             dti,
        "delinq_2yrs":     delinq_2yrs,
        "inq_last_6mths":  inq_last_6mths,
        "open_acc":        open_acc,
        "pub_rec":         pub_rec,
        "revol_util":      revol_util,
        "total_acc":       total_acc,
        "mort_acc":        mort_acc,
        "credit_score":    credit_score,
        "loan_default":    default,
    })
    df.to_csv("data/lendingclub_sample.csv", index=False)
    print(f"[DATA] {n:,} loans generated | Default rate: {default.mean():.1%}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()
    df["loan_to_income"]      = df["loan_amnt"] / (df["annual_inc"].fillna(df["annual_inc"].median()) + 1)
    df["int_rate_x_dti"]      = df["int_rate"] * df["dti"].fillna(df["dti"].median())
    df["has_delinq"]          = (df["delinq_2yrs"] > 0).astype(int)
    df["has_pub_rec"]         = (df["pub_rec"]     > 0).astype(int)
    df["high_util"]           = (df["revol_util"].fillna(50) > 75).astype(int)
    df["emp_years"] = df["emp_length"].map({
        "< 1 year":0,"1 year":1,"2 years":2,"3 years":3,"4 years":4,
        "5 years":5,"6 years":6,"7 years":7,"8 years":8,"9 years":9,
        "10+ years":10
    }).fillna(0)
    print(f"[FEAT] Features after engineering: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def build_preprocessor(num_features, cat_features):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ])


# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────

def train_logistic_regression(preprocessor, X_train, y_train):
    print("\n[LR] Training Logistic Regression...")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    param_dist = {
        "classifier__C":       uniform(0.01, 10),
        "classifier__penalty": ["l1","l2"],
        "classifier__solver":  ["liblinear","saga"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=cv,
                                scoring="roc_auc", n_jobs=-1,
                                random_state=42, verbose=0)
    search.fit(X_train, y_train)
    print(f"[LR] Best CV AUC: {search.best_score_:.4f} | Params: {search.best_params_}")
    return search.best_estimator_


def train_xgboost(preprocessor, X_train, y_train):
    print("\n[XGB] Training XGBoost...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            eval_metric="auc", random_state=42,
            scale_pos_weight=scale_pos, use_label_encoder=False,
            n_jobs=-1)),
    ])
    param_dist = {
        "classifier__n_estimators":    randint(100, 500),
        "classifier__max_depth":       randint(3, 8),
        "classifier__learning_rate":   uniform(0.01, 0.2),
        "classifier__subsample":       uniform(0.6, 0.4),
        "classifier__colsample_bytree":uniform(0.6, 0.4),
        "classifier__min_child_weight":randint(1, 10),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=25, cv=cv,
                                scoring="roc_auc", n_jobs=-1,
                                random_state=42, verbose=0)
    search.fit(X_train, y_train)
    print(f"[XGB] Best CV AUC: {search.best_score_:.4f} | Params: {search.best_params_}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name="Model"):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc   = roc_auc_score(y_test, probs)
    ap    = average_precision_score(y_test, probs)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, preds,
                                 target_names=["Fully Paid","Default"]))
    print(f"  ROC-AUC:  {auc:.4f}  |  Avg Precision: {ap:.4f}")
    return probs, preds, auc


# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.spines.bottom":False,
        "axes.grid":         True,
        "grid.color":        "#E5E7EB",
        "grid.linewidth":    0.8,
        "font.family":       "DejaVu Sans",
        "axes.labelcolor":   PALETTE["dark"],
        "xtick.color":       PALETTE["gray"],
        "ytick.color":       PALETTE["gray"],
        "axes.titlepad":     14,
    })


def plot_eda(df):
    """Stakeholder-friendly EDA overview."""
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("LendingClub Portfolio Overview — Exploratory Analysis",
                 fontsize=15, fontweight="bold", color=PALETTE["dark"], y=1.01)

    # 1. Default rate by grade
    ax = axes[0, 0]
    gr = df.groupby("grade")["loan_default"].mean().reset_index()
    bars = ax.bar(gr["grade"], gr["loan_default"]*100,
                  color=[PALETTE["blue"] if r<.15 else
                         PALETTE["amber"] if r<.25 else
                         PALETTE["red"] for r in gr["loan_default"]],
                  edgecolor="white", linewidth=0.5)
    ax.set_title("Default Rate by Loan Grade", fontweight="bold",
                 color=PALETTE["dark"])
    ax.set_xlabel("Grade"); ax.set_ylabel("Default Rate (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    for bar, val in zip(bars, gr["loan_default"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8,
                color=PALETTE["dark"])

    # 2. Loan amount distribution by default
    ax = axes[0, 1]
    for label, color, alpha in [("Fully Paid", PALETTE["blue"], 0.6),
                                  ("Default",    PALETTE["red"],  0.7)]:
        mask = df["loan_default"] == (0 if label=="Fully Paid" else 1)
        ax.hist(df[mask]["loan_amnt"], bins=25, alpha=alpha,
                color=color, label=label, density=True, edgecolor="white")
    ax.set_title("Loan Amount Distribution", fontweight="bold", color=PALETTE["dark"])
    ax.set_xlabel("Loan Amount (USD)"); ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    ax.legend(frameon=False)

    # 3. DTI vs Default (boxplot)
    ax = axes[0, 2]
    paid    = df[df["loan_default"]==0]["dti"].dropna()
    default = df[df["loan_default"]==1]["dti"].dropna()
    bp = ax.boxplot([paid, default], patch_artist=True,
                    widths=0.5, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(PALETTE["light_blue"])
    bp["boxes"][1].set_facecolor(PALETTE["light_red"])
    bp["boxes"][0].set_edgecolor(PALETTE["blue"])
    bp["boxes"][1].set_edgecolor(PALETTE["red"])
    ax.set_xticklabels(["Fully Paid", "Default"])
    ax.set_title("Debt-to-Income Ratio by Outcome", fontweight="bold",
                 color=PALETTE["dark"])
    ax.set_ylabel("DTI (%)")

    # 4. Loan purpose breakdown
    ax = axes[1, 0]
    purpose_dr = (df.groupby("purpose")["loan_default"]
                    .agg(["mean","count"])
                    .sort_values("mean", ascending=True))
    colors = [PALETTE["red"] if r > 0.18 else
              PALETTE["amber"] if r > 0.13 else
              PALETTE["blue"] for r in purpose_dr["mean"]]
    ax.barh(purpose_dr.index, purpose_dr["mean"]*100,
            color=colors, edgecolor="white")
    ax.set_title("Default Rate by Loan Purpose", fontweight="bold",
                 color=PALETTE["dark"])
    ax.set_xlabel("Default Rate (%)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))

    # 5. Interest rate distribution
    ax = axes[1, 1]
    ax.hist(df[df["loan_default"]==0]["int_rate"], bins=30,
            alpha=0.6, color=PALETTE["blue"], label="Fully Paid",
            density=True, edgecolor="white")
    ax.hist(df[df["loan_default"]==1]["int_rate"], bins=30,
            alpha=0.7, color=PALETTE["red"], label="Default",
            density=True, edgecolor="white")
    ax.set_title("Interest Rate Distribution", fontweight="bold",
                 color=PALETTE["dark"])
    ax.set_xlabel("Interest Rate (%)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    # 6. Revolving utilisation vs default
    ax = axes[1, 2]
    bins = [0,25,50,75,100]
    labels = ["0-25%","25-50%","50-75%","75-100%"]
    df2 = df.copy()
    df2["util_band"] = pd.cut(df2["revol_util"].fillna(50),
                               bins=bins, labels=labels)
    util_dr = df2.groupby("util_band")["loan_default"].mean()
    ax.bar(util_dr.index, util_dr*100,
           color=[PALETTE["blue"],PALETTE["blue"],
                  PALETTE["amber"],PALETTE["red"]],
           edgecolor="white")
    ax.set_title("Default Rate by Revolving Utilisation",
                 fontweight="bold", color=PALETTE["dark"])
    ax.set_xlabel("Utilisation Band")
    ax.set_ylabel("Default Rate (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))

    plt.tight_layout()
    plt.savefig("outputs/01_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] outputs/01_eda_overview.png")


def plot_roc_pr(lr_model, xgb_model, X_test, y_test):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Performance — ROC & Precision-Recall Curves",
                 fontsize=13, fontweight="bold", color=PALETTE["dark"])

    models_info = [
        (lr_model,  "Logistic Regression", PALETTE["blue"]),
        (xgb_model, "XGBoost",             PALETTE["purple"]),
    ]

    for ax_idx, (ax, curve_type) in enumerate(zip(axes, ["ROC","PR"])):
        for model, name, color in models_info:
            probs = model.predict_proba(X_test)[:, 1]
            if curve_type == "ROC":
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc = roc_auc_score(y_test, probs)
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f"{name} (AUC = {auc:.4f})")
            else:
                prec, rec, _ = precision_recall_curve(y_test, probs)
                ap = average_precision_score(y_test, probs)
                ax.plot(rec, prec, color=color, lw=2,
                        label=f"{name} (AP = {ap:.4f})")

        if curve_type == "ROC":
            ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random Classifier")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve", fontweight="bold")
        else:
            baseline = y_test.mean()
            ax.axhline(baseline, color="gray", lw=1, linestyle="--",
                       label=f"Baseline ({baseline:.2f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve", fontweight="bold")

        ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/02_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] outputs/02_roc_pr_curves.png")


def plot_confusion_matrices(lr_model, xgb_model, X_test, y_test):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices — Logistic Regression vs XGBoost",
                 fontsize=13, fontweight="bold", color=PALETTE["dark"])
    for ax, model, name in zip(axes,
                               [lr_model, xgb_model],
                               ["Logistic Regression", "XGBoost"]):
        preds = model.predict(X_test)
        cm    = confusion_matrix(y_test, preds)
        disp  = ConfusionMatrixDisplay(cm,
                    display_labels=["Fully Paid","Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontweight="bold", color=PALETTE["dark"])
        ax.grid(False)

    plt.tight_layout()
    plt.savefig("outputs/03_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] outputs/03_confusion_matrices.png")


def plot_feature_importance(xgb_model, num_features, cat_features, top_n=18):
    """Extract and plot XGBoost feature importances with readable names."""
    set_style()

    ohe_feature_names = (xgb_model.named_steps["preprocessor"]
                          .named_transformers_["cat"]
                          .named_steps["encoder"]
                          .get_feature_names_out(cat_features).tolist())
    all_feature_names = num_features + ohe_feature_names

    importances = xgb_model.named_steps["classifier"].feature_importances_
    fi = (pd.Series(importances, index=all_feature_names)
            .nlargest(top_n)
            .sort_values())

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PALETTE["red"] if i >= len(fi)-5 else
              PALETTE["blue"] for i in range(len(fi))]
    bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="white",
                   height=0.7)

    ax.set_title(f"Top {top_n} Risk Factors — XGBoost Feature Importance",
                 fontsize=13, fontweight="bold", color=PALETTE["dark"], pad=16)
    ax.set_xlabel("Importance Score (Gain)", color=PALETTE["gray"])

    for bar in bars:
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", fontsize=8,
                color=PALETTE["gray"])

    top_patch  = mpatches.Patch(color=PALETTE["red"],   label="Top 5 risk drivers")
    other_patch= mpatches.Patch(color=PALETTE["blue"],  label="Supporting features")
    ax.legend(handles=[top_patch, other_patch], frameon=False,
              loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/04_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] outputs/04_feature_importance.png")


def plot_risk_scorecard(xgb_model, X_test, y_test):
    """
    Stakeholder-friendly risk scorecard:
    Bins borrowers into risk tiers and shows default rate per tier.
    """
    set_style()
    probs = xgb_model.predict_proba(X_test)[:, 1]

    score_df = pd.DataFrame({
        "default_prob": probs,
        "actual":       y_test.values,
    })
    score_df["risk_tier"] = pd.cut(
        score_df["default_prob"],
        bins=[0, .10, .20, .35, .50, 1.0],
        labels=["Very Low\n(0–10%)", "Low\n(10–20%)",
                "Medium\n(20–35%)", "High\n(35–50%)",
                "Very High\n(50%+)"]
    )

    tier_stats = (score_df.groupby("risk_tier", observed=True)
                           .agg(count=("actual","count"),
                                default_rate=("actual","mean"))
                           .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Credit Risk Scorecard — Borrower Segmentation",
                 fontsize=13, fontweight="bold", color=PALETTE["dark"])

    tier_colors = [PALETTE["green"], PALETTE["blue"],
                   PALETTE["amber"], PALETTE["red"], "#7F1D1D"]

    # Left: volume per tier
    ax1 = axes[0]
    bars = ax1.bar(tier_stats["risk_tier"], tier_stats["count"],
                   color=tier_colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("Borrower Distribution by Risk Tier",
                  fontweight="bold", color=PALETTE["dark"])
    ax1.set_xlabel("Risk Tier"); ax1.set_ylabel("Number of Borrowers")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{bar.get_height():,.0f}", ha="center", va="bottom",
                 fontsize=9, color=PALETTE["dark"])

    # Right: actual default rate per tier
    ax2 = axes[1]
    bars2 = ax2.bar(tier_stats["risk_tier"],
                    tier_stats["default_rate"]*100,
                    color=tier_colors, edgecolor="white", linewidth=0.5)
    ax2.set_title("Actual Default Rate per Risk Tier",
                  fontweight="bold", color=PALETTE["dark"])
    ax2.set_xlabel("Risk Tier"); ax2.set_ylabel("Default Rate (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    for bar, val in zip(bars2, tier_stats["default_rate"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold",
                 color=PALETTE["dark"])

    plt.tight_layout()
    plt.savefig("outputs/05_risk_scorecard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] outputs/05_risk_scorecard.png")


def plot_logistic_coefficients(lr_model, num_features, cat_features, top_n=16):
    """
    Logistic regression coefficients — shows direction of risk.
    Positive = increases default probability.
    """
    set_style()

    ohe_names = (lr_model.named_steps["preprocessor"]
                  .named_transformers_["cat"]
                  .named_steps["encoder"]
                  .get_feature_names_out(cat_features).tolist())
    all_names = num_features + ohe_names
    coefs     = lr_model.named_steps["classifier"].coef_[0]

    coef_df = (pd.Series(coefs, index=all_names)
                 .reindex(pd.Series(np.abs(coefs), index=all_names)
                            .nlargest(top_n).index)
                 .sort_values())

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE["red"] if v > 0 else PALETTE["blue"]
              for v in coef_df.values]
    ax.barh(coef_df.index, coef_df.values, color=colors,
            edgecolor="white", height=0.7)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)

    ax.set_title(f"Logistic Regression Coefficients — Top {top_n} Risk Factors",
                 fontsize=13, fontweight="bold", color=PALETTE["dark"], pad=16)
    ax.set_xlabel("Coefficient (positive = higher default risk)",
                  color=PALETTE["gray"])

    pos_patch = mpatches.Patch(color=PALETTE["red"],
                                label="Increases default risk")
    neg_patch = mpatches.Patch(color=PALETTE["blue"],
                                label="Decreases default risk")
    ax.legend(handles=[pos_patch, neg_patch], frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/06_lr_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] outputs/06_lr_coefficients.png")


def plot_model_comparison(lr_model, xgb_model, X_test, y_test):
    """Side-by-side model comparison for stakeholder presentation."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    metrics_data = {}
    for model, name in [(lr_model,"Logistic Regression"),
                         (xgb_model,"XGBoost")]:
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        from sklearn.metrics import f1_score, precision_score, recall_score
        metrics_data[name] = {
            "ROC-AUC":   roc_auc_score(y_test, probs),
            "Avg Prec":  average_precision_score(y_test, probs),
            "F1 (Default)": f1_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall":    recall_score(y_test, preds),
        }

    df_metrics = pd.DataFrame(metrics_data).T
    x = np.arange(len(df_metrics.columns))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_metrics.loc["Logistic Regression"],
                   width, label="Logistic Regression",
                   color=PALETTE["blue"], edgecolor="white")
    bars2 = ax.bar(x + width/2, df_metrics.loc["XGBoost"],
                   width, label="XGBoost",
                   color=PALETTE["purple"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics.columns)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Logistic Regression vs XGBoost",
                 fontsize=13, fontweight="bold", color=PALETTE["dark"])
    ax.legend(frameon=False)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, color=PALETTE["dark"])

    plt.tight_layout()
    plt.savefig("outputs/07_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] outputs/07_model_comparison.png")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  LOAN DEFAULT RISK PREDICTION PIPELINE")
    print("=" * 55)

    # Data
    df  = generate_data()
    df  = engineer_features(df)
    plot_eda(df)

    # Split
    drop_cols = ["loan_default", "sub_grade"]
    X = df.drop(columns=drop_cols)
    y = df["loan_default"]

    num_features = X.select_dtypes(include=["number"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"\n[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"[SPLIT] Default rate — Train: {y_train.mean():.1%} | "
          f"Test: {y_test.mean():.1%}")

    # Build preprocessor
    preprocessor = build_preprocessor(num_features, cat_features)

    # Train
    lr_model  = train_logistic_regression(
        build_preprocessor(num_features, cat_features), X_train, y_train)
    xgb_model = train_xgboost(
        build_preprocessor(num_features, cat_features), X_train, y_train)

    # Evaluate
    evaluate_model(lr_model,  X_test, y_test, "Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Plots
    plot_roc_pr(lr_model, xgb_model, X_test, y_test)
    plot_confusion_matrices(lr_model, xgb_model, X_test, y_test)
    plot_feature_importance(xgb_model, num_features, cat_features)
    plot_risk_scorecard(xgb_model, X_test, y_test)
    plot_logistic_coefficients(lr_model, num_features, cat_features)
    plot_model_comparison(lr_model, xgb_model, X_test, y_test)

    # Save models
    joblib.dump(lr_model,  "models/logistic_regression.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")
    print("\n[SAVE] Models saved to models/")
    print("\n✓ Pipeline complete — all outputs in outputs/")

    return lr_model, xgb_model


if __name__ == "__main__":
    run()
