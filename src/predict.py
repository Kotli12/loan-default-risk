"""
predict.py — Score new loan applications using the trained model.

Usage:
    python src/predict.py --input data/new_applications.csv --threshold 0.35
"""
import argparse, joblib, sys, os
import pandas as pd
import numpy as np

def predict(input_path, threshold=0.35, model_path="models/gradient_boosting.pkl"):
    model = joblib.load(model_path)
    df    = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df):,} applications")

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["default_probability"] = probs.round(4)
    df["risk_tier"] = pd.cut(
        probs,
        bins=[0,.10,.20,.35,.50,1.0],
        labels=["Very Low","Low","Medium","High","Very High"]
    )
    df["decision"] = np.where(preds == 1, "DECLINE", "APPROVE")

    out = "outputs/scored_applications.csv"
    df[["default_probability","risk_tier","decision"]].to_csv(out, index=False)
    print(f"[INFO] Approvals: {(preds==0).sum():,} | Declines: {(preds==1).sum():,}")
    print(f"[INFO] Saved to {out}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--model",     default="models/gradient_boosting.pkl")
    args = parser.parse_args()
    predict(args.input, args.threshold, args.model)