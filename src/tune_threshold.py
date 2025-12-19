import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from data_prep import load_and_clean_data
from config import TARGET, RANDOM_STATE


MODEL_PATH = "models/random_forest_pipeline.joblib"


def evaluate_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, prec, rec, f1


def main():
    # 1. Load data and split (same as training)
    df = load_and_clean_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 2. Load trained model
    model = joblib.load(MODEL_PATH)

    # 3. Get churn probabilities for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4. Define thresholds to test
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

    print("=== Threshold Tuning (Random Forest) ===")
    print("thr   acc     prec    rec     f1")
    print("----------------------------------------")

    results = []

    for thr in thresholds:
        acc, prec, rec, f1 = evaluate_at_threshold(y_test, y_prob, thr)
        results.append((thr, acc, prec, rec, f1))
        print(f"{thr:0.2f}  {acc:0.4f}  {prec:0.4f}  {rec:0.4f}  {f1:0.4f}")

    # 5. pick a "good" threshold (prioritize recall >= 0.80)
    candidates = [r for r in results if r[3] >= 0.80]  # r[3] = recall
    if candidates:
        best = max(candidates, key=lambda r: r[4])  # choose highest F1 among those
        print("\nSuggested threshold (recall >= 0.80, best F1):")
        print(f"Threshold={best[0]:0.2f}, acc={best[1]:0.4f}, prec={best[2]:0.4f}, rec={best[3]:0.4f}, f1={best[4]:0.4f}")
    else:
        print("\nNo threshold reached recall >= 0.80; consider thresholds around 0.4–0.5.")


if __name__ == "__main__":
    main()
