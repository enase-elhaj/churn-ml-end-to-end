

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from features import build_preprocessor
from data_prep import load_and_clean_data
from config import TARGET, REPORTS_DIR


MODEL_PATH = "models/random_forest_pipeline.joblib"


def get_feature_names(preprocessor):
    """
    Reconstruct feature names after preprocessing
    (numeric + one-hot encoded categorical)
    """
    names = []

    # --- Numeric features ---
    numeric_features = preprocessor.transformers_[0][2]
    names.extend(numeric_features)

    # --- Categorical features ---
    cat_transformer = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]

    ohe = cat_transformer.named_steps["encoder"]
    ohe_names = ohe.get_feature_names_out(cat_features)

    names.extend(ohe_names)

    return names


def main():
    df = load_and_clean_data()
    X = df.drop(columns=[TARGET])

    # Load model
    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]

    feature_names = get_feature_names(preprocessor)

    importances = rf.feature_importances_

    feature_importance = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Top 20 Most Important Features ===")
    print(feature_importance.head(20))

    # Save CSV
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    out_path = REPORTS_DIR / "rf_feature_importance.csv"
    feature_importance.to_csv(out_path, index=False)
    print(f"\nSaved feature importance to: {out_path}")

    # Plot
    top_n = feature_importance.head(12)
    plt.figure(figsize=(10,6))
    plt.barh(top_n["feature"], top_n["importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_dir / "rf_feature_importance.png", dpi=200)
    plt.close()

    print(f"Saved feature importance plot to: {fig_dir}/rf_feature_importance.png")


if __name__ == "__main__":
    main()
