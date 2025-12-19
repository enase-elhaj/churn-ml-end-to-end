import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_prep import load_and_clean_data
from config import TARGET, RANDOM_STATE, REPORTS_DIR
from sklearn.model_selection import train_test_split


MODEL_PATH = "models/random_forest_pipeline.joblib"


def main():
    print("Loading data...")
    df = load_and_clean_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]

    print("Transforming test data...")
    X_test_transformed = preprocessor.transform(X_test)

    # Convert to dense array (important for SHAP + DataFrame)
    if hasattr(X_test_transformed, "toarray"):
        X_test_dense = X_test_transformed.toarray()
    else:
        X_test_dense = np.asarray(X_test_transformed)

    # Rebuild feature names
    feature_names = []

    # numeric features
    numeric_features = preprocessor.transformers_[0][2]
    feature_names.extend(numeric_features)

    # categorical features
    cat_transformer = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]
    ohe = cat_transformer.named_steps["encoder"]
    ohe_names = ohe.get_feature_names_out(cat_features)

    feature_names.extend(ohe_names)

    # DataFrame only for readability in plots
    X_test_df = pd.DataFrame(X_test_dense, columns=feature_names)

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_dense)

    # For binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]  # explanations for churn class (1)
    else:
        shap_values_pos = shap_values

    print("Saving SHAP plots...")

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Summary Plot (global explanation)
    shap.summary_plot(shap_values_pos, X_test_df, show=False)
    plt.title("SHAP Summary Plot - Random Forest (Churn Class)")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_plot.png", dpi=200)
    plt.close()

    # 2️⃣ Bar plot (mean absolute shap)
    shap.summary_plot(shap_values_pos, X_test_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Impact - Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_bar_plot.png", dpi=200)
    plt.close()

    print("\nSHAP analysis complete.")
    print(f"Saved plots to: {fig_dir}")


if __name__ == "__main__":
    main()
