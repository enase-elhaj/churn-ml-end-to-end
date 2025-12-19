import joblib   #joblib is used to persist trained models so they can be reused later without retraining.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_prep import load_and_clean_data
from features import build_preprocessor
from config import TARGET, MODELS_DIR, RANDOM_STATE


def train_logistic_regression():
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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),  #Applies preprocessing: Imputation Scaling One-hot encoding
            ("model", LogisticRegression(    #Logistic Regression configuration
                max_iter=2000,
                class_weight="balanced"
            )),
        ]
    )

    pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "logistic_regression_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    print(f"✅ Trained Logistic Regression pipeline saved to: {model_path}")


if __name__ == "__main__":
    train_logistic_regression()
'''
This script trains a Logistic Regression classifier using a scikit-learn pipeline that combines 
preprocessing and modeling. The trained pipeline is serialized to disk using joblib, 
allowing the exact same preprocessing and model to be reused for inference without retraining, 
ensuring reproducibility and production safety.'''