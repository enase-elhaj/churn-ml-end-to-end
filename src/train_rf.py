import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_prep import load_and_clean_data
from features import build_preprocessor
from config import TARGET, MODELS_DIR, RANDOM_STATE


def train_random_forest():
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
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300,       #Number of trees in the forest, stable performance
                max_depth=None,
                min_samples_leaf=10,      #Minimum samples required to be at a leaf node, helps prevent overfitting
                class_weight="balanced",   #improves churn recall
                random_state=RANDOM_STATE,
                n_jobs=-1,     #Utilize all available CPU cores for training
            )),
        ]
    )

    pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "random_forest_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    print(f"✅ Random Forest pipeline trained and saved to: {model_path}")


if __name__ == "__main__":
    train_random_forest()
