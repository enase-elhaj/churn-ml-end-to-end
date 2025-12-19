import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
)

from data_prep import load_and_clean_data
from config import TARGET, RANDOM_STATE, REPORTS_DIR


def main():
    # Load data
    df = load_and_clean_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Same split settings as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Load trained model pipeline
    model = joblib.load("models/logistic_regression_pipeline.joblib")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("=== Evaluation Metrics (Test Set) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    # Create figures directory
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Logistic Regression")
    plt.savefig(fig_dir / "confusion_matrix_logreg.png", bbox_inches="tight")
    plt.close()

    # ROC curve plot
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve - Logistic Regression")
    plt.savefig(fig_dir / "roc_curve_logreg.png", bbox_inches="tight")
    plt.close()

    print(f"\n✅ Saved plots to: {fig_dir}")


if __name__ == "__main__":
    main()
