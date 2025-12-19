import joblib
import pandas as pd

MODEL_PATH = "models/random_forest_pipeline.joblib"
THRESHOLD = 0.45  # tuned decision threshold

def load_model():
    return joblib.load(MODEL_PATH)


def predict_single(customer_dict):
    """
    customer_dict = {
        "Tenure Months": ...,
        "Monthly Charges": ...,
        "Total Charges": ...,
        "Contract": ...,
        "Internet Service": ...,
        etc...
    }
    """
    model = load_model()

    df = pd.DataFrame([customer_dict])
 

    prob = model.predict_proba(df)[0, 1]   # Probability of churn (class 1)
    #pred = model.predict(df)[0]  # Default prediction at 0.5 threshold

     # Use tuned threshold instead of default 0.5
    pred = int(prob >= THRESHOLD)


    return {
        "prediction": int(pred),
        "churn_probability": float(prob),
        "risk_level": (
            "HIGH RISK" if prob >= 0.7
            else "MEDIUM RISK" if prob >= 0.45
            else "LOW RISK"
        ),
    }

if __name__ == "__main__":
    sample_customer = {
        "Tenure Months": 2,
        "Monthly Charges": 75.35,
        "Total Charges": 150.70,
        "Contract": "Month-to-month",
        "Payment Method": "Electronic check",
        "Paperless Billing": "Yes",
        "Internet Service": "Fiber optic",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Tech Support": "No",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Senior Citizen": "No",
        "Gender": "Female",
        "Partner": "No",
        "Dependents": "No",
    }

    result = predict_single(sample_customer)
    print("\n===== CHURN PREDICTION RESULT =====")
    print(result)
  
