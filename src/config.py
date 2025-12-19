from pathlib import Path

# ======================
# Paths
# ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ======================
# Target
# ======================
TARGET = "Churn Value"

# ======================
# Feature groups
# ======================
NUMERIC_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

CATEGORICAL_FEATURES = [
    "Contract",
    "Payment Method",
    "Paperless Billing",
    "Internet Service",
    "Phone Service",
    "Multiple Lines",
    "Tech Support",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Streaming TV",
    "Streaming Movies",
    "Senior Citizen",
    "Gender",
    "Partner",
    "Dependents",
]

# ======================
# Columns to drop
# ======================
DROP_COLUMNS = [
    "CustomerID",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CLTV",
    "Zip Code",
    "Count"
]

# ======================
# Reproducibility
# ======================
RANDOM_STATE = 42

'''
With this file:

EDA decisions are codified

preprocessing is traceable

model training is consistent

experimentation is easy

reproducibility is ensured
'''