import pandas as pd
from config import DATA_RAW, DROP_COLUMNS, TARGET

def load_raw_data() -> pd.DataFrame:
    """Load raw telco churn data from CSV."""
    return pd.read_csv(DATA_RAW)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, non-destructive cleaning:
    - Drop known-leakage / ID / geo columns
    - Convert Total Charges to numeric (coerce invalid to NaN)
    """
    df = df.copy()

    # Drop columns we decided to exclude (safe if missing)
    df.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")

    # Convert Total Charges to numeric (common issue in telco datasets)
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Ensure target exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data.")

    return df

def load_and_clean_data() -> pd.DataFrame:
    df = load_raw_data()
    df = clean_dataframe(df)
    return df

#Script execution block
if __name__ == "__main__":     #This block runs only when the file is executed directly, not when imported.
    df = load_and_clean_data()
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values (top 15):\n", df.isna().sum().sort_values(ascending=False).head(15))
    print("\nTarget distribution:\n", df[TARGET].value_counts(normalize=True))
