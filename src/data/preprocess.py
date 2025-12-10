import os
import pandas as pd
from .load_data import load_dataset


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping rows with missing values and preprocessing features."""
    # Drop rows with missing values
    df = df.dropna().copy()
    
    # Drop ID columns and date columns that aren't useful for prediction
    columns_to_drop = ['CustomerID', 'Last Due Date', 'Last Payment Date']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Encode categorical variables using one-hot encoding
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df


if __name__ == "__main__":
    # Load the raw dataset
    raw = load_dataset("data/raw/train.csv")
    # Clean the dataset
    cleaned = clean_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/churn_clean.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
