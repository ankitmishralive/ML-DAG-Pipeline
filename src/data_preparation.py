import pandas as pd
import numpy as np
from pathlib import Path

def clean_and_impute(df, column):
    """Clean and impute missing values in the specified column of the DataFrame."""
    # Outlier removal using IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    
    # Impute missing values with the mean
    mean_value = df[column].mean()
    df[column].fillna(mean_value, inplace=True)
    return df

def main():
    # Define paths
    current_path = Path(__file__)
    root = current_path.parent.parent
    print("---",root)

    data_path = root / 'data' / 'raw' / 'extracted'
    print("---",data_path)
    cleaned_data_path = root / 'data' / 'interim' / 'cleaned_data.csv'

    # Load data
    df = pd.read_csv(data_path / 'corpus.csv')

    # List of columns to clean
    columns_to_clean = ['ph', 'Sulfate', 'Trihalomethanes']

    # Apply cleaning and imputation
    for column in columns_to_clean:
        df = clean_and_impute(df, column)

    # Save the cleaned data
    df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to {cleaned_data_path}")

if __name__ == "__main__":
    main()
