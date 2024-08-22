import pandas as pd
import numpy as np
from pathlib import Path

def clean_and_impute(df, column):
    """Clean and impute missing values in the specified column of the DataFrame."""
    try:
        # Outlier removal using IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

        # Impute missing values with the mean
        mean_value = df[column].mean()
   
        df[column].fillna(mean_value, inplace=True)
        return df
    except KeyError:
        print(f"Error: The column '{column}' does not exist in the DataFrame.")
        raise
    except Exception as e:
        print(f"An error occurred during cleaning and imputation: {e}")
        raise

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        raise
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def save_data(df, file_path):
    """Save the DataFrame to a CSV file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        df.to_csv(file_path, index=False)
        print(f"Cleaned data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
        raise

def main():
    try:
        # Define paths
        current_path = Path(__file__)
        root = current_path.parent.parent
        data_path = root / 'data' / 'raw' / 'extracted'
        cleaned_data_path = root / 'data' / 'interim' / 'cleaned_data.csv'

        # Load data
        df = load_data(data_path / 'corpus.csv')

        # List of columns to clean
        columns_to_clean = ['ph', 'Sulfate', 'Trihalomethanes']

        # Apply cleaning and imputation
        for column in columns_to_clean:
            df = clean_and_impute(df, column)

        # Save the cleaned data
        save_data(df, cleaned_data_path)
    
    except Exception as e:
        print(f"An error occurred in the main process: {e}")
        raise

if __name__ == "__main__":
    main()
