import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml



def load_params(params_path):
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        print(f"Error loading params.yaml: {e}")
        raise


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        raise
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise


def split_data(X, y, test_size, random_state=42):
    """Split the data into training and testing sets."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error during data splitting: {e}")
        raise


def save_data(X_train, X_test, y_train, y_test, processed_data_dir):
    """Save the split data into CSV files."""
    try:
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(processed_data_dir / 'X_train.csv', index=False)
        X_test.to_csv(processed_data_dir / 'X_test.csv', index=False)
        y_train.to_csv(processed_data_dir / 'y_train.csv', index=False)
        y_test.to_csv(processed_data_dir / 'y_test.csv', index=False)
    except Exception as e:
        print(f"Error saving data: {e}")
        raise


def main():
    # Set up paths
    current_path = Path(__file__)
    root = current_path.parent.parent

    interim_data = root / 'data' / 'interim' 
    processed_data = root / 'data' / 'processed'
    params_path = root / 'params.yaml'

    # Load parameters
    params = load_params(params_path)
    test_size = params['data_splitting']['test_size']

    # Load data
    df = load_data(interim_data / 'cleaned_data.csv')

    # Prepare features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    # Save the split data
    save_data(X_train, X_test, y_train, y_test, processed_data)


if __name__ == "__main__":
    main()
