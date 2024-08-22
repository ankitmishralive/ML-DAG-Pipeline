import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        raise
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise


def load_model(model_path):
    """Load a trained model from a file."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: The model file {model_path} was not found.")
        raise
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    try:
        predictions = model.predict(X_test)
        metrics = {
            'accuracy_score': accuracy_score(y_test, predictions),
            'precision_score': precision_score(y_test, predictions),
            'recall_score': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions)
        }
        return metrics
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics, metrics_file_path):
    """Save evaluation metrics to a JSON file."""
    try:
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        with metrics_file_path.open('w') as file:
            json.dump(metrics, file, indent=4)
        print(f"Metrics saved to {metrics_file_path}")
    except Exception as e:
        print(f"Error saving metrics to {metrics_file_path}: {e}")
        raise


def main():
    try:
        # Define paths
        current_path = Path(__file__)
        root = current_path.parent.parent
        processed_data = root / 'data' / 'processed'
        models_dir = root / 'models'
        metrics_dir = root / 'metrics'

        # Load test data and model
        X_test = load_data(processed_data / 'X_test.csv')
        y_test = load_data(processed_data / 'y_test.csv')
        model = load_model(models_dir / 'model.joblib')

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save the metrics
        save_metrics(metrics, metrics_dir / 'metrics.json')

    except Exception as e:
        print(f"An error occurred in the main process: {e}")
        raise


if __name__ == "__main__":
    main()
