import pandas as pd
from pathlib import Path
import pickle 
import json 
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score,precision_score,f1_score,recall_score


def main():
    current_path = Path(__file__)
    root = current_path.parent.parent

    processed_data = root / 'data' / 'processed'

    X_test = pd.read_csv(processed_data  / 'X_test.csv')
    y_test = pd.read_csv(processed_data  / 'y_test.csv')
    models_dir = root / 'models'
    model_path = models_dir / 'model.pkl'

    with model_path.open('rb') as model_file:
        model = pickle.load(model_file)

    model_prediction = model.predict(X_test)


    metrics_dir = root / 'metrics'

    # Ensure metrics directory exists
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        'accuracy_score': accuracy_score(y_test, model_prediction),
        'precision_score': precision_score(y_test, model_prediction),
        'recall_score': recall_score(y_test, model_prediction),
        'f1_score': f1_score(y_test, model_prediction)
    }

    metrics_file_path = metrics_dir / 'metrics.json'
    with metrics_file_path.open('w') as file:
        json.dump(metrics_dict, file, indent=4)
     





if __name__ == "__main__":
    main()


