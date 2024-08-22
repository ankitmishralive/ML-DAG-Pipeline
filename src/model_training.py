
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import pandas as pd
import joblib
import yaml

colsample_bytree=yaml.safe_load(open("params.yaml"))['model_training']['colsample_bytree']
learning_rate=yaml.safe_load(open("params.yaml"))['model_training']['learning_rate']
max_depth=yaml.safe_load(open("params.yaml"))['model_training']['max_depth']
n_estimators=yaml.safe_load(open("params.yaml"))['model_training']['n_estimators']
num_leaves=yaml.safe_load(open("params.yaml"))['model_training']['num_leaves']
subsample=yaml.safe_load(open("params.yaml"))['model_training']['subsample']

def main():

    current_path = Path(__file__)
    root = current_path.parent.parent

    processed_data = root / 'data' / 'processed'

    X_train = pd.read_csv(processed_data  / 'X_train.csv')
    X_test = pd.read_csv(processed_data  / 'X_test.csv')
    y_train = pd.read_csv(processed_data  / 'y_train.csv')
    y_test = pd.read_csv(processed_data  / 'y_test.csv')

    lgbm_model = lgb.LGBMClassifier(
        colsample_bytree=colsample_bytree,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        subsample=subsample
    )

    # Training 
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
    
    )

    current_path = Path(__file__)
    root = current_path.parent.parent
    models_dir = root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # print("current path :--", current_path)
    # print("root path :--", root)
    # print("models_dir :--", models_dir)

    model_path = models_dir / 'model.joblib'
    joblib.dump(lgbm_model, model_path)
                                                                                                                 

if __name__ == "__main__":
    main()
                                                                                                                                                                   