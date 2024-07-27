
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import pandas as pd
import joblib


def main():

    current_path = Path(__file__)
    root = current_path.parent.parent

    processed_data = root / 'data' / 'processed'

    X_train = pd.read_csv(processed_data  / 'X_train.csv')
    X_test = pd.read_csv(processed_data  / 'X_test.csv')
    y_train = pd.read_csv(processed_data  / 'y_train.csv')
    y_test = pd.read_csv(processed_data  / 'y_test.csv')

    lgbm_model = lgb.LGBMClassifier(
        colsample_bytree=1.0,
        learning_rate=0.05,
        max_depth=20,
        n_estimators=100,
        num_leaves=31,
        subsample=0.8
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

    model_path = models_dir / 'model.pkl'
    joblib.dump(lgbm_model, model_path)
                                                                                                                 

if __name__ == "__main__":
    main()
                                                                                                                                                                   