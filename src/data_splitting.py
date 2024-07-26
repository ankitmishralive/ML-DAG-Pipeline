import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():

    current_path = Path(__file__)
    root = current_path.parent.parent

    interim_data = root / 'data' / 'interim' 
    processed_data = root / 'data' / 'processed'

    # Load data
    df = pd.read_csv(interim_data / 'cleaned_data.csv')
   

    X = df.drop('Potability', axis=1)
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv(processed_data / 'X_train.csv', index=False)
    X_test.to_csv(processed_data / 'X_test.csv', index=False)
    y_train.to_csv(processed_data / 'y_train.csv', index=False)
    y_test.to_csv(processed_data / 'y_test.csv', index=False)


if __name__ == "__main__":
    main()
