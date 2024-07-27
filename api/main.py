from pathlib import Path
from fastapi import FastAPI
import pandas as pd 
import joblib 
import uvicorn 


from data_models import PredictionDataset


app = FastAPI(
    title="Potability Prediction API",
    description="API for predicting water potability based on various chemical properties."
)

current_path = Path(__file__)
root = current_path.parent.parent
model_path =root / 'models'/ 'model.pkl'
model = joblib.load(model_path)



@app.get('/')
def home():
    return "Welcome to Water Potability Prediction API"


@app.post("/predictions")
def do_predictions(test_data:PredictionDataset):
    
    X_test = pd.DataFrame({
        
        'ph': test_data.ph,
        'Hardness': test_data.Hardness,
        'Solids': test_data.Solids,
        'Chloramines': test_data.Chloramines,
        'Sulfate': test_data.Sulfate,
        'Conductivity': test_data.Conductivity,
        'Organic_carbon': test_data.Organic_carbon,
        'Trihalomethanes': test_data.Trihalomethanes,
        'Turbidity': test_data.Turbidity,
    
    }, index=[0]   # Using index=[0] in pd.DataFrame ensures that the DataFrame is created with a single row. 
      )

    prediction = model.predict(X_test)  
    if prediction == 1:
        return {"prediction": "Potable Water is Consumable"}
    else:
        return {"prediction": "Not Potable"}



if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)


