from fastapi import fastAPI
import pickle 


app = fastAPI(
    title="Water Potability Prediction API",
    description="API for predicting water potability based on various chemical properties.",
)

