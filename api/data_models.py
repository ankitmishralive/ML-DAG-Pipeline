

from pydantic import BaseModel



class PredictionDataset(BaseModel):
        ph                : float
        Hardness          :float
        Solids            :float
        Chloramines        :float
        Sulfate            :float
        Conductivity       :float
        Organic_carbon     :float
        Trihalomethanes    :float
        Turbidity          :float


