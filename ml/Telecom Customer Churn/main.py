from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from utils import predict_new_instance  

app = FastAPI(debug=True)

# Define input data
class CustomerData(BaseModel):
    gender: Literal['Male', 'Female']  
    SeniorCitizen: int = Field(..., description="1 for 'yes' and 0 for 'no'") 
    Partner: Literal['Yes', 'No'] = 'No'  
    Dependents: Literal['Yes', 'No'] = 'No'  
    tenure: int = Field(..., gt=0, description="Must be greater than 0") 
    PhoneService: Literal['Yes', 'No'] 
    MultipleLines: Literal['Yes', 'No', 'No phone service'] = 'No' 
    InternetService: Literal['DSL', 'Fiber optic', 'No']  
    OnlineSecurity: Literal['Yes', 'No', 'No internet service'] = 'No'  
    OnlineBackup: Literal['Yes', 'No', 'No internet service'] = 'No'  
    DeviceProtection: Literal['Yes', 'No', 'No internet service'] = 'No'  
    TechSupport: Literal['Yes', 'No', 'No internet service'] = 'No' 
    StreamingTV: Literal['Yes', 'No', 'No internet service'] = 'No'  
    StreamingMovies: Literal['Yes', 'No', 'No internet service'] = 'No'  
    Contract: Literal['Month-to-month', 'One year', 'Two year']  
    PaperlessBilling: Literal['Yes', 'No'] = 'Yes' 
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] 
    MonthlyCharges: float = Field(..., gt=0)  
    TotalCharges: float = Field(..., gt=0)  

@app.post("/predict")
def predict(instance: CustomerData):
    instance_dict = instance.dict()
    prediction = predict_new_instance(instance_dict)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
