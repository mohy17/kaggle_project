from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from utils import x_new  

app = FastAPI(debug=True)

class PassengerInput(BaseModel):
    SibSp: int
    Parch: int
    Fare: float
    Embarked: Literal["C", "Q", "S"]  
    Age: float
    Pclass: Literal["1","2","3"]
    Sex: Literal["male", "female"]  

@app.post("/predict")
async def predict_survival(passenger: PassengerInput):
    # Convert the input into a dictionary format
    input_data = passenger.dict()
    
    # Call the prediction function
    prediction = x_new(input_data)
    
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
