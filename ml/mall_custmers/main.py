from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal
from utils import new_instance


gender_option=Literal['Male','Female']


class custom_data(BaseModel):
    Age: int
    Annual_Income: float
    Spending_Score: float

app=FastAPI(debug=True)

@app.post('/predict')
async def predict(    
    data:custom_data,
    Gender:gender_option =Query(...)
):
      new_data = {
        'Genre': Gender ,
        'Age': data.Age,
        'Annual Income (k$)': data.Annual_Income,
        'Spending Score (1-100)': data.Spending_Score
    }
      
      cluster=new_instance(new_data)
      
      return {f"the cluster is {cluster}"}
