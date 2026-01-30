from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# ✅ Correct way to load model on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "diabetes_model.sav")

diabetes_model = pickle.load(open(model_path, "rb"))

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/diabetes_prediction")
def diabetes_pred(input_data: ModelInput):

    input_list = [
        input_data.Pregnancies,
        input_data.Glucose,
        input_data.BloodPressure,
        input_data.SkinThickness,
        input_data.Insulin,
        input_data.BMI,
        input_data.DiabetesPedigreeFunction,
        input_data.Age
    ]

    prediction = diabetes_model.predict([input_list])

    if prediction[0] == 0:
        return {"prediction": "The person is not diabetic"}
    else:
        return {"prediction": "The person is diabetic"}
