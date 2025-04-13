from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from text_loader.loader_new import DataLoader
import joblib
import xgboost as xgb
import numpy as np

mlflow.set_tracking_uri('data')

app = FastAPI()

booster = xgb.Booster()
booster.load_model("model-inference-endpoint/saved_model/model.json")

#model = joblib.load("model-inference-endpoint/saved_model/model.pkl")
vectorizer = joblib.load("model-inference-endpoint/saved_model/vectorizer.pkl")

loader_new = DataLoader("", False)
#loader_new.vectorizer = vectorizer

class InputText(BaseModel):
    input_texts: str

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    # TODO - task 2 
    # -----------------------------------
    # Goal: our goal is to complete the implementation of this function, 
    #       which takes input data and returns a prediction result from a pre-trained model.

    #Step1 : Cleaning the input text
    try:
        cleaned_text = loader_new.clean_text(input_data.input_texts)

        #Step2 : Vectorized the cleaned text
        vectorized_text = vectorizer.transform([cleaned_text])

        #Step3 : Predict model labels and probability of confidence
        dmatrix = xgb.DMatrix(vectorized_text)

        prob = booster.predict(dmatrix)[0]
        pred = int(prob >= 0.5)
        
        #pred = model.predict(vectorized_text)[0]
        #prob = model.predict_proba(vectorized_text)[0]

        if pred == 1:
            label = "Republican" 
        else:
            label = "Democrat"
            
        confidence = round(prob if pred == 1 else 1 - prob, 3)
    except Exception as e:
        return {"error": str(e)}

    return {
        "input" : input_data.input_texts,
        "cleaned_text" : cleaned_text,
        "party_prediction" : label,
        "confidence_score" : confidence
    }
        
    pass