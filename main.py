#python
from email.policy import default
from turtle import setx, title
from typing import Optional
from enum import Enum


#pydantic
from pydantic import BaseModel
from pydantic import Field

#fastapi
from fastapi import FastAPI
from fastapi import Body, Query, Path, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import json
import numpy as np



app = FastAPI()
origins = ["*"]

pred_dict = {
        "0":"No diabetes",
        "1":"Prediabetes",
        "2":"Diabetes"
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tweet_model = tf.keras.models.load_model("tweets_resulting_model.h5")
diabetes_model = tf.keras.models.load_model("diabetes_resulting_model.h5")

tokenizer = None

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

class Tweet(BaseModel):
    tweet: str = Field(..., min_length=1, max_length=280)
    
class Diabetes(BaseModel):
    highBP: int = Field(..., ge=0, le=1)
    highChol: int = Field(..., ge=0, le=1)
    cholCheck:int = Field(..., ge=0, le=1)
    bmi:int = Field(..., ge=0, le=98)
    smoker:int = Field(..., ge=0, le=1)
    stroke:int = Field(..., ge=0, le=1)
    heartDiseaseorAttack:int = Field(..., ge=0, le=1)
    physActivity:int = Field(..., ge=0, le=1)
    fruits:int = Field(..., ge=0, le=1)
    veggies:int = Field(..., ge=0, le=1)
    hvyAlcoholConsump:int = Field(..., ge=0, le=1)
    anyHealthcare:int = Field(..., ge=0, le=1)
    noDocbcCost:int = Field(..., ge=0, le=1)
    genHlth:int = Field(..., ge=1, le=5)
    mentHlth:int = Field(..., ge=0, le=30)
    physhlth:int = Field(..., ge=0, le=30)
    diffWalk:int = Field(..., ge=0, le=1)
    sex :int = Field(..., ge=0, le=1)
    age :int = Field(..., ge=1, le=12)
    education:int = Field(..., ge=1, le=6)
    income:int = Field(..., ge=1, le=8)

@app.get("/")
def home():
    return {
        "Hello": "world"
    }


## MODELOS OCULAR
@app.post("/model/eye")
def analizar_imagen(image:UploadFile = File(...)):
    
    return {
        "name": image.filename,
        "cambio":"done"
    }


## MODELOS DEPRESION
@app.post("/model/depresion")
def analizar_tweet(tweet:Tweet = Body(...)):

    tweet = tweet.tweet

    tweet_norm = tweet.replace(r'[^a-zA-Z0-9\s{1}áéíóúüñÁÉÍÓÚÑ]', '')
    tweet_norm = tweet_norm.lower().strip().rstrip('\n').rstrip('\r\n')
    
    seq = tokenizer.texts_to_sequences(tweet_norm)
    x = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=34)
    
    prediction = tweet_model.predict(x)[0]
    if (prediction >= 0.5):
        diag = "Depresion"
        prob = prediction
    else:
        diag = "No depresion"
        prob = 1 - prediction
    
    return {
        "Diagnostico": diag,
        "pCorrecto": int(prediction*100),
        "precision": 99
    }

## MODELOS DIABETES
@app.post("/model/diabetes")
def analizar_diabetes(paciente:Diabetes = Body(...)):
    inputs = []

    inputs.append(paciente.highBP)
    inputs.append(paciente.highChol)
    inputs.append(paciente.cholCheck)
    inputs.append(np.log(paciente.bmi))
    inputs.append(paciente.smoker)
    inputs.append(paciente.stroke)
    inputs.append(paciente.heartDiseaseorAttack)
    inputs.append(paciente.physActivity)
    inputs.append(paciente.fruits)
    inputs.append(paciente.veggies)
    inputs.append(paciente.hvyAlcoholConsump)
    inputs.append(paciente.anyHealthcare)
    inputs.append(paciente.noDocbcCost)
    #genHlth = [0,0,0,0,0]
    #genHlth[paciente.genHlth - 1] = 1
    #inputs.extend(genHlth)
    inputs.append(paciente.genHlth)
    mentHlth = [0 for i in range(31)]
    mentHlth[paciente.mentHlth - 1] = 1
    inputs.extend(mentHlth)
    physhlth = [0 for i in range(31)]
    physhlth[paciente.physhlth - 1] = 1
    inputs.extend(physhlth)
    inputs.append(paciente.diffWalk)
    inputs.append(paciente.sex)
    age = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    age[paciente.age - 1] = 1
    inputs.extend(age)
    education = [0,0,0,0,0,0]
    education[paciente.education - 1] = 1
    inputs.extend(education)
    income = [0,0,0,0,0,0,0,0]
    income[paciente.income - 1] = 1
    inputs.extend(income)
    
    inputs = np.array(inputs).reshape(-1, 105)
    
    probs = diabetes_model.predict(inputs)
    
    pred = np.argmax(probs[0])
    
    diag = pred_dict[str(pred)]
    
    print("NO ES TENSORFLOW!!!!")
    print(diag)
    print(pred)
    print(probs[0][pred])
    
    return {
        "Diagnostico": probs.tolist(),
        "pCorrecto": probs.tolist(),
        "precision":86
    }
