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



app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    return {
        "Diagnostico": "loool",
        "pCorrecto":90,
        "precision":85
    }

## MODELOS DIABETES
@app.post("/model/diabetes")
def analizar_tweet(paciente:Diabetes = Body(...)):
    
    return {
        "Diagnostico": "Resultado",
        "pCorrecto":70,
        "precision":80
    }
