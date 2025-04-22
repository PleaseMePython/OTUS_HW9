from fastapi import FastAPI, Query

from pydantic import BaseModel

from infer import predict

app = FastAPI()


class Health(BaseModel):
    pregnancies: float
    glucose: float
    bmi: float
    age: float


class Prediction(BaseModel):
    diabetes: bool


@app.get("/api/predict/", response_model=Prediction)
async def predict_view(health: Health = Query()):
    diabetes = predict(
        pregnancies=health.pregnancies,
        glucose=health.glucose,
        bmi=health.bmi,
        age=health.age,
    )
    return Prediction(diabetes=diabetes)
