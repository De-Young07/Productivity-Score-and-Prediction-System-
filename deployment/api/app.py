from fastapi import FastAPI
from inference.predict import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Productivity Prediction API"}

@app.post("/predict")

def predict_productivity(data: dict):

    result = predict(data)

    return {
        "predicted_productivity_score": result
    }