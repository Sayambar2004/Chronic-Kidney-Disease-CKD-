from fastapi import FastAPI
import pandas as pd
import joblib

# Load artifacts

model = joblib.load(
    "stack_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

selected_features = joblib.load(
    "selected_features.pkl"
)

app = FastAPI()


@app.get("/")
def home():

    return {
        "message":
        "Disease Risk Stratification API"
    }


@app.post("/predict")
def predict(data: dict):

    # Convert request to dataframe

    df = pd.DataFrame([data])

    # Ensure same feature order

    df = df[selected_features]

    # Scale

    scaled_data = scaler.transform(df)

    # Predict

    prediction = model.predict(
        scaled_data
    )[0]

    probability = model.predict_proba(
        scaled_data
    )[0][1]

    return {
        "prediction": int(prediction),
        "risk_level":
            "High Risk (CKD)"
            if prediction == 1
            else "Low Risk",

        "probability": float(probability)
    }