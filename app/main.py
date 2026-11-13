from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

app = FastAPI(
    title="Smart Retail Analytics API",
    description="Demo API for demand forecasting, purchase prediction, customer segmentation, and anomaly detection.",
    version="1.0.0",
)

def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Model file {path.name} not found. Run `python generate_data.py` "
            "and then `python train_models.py` from the project root."
        )
    return joblib.load(path)

@app.get("/")
def home():
    return {
        "message": "Smart Retail Analytics API is running",
        "docs": "/docs",
        "endpoints": ["/forecast", "/purchase", "/segment", "/anomaly"]
    }

class ForecastInput(BaseModel):
    price: float = Field(gt=0, description="Product price")
    discount: float = Field(ge=0, le=1, description="Discount fraction, e.g. 0.15 for 15%")
    advertising_spend: float = Field(ge=0)
    previous_sales: int = Field(ge=0)

@app.post("/forecast")
def forecast(data: ForecastInput):
    bundle = load_model("forecast_model.joblib")
    row = pd.DataFrame([data.model_dump()])[bundle["features"]]
    prediction = float(bundle["model"].predict(row)[0])
    return {"predicted_sales_quantity": max(0, round(prediction, 2))}

class PurchaseInput(BaseModel):
    price: float = Field(gt=0)
    discount: float = Field(ge=0, le=1)
    advertising_spend: float = Field(ge=0)
    customer_age: int = Field(ge=18, le=120)
    customer_spend: float = Field(ge=0)

@app.post("/purchase")
def purchase(data: PurchaseInput):
    bundle = load_model("purchase_model.joblib")
    row = pd.DataFrame([data.model_dump()])[bundle["features"]]
    model = bundle["model"]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {
        "purchase_prediction": "Likely to purchase" if prediction == 1 else "Unlikely to purchase",
        "purchase_probability": round(probability, 3)
    }

class SegmentInput(BaseModel):
    customer_spend: float = Field(ge=0)
    purchase_frequency: int = Field(ge=0)

@app.post("/segment")
def segment(data: SegmentInput):
    bundle = load_model("segment_model.joblib")
    row = pd.DataFrame([data.model_dump()])[bundle["features"]]
    cluster = int(bundle["model"].predict(row)[0])
    return {
        "customer_segment_id": cluster,
        "note": "Cluster IDs are arbitrary. Compare cluster averages in the dataset before naming them."
    }

class AnomalyInput(BaseModel):
    sales_quantity: float = Field(ge=0)
    price: float = Field(gt=0)
    discount: float = Field(ge=0, le=1)

@app.post("/anomaly")
def anomaly(data: AnomalyInput):
    bundle = load_model("anomaly_model.joblib")
    row = pd.DataFrame([data.model_dump()])[bundle["features"]]
    scaled_row = bundle["scaler"].transform(row)
    # DBSCAN has no predict() for new samples. Assign to the nearest learned core
    # point within eps; otherwise flag as an anomaly.
    model = bundle["model"]
    core_points = model.components_
    if len(core_points) == 0:
        return {"is_anomaly": True, "label": -1}
    distances = np.sqrt(((core_points - scaled_row[0]) ** 2).sum(axis=1))
    nearest = float(distances.min())
    is_anomaly = nearest > model.eps
    return {
        "is_anomaly": bool(is_anomaly),
        "label": -1 if is_anomaly else 0,
        "nearest_core_point_distance": round(nearest, 3)
    }
