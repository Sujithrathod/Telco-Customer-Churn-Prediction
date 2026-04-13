"""
api.py  –  FastAPI backend
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load model artefacts ─────────────────────────────────────────────────────
MODEL_PATH = Path("model.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError("model.pkl not found — run train_model.py first.")

with open(MODEL_PATH, "rb") as f:
    artefacts = pickle.load(f)

model    = artefacts["model"]
scaler   = artefacts["scaler"]
FEATURES = artefacts["features"]

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Telco Churn Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input schema ─────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender: str                  = Field(..., example="Male")
    SeniorCitizen: int           = Field(..., example=0)
    Partner: str                 = Field(..., example="Yes")
    Dependents: str              = Field(..., example="No")
    tenure: int                  = Field(..., example=12)
    PhoneService: str            = Field(..., example="Yes")
    MultipleLines: str           = Field(..., example="No")
    InternetService: str         = Field(..., example="Fiber optic")
    OnlineSecurity: str          = Field(..., example="No")
    OnlineBackup: str            = Field(..., example="No")
    DeviceProtection: str        = Field(..., example="No")
    TechSupport: str             = Field(..., example="No")
    StreamingTV: str             = Field(..., example="No")
    StreamingMovies: str         = Field(..., example="No")
    Contract: str                = Field(..., example="Month-to-month")
    PaperlessBilling: str        = Field(..., example="Yes")
    PaymentMethod: str           = Field(..., example="Electronic check")
    MonthlyCharges: float        = Field(..., example=70.35)
    TotalCharges: float          = Field(..., example=844.20)

# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(data: CustomerInput) -> np.ndarray:
    d = data.dict()

    # Engineered features
    d['TenureGroup'] = pd.cut(
        [d['tenure']], bins=[-1, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4+yr']
    )[0]
    d['ChargesPerMonth'] = d['TotalCharges'] / (d['tenure'] + 1)
    d['HighRisk'] = int(
        d['Contract'] == 'Month-to-month' and
        d['InternetService'] == 'Fiber optic'
    )

    row = pd.DataFrame([d])
    row_enc = pd.get_dummies(row, drop_first=True)

    # Align columns to training feature set
    row_enc = row_enc.reindex(columns=FEATURES, fill_value=0)
    return scaler.transform(row_enc)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Telco Churn Predictor API is running 🚀"}

@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        X = build_features(customer)
        churn_prob  = float(model.predict_proba(X)[0][1])
        churn_label = int(churn_prob >= 0.45)   # tuned threshold

        risk = (
            "🔴 High Risk"   if churn_prob >= 0.65 else
            "🟡 Medium Risk" if churn_prob >= 0.40 else
            "🟢 Low Risk"
        )
        return {
            "churn_probability": round(churn_prob, 4),
            "churn_prediction":  churn_label,
            "risk_level":        risk,
            "recommendation": (
                "Immediate retention action needed — offer discount or upgrade."
                if churn_label else
                "Customer appears stable. Monitor monthly charges."
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(FEATURES)}
