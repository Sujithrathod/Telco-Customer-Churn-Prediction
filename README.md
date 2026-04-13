# 📡 Telco Customer Churn Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-customer-churn-predictor-web.streamlit.app/)

🔗 **Live Demo:** [https://telco-customer-churn-predictor-web.streamlit.app](https://telco-customer-churn-predictor-web.streamlit.app/)

An end-to-end machine learning application that predicts customer churn for a telecommunications company. The system uses an ensemble of three classifiers to identify at-risk customers and recommend retention actions in real time.

---

## 📌 Problem Statement

Customer churn is one of the most critical challenges faced by the telecommunications industry. When a customer leaves (churns), the company loses not only recurring revenue but also the acquisition cost invested in that customer. Studies show that acquiring a new customer costs **5–7× more** than retaining an existing one.

This project aims to:

1. **Predict** whether a telecom customer is likely to churn based on their demographics, service subscriptions, and billing history
2. **Quantify** churn risk with a probability score and categorize it into actionable risk levels (High / Medium / Low)
3. **Provide** retention recommendations so customer success teams can proactively intervene
4. **Serve** predictions through a REST API and an interactive web UI for real-time decision-making

---

## 📊 Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|----------|-------|
| Rows | 7,043 customers |
| Features | 21 (demographics, services, billing) |
| Target | `Churn` (Yes/No) |
| Class Imbalance | ~73% No Churn / ~27% Churn |

### Key Features Used

| Category | Features |
|----------|----------|
| **Demographics** | Gender, Senior Citizen, Partner, Dependents |
| **Account Info** | Tenure (months), Contract type, Paperless Billing, Payment Method |
| **Services** | Phone, Multiple Lines, Internet, Online Security, Backup, Device Protection, Tech Support, Streaming TV/Movies |
| **Billing** | Monthly Charges, Total Charges |
| **Engineered** | Tenure Group, Charges Per Month, High Risk Flag |

---

## 🧠 Methodology

### Feature Engineering

Three custom features are created to boost model performance:

| Feature | Logic | Rationale |
|---------|-------|-----------|
| `TenureGroup` | Bins tenure into `0-1yr`, `1-2yr`, `2-4yr`, `4+yr` | Captures non-linear relationship between tenure and churn |
| `ChargesPerMonth` | `TotalCharges / (tenure + 1)` | Normalizes spending across different tenures |
| `HighRisk` | `1` if Month-to-month contract **AND** Fiber optic internet | Captures the highest-churn customer segment |

### Handling Class Imbalance

The dataset has a 73/27 class split. **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training set to generate synthetic churn samples and balance the classes.

### Model Architecture

A **Soft Voting Ensemble** combining three diverse classifiers:

```
                    ┌─────────────────────┐
                    │  Voting Classifier  │
                    │   (Soft Voting)     │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌─────────────┐  ┌──────────┐
     │ Random     │  │ Logistic    │  │ XGBoost  │
     │ Forest     │  │ Regression  │  │          │
     │ (199 trees)│  │ (balanced)  │  │(300 trees│
     │ depth=27   │  │ max_iter=   │  │ depth=6) │
     │            │  │ 2000        │  │          │
     └────────────┘  └─────────────┘  └──────────┘
```

- **Random Forest** — Captures non-linear patterns and feature interactions
- **Logistic Regression** — Provides linear decision boundary with class weight balancing
- **XGBoost** — Gradient-boosted trees for sequential error correction

Soft voting averages the predicted probabilities from all three models, producing a more robust and calibrated prediction than any single model.

### Prediction Threshold

The default classification threshold is tuned to **0.45** (instead of the standard 0.50) to favor **recall** — catching more actual churners at the cost of slightly more false positives. In a business context, missing a churner is more costly than incorrectly flagging a loyal customer.

### Risk Levels

| Risk Level | Probability Range | Action |
|------------|-------------------|--------|
| 🔴 High Risk | ≥ 65% | Immediate retention action — offer discount or upgrade |
| 🟡 Medium Risk | 40% – 65% | Monitor closely, consider proactive outreach |
| 🟢 Low Risk | < 40% | Customer appears stable, monitor monthly charges |

---

## 🏗️ Architecture

```
┌────────────────────┐        HTTP POST         ┌────────────────────┐
│                    │      /predict             │                    │
│   Streamlit UI     │ ───────────────────────►  │   FastAPI Backend  │
│   (app.py)         │                           │   (api.py)         │
│                    │  ◄───────────────────────  │                    │
│   Port: 8501       │    JSON response          │   Port: 8000       │
└────────────────────┘                           └────────┬───────────┘
                                                          │
                                                          │ loads
                                                          ▼
                                                 ┌────────────────────┐
                                                 │    model.pkl       │
                                                 │  ┌──────────────┐  │
                                                 │  │ Ensemble     │  │
                                                 │  │ Scaler       │  │
                                                 │  │ Feature List │  │
                                                 │  └──────────────┘  │
                                                 └────────────────────┘
```

---

## 📁 Project Structure

```
telco-churn-predictor/
├── train_model.py       # Trains the ensemble model and saves model.pkl
├── api.py               # FastAPI REST API backend
├── app.py               # Streamlit interactive frontend
├── model.pkl            # Trained model artifacts (ensemble + scaler + features)
├── requirements.txt     # Python dependencies (pinned versions)
├── Dockerfile           # Docker container for API deployment
├── .gitignore           # Git ignore rules
├── WA_Fn-UseC_-Telco-Customer-Churn.xls  # Dataset
├── Telco_Churn_Prediction_Structured.ipynb  # Exploratory analysis notebook
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-predictor.git
cd telco-churn-predictor
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

This creates `model.pkl` containing the trained ensemble, scaler, and feature list. Training prints a classification report and ROC-AUC score to the console.

### 4. Start the API Server

```bash
uvicorn api:app --reload --port 8000
```

- API Root: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 5. Start the Streamlit UI

Open a **second terminal**:

```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## 📡 API Reference

### `GET /`

Returns API status message.

### `GET /health`

Returns health check with model feature count.

### `POST /predict`

Predicts churn risk for a single customer.

**Request Body:**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 210.0
}
```

**Response:**

```json
{
  "churn_probability": 0.7821,
  "churn_prediction": 1,
  "risk_level": "🔴 High Risk",
  "recommendation": "Immediate retention action needed — offer discount or upgrade."
}
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t churn-api .

# Run the container
docker run -p 8000:8000 churn-api
```

---

## ☁️ Cloud Deployment

🔗 **Live App:** [https://telco-customer-churn-predictor-web.streamlit.app](https://telco-customer-churn-predictor-web.streamlit.app/)

| Component | Platform | Tier |
|-----------|----------|------|
| FastAPI API | [Render.com](https://render.com) | Free |
| Streamlit UI | [Streamlit Community Cloud](https://share.streamlit.io) | Free |

**Render Start Command:**
```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Streamlit Secrets** (set in Streamlit Cloud dashboard):
```toml
API_URL = "https://your-render-app.onrender.com"
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| ML Models | Scikit-learn, XGBoost, Imbalanced-learn (SMOTE) |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Data | Pandas, NumPy |
| Containerization | Docker |

---

## 📄 License

This project is for educational purposes. The dataset is provided by IBM and available on Kaggle under public domain.
