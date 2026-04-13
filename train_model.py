"""
train_model.py
Run this ONCE to train the model and save it as model.pkl
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── 1. Load & Clean ──────────────────────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.xls')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# ── 2. Feature Engineering ───────────────────────────────────────────────────
df['TenureGroup'] = pd.cut(
    df['tenure'], bins=[-1, 12, 24, 48, 72],
    labels=['0-1yr', '1-2yr', '2-4yr', '4+yr']
)
df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['HighRisk'] = (
    (df['Contract'] == 'Month-to-month') &
    (df['InternetService'] == 'Fiber optic')
).astype(int)

# ── 3. Encode ────────────────────────────────────────────────────────────────
df_final = pd.get_dummies(df, drop_first=True)
feature_columns = [c for c in df_final.columns if c != 'Churn']
X = df_final[feature_columns]
y = df_final['Churn']

# ── 4. Split + SMOTE + Scale ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(X_test)

# ── 5. Train Ensemble ────────────────────────────────────────────────────────
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, eval_metric='logloss', random_state=42)
rf  = RandomForestClassifier(n_estimators=199, max_depth=27,
                              min_samples_split=3, random_state=42)
lr  = LogisticRegression(max_iter=2000, class_weight='balanced')

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('xgb', xgb)],
    voting='soft'
)
ensemble.fit(X_res_scaled, y_res)

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred  = ensemble.predict(X_test_scaled)
y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── 7. Save artefacts ────────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model':    ensemble,
        'scaler':   scaler,
        'features': feature_columns
    }, f)

print("\n✅  Saved → model.pkl")
