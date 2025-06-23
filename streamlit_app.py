# streamlit_app.py - OzLotter: Intelligent Oz Lotto Prediction Engine

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="OzLotter", page_icon="🎯", layout="wide")

# -------------------------------
# Configurable λ-weights
# -------------------------------
LAMBDA = {
    'lambda1': 1.0,
    'lambda2': 1.0,
    'lambda3': 1.0,
    'lambda4': 1.0,
    'lambda5': 1.0,
    'lambda6': 1.0,
    'lambda7': 1.0
}

ALPHA = 0.25
MODEL_PATH = "models/oz_lotto_rf.joblib"
DATA_PATH = "data/historical_draws.csv"

# -------------------------------
# Ψⁿˡ(Ω) Scoring Function
# -------------------------------
def psi_nabla_lambda(X):
    weighted_sum = (
        LAMBDA['lambda1'] * X['fourier'] +
        LAMBDA['lambda2'] * X['entropy'] +
        LAMBDA['lambda3'] * X['mahalanobis'] +
        LAMBDA['lambda4'] * X['bayesian'] +
        LAMBDA['lambda5'] * X['penalty'] +
        LAMBDA['lambda6'] * X['momentum'] +
        LAMBDA['lambda7'] * X['integral']
    )
    amplified = weighted_sum * np.exp(ALPHA * X['avg_freq'])
    noise = np.random.normal(0, 0.1, size=len(X))
    return amplified + noise

# -------------------------------
# ML Classifier Training (Self-Retraining)
# -------------------------------
def train_classifier(data):
    drop_cols = ['division', 'date', 'main', 'supp']
    features = data.drop(columns=[col for col in drop_cols if col in data.columns], errors='ignore')
    features = features.select_dtypes(include=[np.number])
    labels = data['division']
    if features.empty:
        raise ValueError("No numeric features left for training after filtering.")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODEL_PATH)
    return rf

# -------------------------------
# Load Model or Train if Absent
# -------------------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except ValueError as e:
            st.warning("⚠️ Model is incompatible with current environment. Attempting retrain...")
            df = load_data()
            if not df.empty and 'division' in df.columns:
                return train_classifier(df)
            else:
                st.error("Model cannot be reloaded or retrained: No suitable data available.")
                return None
    else:
        st.warning("No model found. Please train with historical data.")
        return None

# -------------------------------
# Load Data
# -------------------------------
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()

# Sidebar branding
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Lotto_icon.svg/2048px-Lotto_icon.svg.png", width=100)
st.sidebar.markdown("## OzLotter")
st.sidebar.markdown("Smart. Predictive. Evolving.")

st.title("🎯 OzLotter — Intelligent Lotto Insights with Ψⁿˡ(Ω)")
st.markdown("Welcome to OzLotter — your AI-powered assistant for smarter Oz Lotto predictions.")

# Draw Entry Form
with st.expander("📥 Enter New Oz Lotto Draw"):
    with st.form("draw_entry_form"):
        date = st.date_input("Draw Date")
        main = st.text_input("7 Main Numbers (comma separated)")
        supp = st.text_input("3 Supplementary Numbers (comma separated)")
        submitted = st.form_submit_button("Submit Draw")

        if submitted:
            try:
                main_numbers = [int(x.strip()) for x in main.split(",")]
                supp_numbers = [int(x.strip()) for x in supp.split(",")]
                assert len(main_numbers) == 7 and len(supp_numbers) == 3

                st.success(f"Draw saved: {date} | Main: {main_numbers} | Supps: {supp_numbers}")
                new_row = pd.DataFrame([{"date": date, "main": main, "supp": supp}])
                if os.path.exists(DATA_PATH):
                    df = pd.read_csv(DATA_PATH)
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    df = new_row
                df.to_csv(DATA_PATH, index=False)
                st.rerun()
            except:
                st.error("Invalid input. Please ensure 7 main and 3 supplementary numbers.")

# Formula-Based Prediction Generator
with st.expander("🧠 Generate Formula-Based Predictions"):
    model = load_model()
    if model:
        num_sets = st.slider("Number of prediction sets", 5, 100, 20)
        df_synthetic = pd.DataFrame({
            'fourier': np.random.uniform(0, 1, num_sets),
            'entropy': np.random.uniform(0, 1, num_sets),
            'mahalanobis': np.random.uniform(0, 1, num_sets),
            'bayesian': np.random.uniform(0, 1, num_sets),
            'penalty': np.random.uniform(0, 1, num_sets),
            'momentum': np.random.uniform(0, 1, num_sets),
            'integral': np.random.uniform(0, 1, num_sets),
            'avg_freq': np.random.uniform(0, 1, num_sets),
        })
        df_synthetic['psi_score'] = psi_nabla_lambda(df_synthetic)
        feature_cols = [col for col in df_synthetic.columns if col != 'psi_score']
        df_synthetic['division_pred'] = model.predict(df_synthetic[feature_cols])
        df_synthetic['div_prob'] = model.predict_proba(df_synthetic[feature_cols]).max(axis=1)
        st.write("### Top Ranked Formula-Based Prediction Sets")
        st.dataframe(df_synthetic.sort_values(by='psi_score', ascending=False).head(10))
