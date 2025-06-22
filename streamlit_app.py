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

# -------------------------------
# App Interface
import shap
import matplotlib.pyplot as plt

# Sidebar branding
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Lotto_icon.svg/2048px-Lotto_icon.svg.png", width=100)
st.sidebar.markdown("## OzLotter")
st.sidebar.markdown("Smart. Predictive. Evolving.")
# -------------------------------
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

# Upload Predictions
with st.expander("📊 Upload Prediction Feature Set for Scoring"):
    upload = st.file_uploader("Upload CSV with required feature columns", type="csv")
    if upload:
        df_pred = pd.read_csv(upload)
        if all(col in df_pred.columns for col in ['fourier','entropy','mahalanobis','bayesian','penalty','momentum','integral','avg_freq']):
            df_pred['psi_score'] = psi_nabla_lambda(df_pred)
            model = load_model()
            if model:
                df_pred['division_pred'] = model.predict(df_pred)
                df_pred['div_prob'] = model.predict_proba(df_pred).max(axis=1)
            st.dataframe(df_pred.head())
        else:
            st.error("CSV missing required columns.")

# SHAP Analysis + Retrain
with st.expander("📊 Feature Impact (SHAP Analysis)"):
    df_hist = load_data()
    model = None
    try:
        model = load_model()
    except Exception as e:
        st.warning("Model loading failed during SHAP analysis. Attempting fallback...")
        if not df_hist.empty and 'division' in df_hist.columns:
            model = train_classifier(df_hist)

    if model and not df_hist.empty and 'division' in df_hist.columns:
        try:
            explainer = shap.Explainer(model, df_hist.drop(columns=['division', 'date', 'main', 'supp'], errors='ignore'))
            shap_values = explainer(df_hist.drop(columns=['division', 'date', 'main', 'supp'], errors='ignore'))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("### Feature Importance Summary")
            shap.summary_plot(shap_values, df_hist.drop(columns=['division', 'date', 'main', 'supp'], errors='ignore'), plot_type="bar")
            st.pyplot(bbox_inches='tight')
        except Exception as ex:
            st.error("❌ SHAP analysis failed: " + str(ex))

    if st.button("Retrain Now"):
        df_hist = load_data()
        if 'division' in df_hist.columns:
            model = train_classifier(df_hist)
            st.success("Model retrained and saved.")
        else:
            st.error("Division labels missing in dataset. Cannot train.")