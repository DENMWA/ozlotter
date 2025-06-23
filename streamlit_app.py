
# streamlit_app.py - OzLotter (Formula-Only Mode)

import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime

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
DATA_PATH = "data/historical_draws.csv"

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

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Lotto_icon.svg/2048px-Lotto_icon.svg.png", width=100)
st.sidebar.markdown("## OzLotter (Formula Mode Only)")
st.sidebar.markdown("Smart. Predictive. Strictly Formulaic.")

st.title("🎯 OzLotter — Ψⁿˡ(Ω) Formula Predictions Only")

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
                new_row = pd.DataFrame([{"date": date, "main": main, "supp": supp}])
                if os.path.exists(DATA_PATH):
                    df = pd.read_csv(DATA_PATH)
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    df = new_row
                df.to_csv(DATA_PATH, index=False)
                st.success(f"Saved: {main_numbers} + {supp_numbers}")
                st.rerun()
            except:
                st.error("⚠️ Please ensure 7 main + 3 supplementary numbers correctly entered.")

# Formula Prediction Mode
with st.expander("🔮 Generate Ψⁿˡ(Ω) Formula-Based Predictions"):
    num_sets = st.slider("How many prediction sets?", 5, 100, 20)
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
    st.dataframe(df_synthetic.sort_values(by="psi_score", ascending=False).head(10))
