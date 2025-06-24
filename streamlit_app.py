# streamlit_app.py — Startup Fallback for OzLotter
import streamlit as st
import os

st.set_page_config(page_title="OzLotter", layout="wide")
st.title("✅ OzLotter — App Loaded Successfully")

try:
    if os.path.exists("models/oz_lotto_rf.joblib"):
        st.success("Model file located at models/oz_lotto_rf.joblib")
    else:
        st.warning("⚠️ Model file not found yet. Train the model using historical data.")
except Exception as e:
    st.error(f"App encountered an error during startup: {e}")
