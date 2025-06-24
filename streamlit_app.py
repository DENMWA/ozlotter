
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("ozlotto_division_predictor.pkl")

# Formula scoring function (simplified)
def psi_nabla_lambda(X):
    LAMBDA = {
        'lambda1': 1.0, 'lambda2': 1.0, 'lambda3': 1.0, 'lambda4': 1.0,
        'lambda5': 1.0, 'lambda6': 1.0, 'lambda7': 1.0
    }
    ALPHA = 0.01
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

# UI
st.title("Oz Lotto Predictor")
st.markdown("Generate 200 number sets, score them, and show top 100 by ML prediction.")

if st.button("Generate Predictions"):
    np.random.seed(42)
    sets = []
    for _ in range(200):
        numbers = np.sort(np.random.choice(range(1, 48), 7, replace=False))
        gaps = np.diff(numbers)
        features = {
            'fourier': np.sum(np.sin(numbers)),
            'entropy': -np.sum((np.bincount(numbers, minlength=48)[1:] / 7).clip(min=1e-9) * np.log2((np.bincount(numbers, minlength=48)[1:] / 7).clip(min=1e-9))),
            'mahalanobis': np.sqrt(np.sum(((numbers - 24) / 10) ** 2)),
            'bayesian': np.mean(gaps),
            'penalty': np.sum(gaps < 3),
            'momentum': np.std(gaps),
            'integral': np.sum(numbers),
            'avg_freq': np.mean(numbers)
        }
        sets.append({**features, 'numbers': numbers})

    df = pd.DataFrame(sets)
    df['score'] = psi_nabla_lambda(df)
    df['division_pred'] = model.predict(df.iloc[:, :-2])
    df['div_prob'] = model.predict_proba(df.iloc[:, :-2]).max(axis=1)

    top_100 = df.sort_values(by=['division_pred', 'div_prob'], ascending=[True, False]).head(100)
    st.dataframe(top_100[['numbers', 'score', 'division_pred', 'div_prob']])
    st.download_button("Download Top 100", data=top_100.to_csv(index=False), file_name="top_100_predictions.csv")
