import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Model and Supporting Files ---
try:
    model = joblib.load("best_car_model.pkl")
    df = pd.read_csv("clean_car_dataset.csv")
    score_df = pd.read_csv("model_scores.csv")
    with open("selected_model.txt", "r") as f:
        selected_model_name = f.read().strip()
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor")

# --- Show Model Info ---
st.markdown("### 🤖 Model Used")
st.success(f"**{selected_model_name}** (based on highest R²)")

st.markdown("### 📊 Model Evaluation Metrics")
st.dataframe(score_df.set_index("name").round(2))

st.markdown("---")

# --- Input Section ---
st.markdown("### 🧾 Enter Car Details to Predict Price")

def user_input_features():
    input_data = {}
    for col in df.columns:
        if col == 'price':
            continue
        elif df[col].dtype == 'object':
            options = sorted(df[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(f"{col}:", options)
        else:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            default_val = int(df[col].median())
            input_data[col] = st.slider(f"{col}:", min_val, max_val, default_val)
    return pd.DataFrame([input_data])

input_df = user_input_features()

# --- Prediction ---
if st.button("🚀 Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Car Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
