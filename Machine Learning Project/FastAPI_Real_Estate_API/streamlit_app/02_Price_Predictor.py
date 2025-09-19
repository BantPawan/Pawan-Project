# 02_Price_Predictor.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import pickle

# Define base directory and file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Dataset", "pipeline_compressed.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "df.pkl")

# Load DataFrame
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)
else:
    st.error(f"Data file not found at {DATA_PATH}")
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

# Load model pipeline
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
else:
    st.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# --- Your predictor UI logic here ---
# ----------------------------
# Utility
# ----------------------------
def format_price(value: float) -> str:
    return f"₹ {value:,.2f} Cr"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Price Predictor", page_icon="🏠")
st.title("🏠 Real Estate Price Predictor")
st.write("Enter property details below to predict the price range.")

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", df["property_type"].unique())
    sector = st.selectbox("Sector", df["sector"].unique())
    bedroom = st.selectbox("Bedrooms", sorted(df["bedRoom"].unique()))
    bathroom = st.selectbox("Bathrooms", sorted(df["bathroom"].unique()))
    balcony = st.selectbox("Balconies", sorted(df["balcony"].unique()))

with col2:
    age_possession = st.selectbox("Age / Possession", df["agePossession"].unique())
    built_up_area = st.number_input("Built-up Area (sq ft)", min_value=500, max_value=10000, step=50)
    servant_room = st.selectbox("Servant Room", df["servant room"].unique())
    store_room = st.selectbox("Store Room", df["store room"].unique())
    furnishing_type = st.selectbox("Furnishing Type", df["furnishing_type"].unique())

luxury_category = st.selectbox("Luxury Category", df["luxury_category"].unique())
floor_category = st.selectbox("Floor Category", df["floor_category"].unique())

# Collect input
input_data = pd.DataFrame([[
    property_type, sector, bedroom, bathroom, balcony, age_possession,
    built_up_area, servant_room, store_room, furnishing_type,
    luxury_category, floor_category
]], columns=[
    "property_type", "sector", "bedRoom", "bathroom", "balcony",
    "agePossession", "built_up_area", "servant room", "store room",
    "furnishing_type", "luxury_category", "floor_category"
])

# Prediction
if st.button("💰 Predict Price"):
    try:
        base_price = np.expm1(pipeline.predict(input_data))[0]
        low_price, high_price = base_price - 0.22, base_price + 0.22
        st.success(f"Predicted Price Range: {format_price(low_price)} - {format_price(high_price)}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
