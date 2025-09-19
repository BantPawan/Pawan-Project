import streamlit as st
import requests
import pandas as pd
import os
import json

st.set_page_config(page_title="Price Predictor", page_icon="🏠")
st.title("🏠 Real Estate Price Predictor")
st.write("Enter property details below to predict the price range.")

# FastAPI backend URL (will be updated for Render deployment)
API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/predict_price")

# Load dataset for dropdown options
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "df.pkl")

if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "rb") as f:
        df = pd.read_pickle(f)
    st.success("✅ Dataset loaded successfully")
else:
    st.error("❌ df.pkl not found in Dataset folder")
    st.stop()

# Utility function to format price
def format_price(value: float) -> str:
    return f"₹ {value:,.2f} Cr"

# Streamlit UI
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
input_data = {
    "property_type": property_type,
    "sector": sector,
    "bedrooms": float(bedroom),
    "bathroom": float(bathroom),
    "balcony": balcony,
    "property_age": age_possession,
    "built_up_area": float(built_up_area),
    "servant_room": float(servant_room),
    "store_room": float(store_room),
    "furnishing_type": furnishing_type,
    "luxury_category": luxury_category,
    "floor_category": floor_category
}

# Prediction
if st.button("💰 Predict Price"):
    try:
        # Send POST request to FastAPI
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()  # Raise exception for bad status codes
        result = response.json()

        # Display results
        low_price = result["low_price_cr"]
        high_price = result["high_price_cr"]
        formatted_range = result["formatted_range"]
        st.success(f"Predicted Price Range: {formatted_range}")
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {str(e)}")