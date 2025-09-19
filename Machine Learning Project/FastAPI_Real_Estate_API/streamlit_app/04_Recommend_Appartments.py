import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Recommendations", page_icon="🏘")
st.title("🏘 Apartment Recommendations")

# --- FastAPI backend URL ---
FASTAPI_URL = "http://127.0.0.1:8000"
# FASTAPI_URL = "https://your-fastapi-service.onrender.com"

# --- Inputs ---
selected_apartment = st.text_input("Enter Apartment Name")

if st.button("Recommend"):
    try:
        res = requests.get(f"{FASTAPI_URL}/recommend", params={"property_name": selected_apartment, "top_n": 5})
        if res.status_code == 200:
            data = pd.DataFrame(res.json())
            st.dataframe(data)
        else:
            st.error(f"Error {res.status_code}: {res.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
