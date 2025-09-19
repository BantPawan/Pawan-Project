# Main_Page.py
import streamlit as st
import os

st.set_page_config(
    page_title="Real Estate Analytics App",
    page_icon="🏠",
    layout="wide"
)

st.sidebar.title("📑 Navigation")

BASE_DIR = os.path.dirname(__file__)

pages = {
    "🏡 Home": "01_Home.py",
    "📈 Price Predictor": "02_Price_Predictor.py",
    "📊 Analysis": "03_Analysis_App.py",
    "🏘 Recommend Apartments": "04_Recommend_Appartments.py"
}

choice = st.sidebar.radio("Go to", list(pages.keys()))

# Update query params
st.query_params["page"] = pages[choice]

page_path = os.path.join(BASE_DIR, pages[choice])

if os.path.exists(page_path):
    with open(page_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
else:
    st.error(f"❌ Page file not found: {page_path}")
