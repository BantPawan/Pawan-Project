import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests

st.set_page_config(page_title="Analysis", page_icon="📊")
st.title("📊 Real Estate Analysis")

# --- FastAPI backend URL ---
FASTAPI_URL = "http://127.0.0.1:8000"
# FASTAPI_URL = "https://your-fastapi-service.onrender.com"

@st.cache_data
def load_analysis_data():
    try:
        res = requests.get(f"{FASTAPI_URL}/analysis_data")
        if res.status_code == 200:
            return pd.DataFrame(res.json())
        else:
            st.error(f"Error fetching data: {res.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return pd.DataFrame()

df = load_analysis_data()

if not df.empty:
    # Convert numeric
    num_cols = ["price", "price_per_sqft", "built_up_area", "latitude", "longitude"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # --- Geomap ---
    st.header("Sector Price per Sqft Geomap")
    group_df = df.groupby("sector")[num_cols].mean().reset_index()
    fig = px.scatter_mapbox(
        group_df,
        lat="latitude",
        lon="longitude",
        color="price_per_sqft",
        size="built_up_area",
        zoom=10,
        mapbox_style="open-street-map",
        hover_name="sector",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- BHK Pie ---
    st.header("BHK Distribution")
    sector_options = ["overall"] + df["sector"].unique().tolist()
    selected_sector = st.selectbox("Select Sector", sector_options)

    if selected_sector == "overall":
        fig2 = px.pie(df, names="bedRoom", title="Overall BHK Distribution")
    else:
        fig2 = px.pie(df[df["sector"] == selected_sector], names="bedRoom", title=f"BHK in {selected_sector}")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Side by side distplot ---
    st.header("Price Distribution by Property Type")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    sns.histplot(df[df["property_type"] == "house"]["price"], label="house", kde=True, ax=ax_dist)
    sns.histplot(df[df["property_type"] == "flat"]["price"], label="flat", kde=True, ax=ax_dist)
    ax_dist.legend()
    st.pyplot(fig_dist)
