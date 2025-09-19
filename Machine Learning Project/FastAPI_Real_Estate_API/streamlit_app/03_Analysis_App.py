# 03_Analysis_App.py
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

st.title("Analysis")

# Go one level up from current script folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# Paths
DATASET_PATH = os.path.join(DATASET_DIR, "data_viz1.csv")
FEATURE_PATH = os.path.join(DATASET_DIR, "feature_text.pkl")

# Load data
new_df = pd.read_csv(DATASET_PATH)
feature_text = pickle.load(open(FEATURE_PATH, "rb"))

# Convert numeric columns
num_cols = ["price", "price_per_sqft", "built_up_area", "latitude", "longitude"]
new_df[num_cols] = new_df[num_cols].apply(pd.to_numeric, errors="coerce")

# Group by sector
group_df = new_df.groupby("sector")[num_cols].mean()

# --- Sector Price per Sqft Geomap ---
st.header("Sector Price per Sqft Geomap")
fig = px.scatter_map(
    group_df,
    lat="latitude",
    lon="longitude",
    color="price_per_sqft",
    size="built_up_area",
    color_continuous_scale=px.colors.cyclical.IceFire,
    zoom=10,
    map_style="open-street-map",  # updated for scatter_map
    width=1200,
    height=700,
    hover_name=group_df.index,
)
st.plotly_chart(fig, use_container_width=True)

# --- Features Wordcloud ---
st.header("Features Wordcloud")
wordcloud = WordCloud(
    width=800,
    height=800,
    background_color="black",
    stopwords=set(["s"]),
    min_font_size=10,
).generate(feature_text)

fig_wc, ax_wc = plt.subplots(figsize=(8, 8))
ax_wc.imshow(wordcloud, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)

# --- Area Vs Price ---
st.header("Area Vs Price")
property_type = st.selectbox("Select Property Type", ["flat", "house"])
filtered_df = new_df[new_df["property_type"] == property_type]
fig1 = px.scatter(
    filtered_df,
    x="built_up_area",
    y="price",
    color="bedRoom",
    title=f"Area Vs Price for {property_type.capitalize()}",
)
st.plotly_chart(fig1, use_container_width=True)

# --- BHK Pie Chart ---
st.header("BHK Pie Chart")
sector_options = ["overall"] + new_df["sector"].unique().tolist()
selected_sector = st.selectbox("Select Sector", sector_options)

if selected_sector == "overall":
    fig2 = px.pie(new_df, names="bedRoom", title="Distribution of BHK in Overall Data")
else:
    fig2 = px.pie(
        new_df[new_df["sector"] == selected_sector],
        names="bedRoom",
        title=f"Distribution of BHK in {selected_sector}",
    )
st.plotly_chart(fig2, use_container_width=True)

# --- Side by Side Distplot ---
st.header("Side by Side Distplot for Property Type")
fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
sns.histplot(new_df[new_df["property_type"] == "house"]["price"], label="house", kde=True, ax=ax_dist)
sns.histplot(new_df[new_df["property_type"] == "flat"]["price"], label="flat", kde=True, ax=ax_dist)
ax_dist.legend()
ax_dist.set_title("Price Distribution by Property Type")
st.pyplot(fig_dist)
