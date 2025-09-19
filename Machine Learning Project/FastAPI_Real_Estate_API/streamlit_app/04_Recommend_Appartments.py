# 04_Recommend_Appartments.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.title("Recommend Apartments")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

LOC_PATH = os.path.join(DATASET_DIR, "location_distance.pkl")
SIM1_PATH = os.path.join(DATASET_DIR, "cosine_sim1.pkl")
SIM2_PATH = os.path.join(DATASET_DIR, "cosine_sim2.pkl")
SIM3_PATH = os.path.join(DATASET_DIR, "cosine_sim3.pkl")

location_df = pickle.load(open(LOC_PATH, "rb"))
cosine_sim1 = pickle.load(open(SIM1_PATH, "rb"))
cosine_sim2 = pickle.load(open(SIM2_PATH, "rb"))
cosine_sim3 = pickle.load(open(SIM3_PATH, "rb"))


def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1.0 * cosine_sim3
    if property_name not in location_df.index:
        st.error("Property not found in dataset")
        return pd.DataFrame(columns=["PropertyName", "SimilarityScore"])

    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sorted_scores[1 : top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1 : top_n + 1]]
    top_properties = location_df.index[top_indices].tolist()

    return pd.DataFrame({"PropertyName": top_properties, "SimilarityScore": top_scores})

# --- Location search ---
st.header("Select Location and Radius")
selected_location = st.selectbox("Location", sorted(location_df.columns.to_list()))
radius = st.number_input("Radius in Kms", min_value=0.0, step=0.1)

if st.button("Search"):
    if radius > 0:
        result_ser = location_df[location_df[selected_location] < radius * 1000][selected_location].sort_values()
        for key, value in result_ser.items():
            st.text(f"{key} {round(value / 1000)} kms")
    else:
        st.warning("Please enter a valid radius.")

# --- Recommendations ---
st.header("Recommend Apartments")
selected_apartment = st.selectbox("Select an Apartment", sorted(location_df.index.to_list()))

if st.button("Recommend"):
    recommendation_df = recommend_properties_with_scores(selected_apartment)
    if not recommendation_df.empty:
        st.dataframe(recommendation_df)
    else:
        st.warning("No recommendations available for the selected apartment.")
