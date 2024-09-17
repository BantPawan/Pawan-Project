import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Price Predictor")

# Load data and model
with open('../df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('../pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
st.header('Enter your inputs')

# User inputs
property_type = st.selectbox('Property Type', ['flat', 'house'])

sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom', sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area', min_value=0.0, step=0.1))

servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))

store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))

luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))

floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):

    # Form a DataFrame
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)

    # Predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # Display results
    st.write(f"The price of the property is between {round(low, 2)} Cr and {round(high, 2)} Cr")


