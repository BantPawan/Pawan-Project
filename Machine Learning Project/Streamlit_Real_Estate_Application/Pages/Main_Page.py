import streamlit as st

Home = st.Page(
    page = "01_Home.py",
    title="Home",
    default=True
)

Price_Predictor = st.Page(
    page = "02_Price_Predictor.py",
    title="Price Predictor"
)

Analysis_App = st.Page(
    page = "03_Analysis_App.py",
    title="Analysis"
)

Recommend_Appartments=st.Page(
    page = "04_Recommend_Appartments.py",
    title="Recommend Appartments"
)


pg = st.navigation(pages = [Home,Price_Predictor,Analysis_App,Recommend_Appartments])

pg.run()