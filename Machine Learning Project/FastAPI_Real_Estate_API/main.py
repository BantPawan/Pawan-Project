from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
import pandas as pd
import numpy as np
from typing import List

app = FastAPI(title="Real Estate Analytics API")

# Load data and model at startup
try:
    with open('Dataset/df.pkl', 'rb') as file:
        df = pickle.load(file)
    pipeline = joblib.load('Dataset/pipeline_compressed.pkl')  # Load compressed model
    location_df = pickle.load(open('Dataset/location_distance.pkl', 'rb'))
    cosine_sim1 = pickle.load(open('Dataset/cosine_sim1.pkl', 'rb'))
    cosine_sim2 = pickle.load(open('Dataset/cosine_sim2.pkl', 'rb'))
    cosine_sim3 = pickle.load(open('Dataset/cosine_sim3.pkl', 'rb'))
except FileNotFoundError as e:
    raise Exception(f"Model or data file not found: {e}")

# Pydantic models for request validation
class PropertyInput(BaseModel):
    property_type: str
    sector: str
    bedrooms: float
    bathroom: float
    balcony: str
    property_age: str
    built_up_area: float
    servant_room: float
    store_room: float
    furnishing_type: str
    luxury_category: str
    floor_category: str

class RecommendationInput(BaseModel):
    property_name: str
    top_n: int = 5

class LocationSearchInput(BaseModel):
    location: str
    radius: float

# Price Predictor Endpoint
@app.post("/predict_price")
async def predict_price(input: PropertyInput):
    if input.sector not in df['sector'].unique():
        raise HTTPException(status_code=400, detail="Invalid sector")
    if input.property_type not in ['flat', 'house']:
        raise HTTPException(status_code=400, detail="Invalid property type")

    # Form DataFrame
    data = [[
        input.property_type, input.sector, input.bedrooms, input.bathroom, input.balcony,
        input.property_age, input.built_up_area, input.servant_room, input.store_room,
        input.furnishing_type, input.luxury_category, input.floor_category
    ]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']
    
    one_df = pd.DataFrame(data, columns=columns)

    # Predict
    try:
        base_price = np.expm1(pipeline.predict(one_df))[0]
        low = base_price - 0.22
        high = base_price + 0.22
        return {
            "low_price_cr": round(low, 2),
            "high_price_cr": round(high, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Apartment Recommendation Endpoint
@app.post("/recommend_apartments")
async def recommend_apartments(input: RecommendationInput):
    def recommend_properties_with_scores(property_name: str, top_n: int = 5):
        cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
        if property_name not in location_df.index:
            raise HTTPException(status_code=404, detail="Property not found")
        
        sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
        top_properties = location_df.index[top_indices].tolist()
        
        return [
            {"PropertyName": prop, "SimilarityScore": score}
            for prop, score in zip(top_properties, top_scores)
        ]
    
    try:
        recommendations = recommend_properties_with_scores(input.property_name, input.top_n)
        return {"recommendations": recommendations}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

# Location Search Endpoint
@app.post("/search_locations")
async def search_locations(input: LocationSearchInput):
    if input.location not in location_df.columns:
        raise HTTPException(status_code=400, detail="Invalid location")
    if input.radius <= 0:
        raise HTTPException(status_code=400, detail="Radius must be positive")
    
    try:
        result_ser = location_df[location_df[input.location] < input.radius * 1000][input.location].sort_values()
        results = [{"property": key, "distance_km": round(value / 1000, 2)} for key, value in result_ser.items()]
        return {"locations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Real Estate Analytics API"}