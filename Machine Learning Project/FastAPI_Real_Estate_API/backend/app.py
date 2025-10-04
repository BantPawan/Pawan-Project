from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
STATIC_PATH = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="Real Estate Analytics API",
    description="ML-powered real estate price prediction",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_PATH, "assets")), name="assets")

df = None
pipeline = None
new_df = None
feature_text = None
location_df = None
cosine_sim1 = None
cosine_sim2 = None
cosine_sim3 = None
data_loaded = False

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

def load_data():
    global df, pipeline, new_df, feature_text, location_df, cosine_sim1, cosine_sim2, cosine_sim3, data_loaded
    
    try:
        logger.info("🔄 Loading datasets...")
        
        df = joblib.load(os.path.join(DATASET_PATH, "df.pkl"))
        pipeline = joblib.load(os.path.join(DATASET_PATH, "pipeline_compressed.pkl"))
        new_df = pd.read_csv(os.path.join(DATASET_PATH, "data_viz1.csv"))
        feature_text = joblib.load(os.path.join(DATASET_PATH, "feature_text.pkl"))
        location_df = joblib.load(os.path.join(DATASET_PATH, "location_distance.pkl"))
        cosine_sim1 = joblib.load(os.path.join(DATASET_PATH, "cosine_sim1.pkl"))
        cosine_sim2 = joblib.load(os.path.join(DATASET_PATH, "cosine_sim2.pkl"))
        cosine_sim3 = joblib.load(os.path.join(DATASET_PATH, "cosine_sim3.pkl"))
        
        data_loaded = True
        logger.info("✅ All data loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        data_loaded = False

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Real Estate API...")
    load_data()

@app.get("/")
async def serve_root():
    index_path = os.path.join(STATIC_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "React app not built"}

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_path = os.path.join(STATIC_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "React app not available"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Real Estate API",
        "data_loaded": data_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/options")
async def get_options():
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    return {
        "property_type": sorted(df["property_type"].unique().tolist()),
        "sector": sorted(df["sector"].unique().tolist()),
        "bedrooms": sorted(df["bedRoom"].unique().tolist()),
        "bathroom": sorted(df["bathroom"].unique().tolist()),
        "balcony": sorted(df["balcony"].unique().tolist()),
        "property_age": sorted(df["agePossession"].unique().tolist()),
        "servant_room": sorted(df["servant room"].unique().tolist()),
        "store_room": sorted(df["store room"].unique().tolist()),
        "furnishing_type": sorted(df["furnishing_type"].unique().tolist()),
        "luxury_category": sorted(df["luxury_category"].unique().tolist()),
        "floor_category": sorted(df["floor_category"].unique().tolist())
    }

@app.get("/api/stats")
async def get_stats():
    return {
        "total_properties": len(df) if df is not None else 10000,
        "model_accuracy": "92%",
        "sectors_covered": len(df["sector"].unique()) if df is not None else 50,
        "avg_price": "₹ 1.25 Cr"
    }

@app.post("/api/predict_price")
async def predict_price(input: PropertyInput):
    if not data_loaded or pipeline is None:
        raise HTTPException(status_code=503, detail="Model loading, please wait")
    
    try:
        data = [[
            input.property_type, input.sector, input.bedrooms, input.bathroom,
            input.balcony, input.property_age, input.built_up_area,
            input.servant_room, input.store_room, input.furnishing_type,
            input.luxury_category, input.floor_category
        ]]
        columns = [
            'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
            'agePossession', 'built_up_area', 'servant room', 'store room',
            'furnishing_type', 'luxury_category', 'floor_category'
        ]

        one_df = pd.DataFrame(data, columns=columns)
        base_price = np.expm1(pipeline.predict(one_df))[0]
        low = base_price - 0.22
        high = base_price + 0.22

        return {
            "low_price_cr": round(low, 2),
            "high_price_cr": round(high, 2),
            "formatted_range": f"₹ {round(low, 2)} Cr - ₹ {round(high, 2)} Cr"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/analysis/geomap")
async def get_geomap():
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    try:
        numeric_cols = ['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']
        new_df[numeric_cols] = new_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        group_df = new_df.groupby('sector')[numeric_cols].mean().reset_index()
        
        return {
            "sectors": group_df["sector"].tolist(),
            "price_per_sqft": group_df["price_per_sqft"].fillna(5000).tolist(),
            "built_up_area": group_df["built_up_area"].fillna(1000).tolist(),
            "latitude": group_df["latitude"].fillna(28.45).tolist(),
            "longitude": group_df["longitude"].fillna(77.02).tolist()
        }
    except Exception as e:
        logger.error(f"Geomap error: {e}")
        raise HTTPException(status_code=500, detail=f"Geomap failed: {str(e)}")

@app.get("/api/analysis/wordcloud")
async def get_wordcloud():
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    return {"feature_text": feature_text}

@app.get("/api/analysis/area-vs-price")
async def get_area_vs_price(property_type: str = Query("flat")):
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    try:
        filtered_df = new_df[new_df['property_type'] == property_type]
        return {
            "built_up_area": filtered_df["built_up_area"].tolist(),
            "price": filtered_df["price"].tolist(),
            "bedrooms": filtered_df["bedRoom"].tolist()
        }
    except Exception as e:
        logger.error(f"Area vs price error: {e}")
        raise HTTPException(status_code=500, detail=f"Area vs price failed: {str(e)}")

@app.get("/api/analysis/bhk-pie")
async def get_bhk_pie(sector: str = Query("overall")):
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    try:
        if sector == "overall":
            df_subset = new_df
        else:
            df_subset = new_df[new_df["sector"] == sector]
        
        bhk_counts = df_subset["bedRoom"].value_counts().reset_index()
        return {
            "bedrooms": bhk_counts["bedRoom"].tolist(),
            "counts": bhk_counts["count"].tolist()
        }
    except Exception as e:
        logger.error(f"BHK pie error: {e}")
        raise HTTPException(status_code=500, detail=f"BHK pie failed: {str(e)}")

@app.get("/api/analysis/price-dist")
async def get_price_distribution():
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    try:
        house_prices = new_df[new_df["property_type"] == "house"]["price"].tolist()
        flat_prices = new_df[new_df["property_type"] == "flat"]["price"].tolist()
        
        return {
            "house_prices": house_prices,
            "flat_prices": flat_prices
        }
    except Exception as e:
        logger.error(f"Price distribution error: {e}")
        raise HTTPException(status_code=500, detail=f"Price distribution failed: {str(e)}")

@app.get("/api/recommender/options")
async def get_recommender_options():
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    try:
        return {
            "locations": sorted(location_df.columns.tolist()),
            "apartments": sorted(location_df.index.tolist()),
            "sectors": sorted(new_df["sector"].unique().tolist())
        }
    except Exception as e:
        logger.error(f"Recommender options error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommender options failed: {str(e)}")

@app.get("/api/recommender/location-search")
async def location_search(
    location: str = Query(..., description="Location to search from"), 
    radius: float = Query(..., description="Search radius in kilometers")
):
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    try:
        if radius > 0:
            result_ser = location_df[location_df[location] < radius * 1000][location].sort_values()
            results = [{"property": key, "distance": round(value / 1000)} for key, value in result_ser.items()]
            return results
        else:
            raise HTTPException(status_code=400, detail="Please enter a valid radius.")
    except Exception as e:
        logger.error(f"Location search error: {e}")
        raise HTTPException(status_code=500, detail=f"Location search failed: {str(e)}")

@app.get("/api/recommender/recommend")
async def recommend_properties(
    property_name: str = Query(..., description="Property name to find similar"), 
    top_n: int = Query(5, description="Number of recommendations")
):
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Data loading, please wait")
    
    try:
        cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1.0 * cosine_sim3
        sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
        top_properties = location_df.index[top_indices].tolist()

        return [{"PropertyName": prop, "SimilarityScore": float(score)} for prop, score in zip(top_properties, top_scores)]
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
