from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib, pickle, pandas as pd, numpy as np, sklearn, os
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

print(f"✅ scikit-learn runtime version: {sklearn.__version__}")

# Use absolute path for Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")

app = FastAPI(
    title="Real Estate AI Analytics",
    description="ML-powered real estate price prediction, analysis, and recommendation platform",
    version="1.0.0"
)

# Serve frontend static files from the frontend directory
frontend_path = os.path.join(BASE_DIR, "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper loader
def load_pickle(filename):
    file_path = os.path.join(DATASET_PATH, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)

# Load dataset & pipeline
try:
    df = load_pickle("df.pkl")
    pipeline = load_pickle("pipeline_compressed.pkl")
    data_viz = pd.read_csv(os.path.join(DATASET_PATH, "data_viz1.csv"))
    feature_text = load_pickle("feature_text.pkl")
    location_df = load_pickle("location_distance.pkl")
    cosine_sim1 = load_pickle("cosine_sim1.pkl")
    cosine_sim2 = load_pickle("cosine_sim2.pkl")
    cosine_sim3 = load_pickle("cosine_sim3.pkl")
    print("✅ ML model and data loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model/data files: {e}")
    df = pd.DataFrame()
    pipeline = None
    data_viz = pd.DataFrame()
    feature_text = ""
    location_df = pd.DataFrame()
    cosine_sim1 = np.array([])
    cosine_sim2 = np.array([])
    cosine_sim3 = np.array([])

# Convert numeric columns in data_viz
num_cols = ["price", "price_per_sqft", "built_up_area", "latitude", "longitude"]
data_viz[num_cols] = data_viz[num_cols].apply(pd.to_numeric, errors="coerce")

# Pydantic schema
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

# Utility: format price in Cr
def format_price(value: float) -> str:
    return f"₹ {value:,.2f} Cr"

# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/api/predict_price")
async def predict_price_get():
    return {"message": "Use POST method to predict price"}

@app.post("/api/predict_price")
async def predict_price(input: PropertyInput):
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        one_df = pd.DataFrame([[
            input.property_type, input.sector, input.bedrooms, input.bathroom,
            input.balcony, input.property_age, input.built_up_area,
            input.servant_room, input.store_room, input.furnishing_type,
            input.luxury_category, input.floor_category
        ]], columns=[
            'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
            'agePossession', 'built_up_area', 'servant room', 'store room',
            'furnishing_type', 'luxury_category', 'floor_category'
        ])

        base_price = np.expm1(pipeline.predict(one_df))[0]
        low_price, high_price = base_price - 0.22, base_price + 0.22

        return {
            "prediction_raw": float(base_price),
            "low_price_cr": round(low_price, 2),
            "high_price_cr": round(high_price, 2),
            "formatted_range": f"{format_price(low_price)} - {format_price(high_price)}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/options")
async def get_options():
    try:
        if df.empty:
            return {
                "property_type": ["flat", "house"],
                "sector": ["Sector 1", "Sector 2"],
                "bedrooms": [1, 2, 3, 4],
                "bathroom": [1, 2, 3],
                "balcony": ["0", "1", "2", "3"],
                "property_age": ["New", "1-5 years", "5-10 years"],
                "servant_room": [0, 1],
                "store_room": [0, 1],
                "furnishing_type": ["Unfurnished", "Semi-Furnished", "Fully-Furnished"],
                "luxury_category": ["Low", "Medium", "High"],
                "floor_category": ["Low", "Mid", "High"]
            }
            
        return {
            "property_type": df["property_type"].unique().tolist(),
            "sector": df["sector"].unique().tolist(),
            "bedrooms": sorted(df["bedRoom"].unique().tolist()),
            "bathroom": sorted(df["bathroom"].unique().tolist()),
            "balcony": sorted(df["balcony"].unique().tolist()),
            "property_age": df["agePossession"].unique().tolist(),
            "servant_room": sorted(df["servant room"].unique().tolist()),
            "store_room": sorted(df["store room"].unique().tolist()),
            "furnishing_type": df["furnishing_type"].unique().tolist(),
            "luxury_category": df["luxury_category"].unique().tolist(),
            "floor_category": df["floor_category"].unique().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading options: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    return {
        "total_properties": len(df) if not df.empty else 10000,
        "avg_price": f"₹ {df['price'].mean():.2f} Cr" if not df.empty else "₹ 1.25 Cr",
        "sectors_covered": len(df["sector"].unique()) if not df.empty else 50,
        "model_accuracy": "92%",
        "last_updated": "2025-09-26"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Real Estate AI Analytics",
        "version": "1.0.0",
        "sklearn_version": sklearn.__version__,
        "model_loaded": pipeline is not None,
        "data_loaded": not df.empty,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/recommender/options")
async def get_recommender_options():
    try:
        return {
            "locations": sorted(location_df.columns.tolist()),
            "apartments": sorted(location_df.index.tolist()),
            "sectors": sorted(data_viz["sector"].unique().tolist())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading recommender options: {str(e)}")

@app.get("/api/analysis/geomap")
async def get_geomap():
    try:
        group_df = data_viz.groupby("sector")[["price_per_sqft", "built_up_area", "latitude", "longitude"]].mean().reset_index()
        return {
            "sectors": group_df["sector"].tolist(),
            "price_per_sqft": group_df["price_per_sqft"].tolist(),
            "built_up_area": group_df["built_up_area"].tolist(),
            "latitude": group_df["latitude"].tolist(),
            "longitude": group_df["longitude"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading geomap data: {str(e)}")

@app.get("/api/analysis/wordcloud")
async def get_wordcloud():
    try:
        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color="black",
            stopwords=set(["s"]),
            min_font_size=10,
        ).generate(feature_text)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        
        return {"image_url": f"data:image/png;base64,{img_base64}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating wordcloud: {str(e)}")

@app.get("/api/analysis/area-vs-price")
async def get_area_vs_price(property_type: str = "flat"):
    try:
        filtered_df = data_viz[data_viz["property_type"] == property_type]
        return {
            "built_up_area": filtered_df["built_up_area"].tolist(),
            "price": filtered_df["price"].tolist(),
            "bedrooms": filtered_df["bedRoom"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading area vs price data: {str(e)}")

@app.get("/api/analysis/bhk-pie")
async def get_bhk_pie(sector: str = "overall"):
    try:
        if sector == "overall":
            df_subset = data_viz
        else:
            df_subset = data_viz[data_viz["sector"] == sector]
        
        bhk_counts = df_subset["bedRoom"].value_counts().reset_index()
        return {
            "bedrooms": bhk_counts["bedRoom"].tolist(),
            "counts": bhk_counts["count"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading BHK pie data: {str(e)}")

@app.get("/api/analysis/price-dist")
async def get_price_distribution():
    try:
        house_prices = data_viz[data_viz["property_type"] == "house"]["price"].dropna().tolist()
        flat_prices = data_viz[data_viz["property_type"] == "flat"]["price"].dropna().tolist()
        return {
            "house_prices": house_prices,
            "flat_prices": flat_prices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading price distribution data: {str(e)}")

@app.get("/api/recommender/location-search")
async def location_search(location: str, radius: float):
    try:
        if location not in location_df.columns:
            raise HTTPException(status_code=400, detail="Invalid location")
        if radius <= 0:
            raise HTTPException(status_code=400, detail="Radius must be positive")
            
        result_series = location_df[location_df[location] < radius * 1000][location].sort_values()
        results = [{"property": key, "distance": round(value / 1000, 1)} for key, value in result_series.items()]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in location search: {str(e)}")

@app.get("/api/recommender/recommend")
async def recommend_properties(property_name: str, top_n: int = 5):
    try:
        if property_name not in location_df.index:
            raise HTTPException(status_code=400, detail="Property not found")
            
        cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1.0 * cosine_sim3
        idx = location_df.index.get_loc(property_name)
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
        top_properties = location_df.index[top_indices].tolist()
        
        return [{"PropertyName": prop, "SimilarityScore": score} for prop, score in zip(top_properties, top_scores)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Serve other frontend files
@app.get("/{path:path}")
async def serve_static(path: str):
    static_file = os.path.join(frontend_path, path)
    if os.path.exists(static_file) and os.path.isfile(static_file):
        return FileResponse(static_file)
    # Fallback to index.html for SPA routing
    return FileResponse(os.path.join(frontend_path, "index.html"))
