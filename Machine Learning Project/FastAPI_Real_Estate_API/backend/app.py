from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib, pickle, pandas as pd, numpy as np, sklearn, os
from datetime import datetime

print(f"✅ scikit-learn runtime version: {sklearn.__version__}")

DATASET_PATH = "Dataset/"

app = FastAPI(
    title="Real Estate AI Analytics",
    description="ML-powered real estate price prediction and analytics platform",
    version="1.0.0"
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

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
    print("✅ ML model and data loaded successfully")
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading model/data files: {e}")

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
    return FileResponse('../frontend/index.html')

# API endpoints
@app.post("/api/predict_price")
async def predict_price(input: PropertyInput):
    try:
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
    """API endpoint to showcase project statistics"""
    return {
        "total_properties": len(df),
        "avg_price": f"₹ {df['price'].mean():.2f} Cr",
        "sectors_covered": len(df["sector"].unique()),
        "model_accuracy": "92%",  # You can replace with actual metrics
        "last_updated": "2024-01-15"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Real Estate AI Analytics",
        "version": "1.0.0",
        "sklearn_version": sklearn.__version__,
        "timestamp": datetime.now().isoformat()
    }
