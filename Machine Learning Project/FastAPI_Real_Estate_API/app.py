from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, pickle, pandas as pd, numpy as np, sklearn, os

print(f"✅ scikit-learn runtime version: {sklearn.__version__}")

DATASET_PATH = "Dataset/"

app = FastAPI(title="Real Estate Analytics API")

# Allow Streamlit/frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------
# Helper loader
# -----------------------------
def load_pickle(filename):
    file_path = os.path.join(DATASET_PATH, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)

# -----------------------------
# Load dataset & pipeline
# -----------------------------
try:
    df = load_pickle("df.pkl")
    pipeline = load_pickle("pipeline_compressed.pkl")
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading model/data files: {e}")

# -----------------------------
# Pydantic schema
# -----------------------------
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

# -----------------------------
# Utility: format price in Cr
# -----------------------------
def format_price(value: float) -> str:
    return f"₹ {value:,.2f} Cr"

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict_price")
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
            "sklearn_version": sklearn.__version__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "✅ Real Estate Analytics API is working!",
            "sklearn_version": sklearn.__version__}
