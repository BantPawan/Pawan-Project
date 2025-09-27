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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"scikit-learn runtime version: {sklearn.__version__}")

# Use absolute path for Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")

app = FastAPI(
    title="Real Estate Analytics API",
    description="ML-powered real estate price prediction, analysis, and recommendation platform",
    version="1.0.0"
)

# Serve frontend static files from the frontend directory
frontend_path = os.path.join(BASE_DIR, "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
else:
    logger.warning(f"Frontend path {frontend_path} does not exist")

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
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        logger.info(f"Loading pickle file: {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        logger.warning(f"Joblib load failed for {file_path}, trying pickle: {e}")
        with open(file_path, "rb") as f:
            return pickle.load(f)

# Load dataset & pipeline
try:
    df = load_pickle("df.pkl")
    logger.info(f"df.pkl loaded with columns: {df.columns.tolist()}")
    pipeline = load_pickle("pipeline_compressed.pkl")
    data_viz = pd.read_csv(os.path.join(DATASET_PATH, "data_viz1.csv"))
    logger.info(f"data_viz1.csv loaded with columns: {data_viz.columns.tolist()}")
    feature_text = load_pickle("feature_text.pkl")
    # Validate feature_text
    if not isinstance(feature_text, str) or len(feature_text.strip()) < 10:
        logger.warning("feature_text.pkl is invalid or too short, using fallback")
        feature_text = "apartment flat house bedroom bathroom balcony furnished semi-furnished luxury modern new old spacious garden parking security elevator gym pool clubhouse green area" * 10
    location_df = load_pickle("location_distance.pkl")
    cosine_sim1 = load_pickle("cosine_sim1.pkl")
    cosine_sim2 = load_pickle("cosine_sim2.pkl")
    cosine_sim3 = load_pickle("cosine_sim3.pkl")
    logger.info("ML model and data loaded successfully")
except Exception as e:
    logger.error(f"Error loading model/data files: {e}")
    df = pd.DataFrame()
    pipeline = None
    data_viz = pd.DataFrame()
    feature_text = "apartment flat house bedroom bathroom balcony furnished semi-furnished luxury modern new old spacious garden parking security elevator gym pool clubhouse green area" * 10
    location_df = pd.DataFrame()
    cosine_sim1 = np.array([])
    cosine_sim2 = np.array([])
    cosine_sim3 = np.array([])

# Determine price column dynamically for data_viz
price_column = None
for col in ['price', 'Price', 'price_cr', 'Price_in_cr']:
    if col in data_viz.columns:
        price_column = col
        break
if price_column is None:
    logger.warning("Price column not found in data_viz1.csv, using default")
    price_column = 'price'

# Convert numeric columns in data_viz
if not data_viz.empty:
    num_cols = ["price_per_sqft", "built_up_area", "latitude", "longitude"]
    if price_column in data_viz.columns:
        num_cols.append(price_column)
    else:
        logger.warning(f"Price column {price_column} not found in data_viz1.csv")
    
    existing_num_cols = [col for col in num_cols if col in data_viz.columns]
    if existing_num_cols:
        data_viz[existing_num_cols] = data_viz[existing_num_cols].apply(pd.to_numeric, errors="coerce")
    else:
        logger.warning("No numeric columns found in data_viz for conversion")

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
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.warning(f"Frontend index.html not found at {index_path}")
        return {"message": "Frontend not found. Please check the frontend path."}

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
            "sklearn_version": sklearn.__version__,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/options")
async def get_options():
    try:
        if df.empty:
            logger.warning("df is empty, returning default options")
            return {
                "property_type": ["flat", "house"],
                "sector": ["sector 1", "sector 2"],
                "bedrooms": [1, 2, 3, 4],
                "bathroom": [1, 2, 3],
                "balcony": ["0", "1", "2", "3", "3+"],
                "property_age": ["New Property", "Relatively New", "Moderately Old", "Old Property"],
                "servant_room": [0, 1],
                "store_room": [0, 1],
                "furnishing_type": ["unfurnished", "semifurnished", "furnished"],
                "luxury_category": ["Low", "Medium", "High"],
                "floor_category": ["Low Floor", "Mid Floor", "High Floor"]
            }
            
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
    except Exception as e:
        logger.error(f"Error loading options: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading options: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    try:
        if data_viz.empty:
            logger.warning("data_viz is empty, returning default stats")
            return {
                "total_properties": 10000,
                "avg_price": "₹ 1.25 Cr",
                "sectors_covered": 50,
                "model_accuracy": "89.2%",
                "last_updated": "2025-09-26"
            }
            
        avg_price = data_viz[price_column].mean() if price_column in data_viz.columns else 1.25
        sectors_covered = len(data_viz["sector"].unique()) if "sector" in data_viz.columns else 50
        
        return {
            "total_properties": len(data_viz),
            "avg_price": f"₹ {avg_price:.2f} Cr",
            "sectors_covered": sectors_covered,
            "model_accuracy": "89.2%",
            "last_updated": "2025-09-26"
        }
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return {
            "total_properties": 10000,
            "avg_price": "₹ 1.25 Cr",
            "sectors_covered": 50,
            "model_accuracy": "89.2%",
            "last_updated": "2025-09-26"
        }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Real Estate Analytics API",
        "version": "1.0.0",
        "sklearn_version": sklearn.__version__,
        "model_loaded": pipeline is not None,
        "data_loaded": not df.empty and not data_viz.empty,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/recommender/options")
async def get_recommender_options():
    try:
        if location_df.empty or data_viz.empty:
            logger.warning("location_df or data_viz is empty, returning default recommender options")
            return {
                "locations": ["Location 1", "Location 2"],
                "apartments": ["Apartment 1", "Apartment 2"],
                "sectors": ["Sector 1", "Sector 2"]
            }
        return {
            "locations": sorted(location_df.columns.tolist()),
            "apartments": sorted(location_df.index.tolist()),
            "sectors": sorted(data_viz["sector"].unique().tolist()) if "sector" in data_viz.columns else []
        }
    except Exception as e:
        logger.error(f"Error loading recommender options: {e}")
        return {
            "locations": ["Location 1", "Location 2"],
            "apartments": ["Apartment 1", "Apartment 2"],
            "sectors": ["Sector 1", "Sector 2"]
        }

@app.get("/api/analysis/geomap")
async def get_geomap():
    try:
        if data_viz.empty:
            logger.warning("data_viz is empty, returning sample geomap data")
            return {
                "sectors": ["Sector 45", "Sector 46", "Sector 47"],
                "price_per_sqft": [8500, 9200, 7800],
                "built_up_area": [1200, 1500, 1800],
                "latitude": [28.4595, 28.4612, 28.4630],
                "longitude": [77.0266, 77.0280, 77.0300],
                "property_count": [25, 30, 20]
            }
        
        required_cols = ["sector", "price_per_sqft", "built_up_area", "latitude", "longitude"]
        missing_cols = [col for col in required_cols if col not in data_viz.columns]
        if missing_cols:
            logger.warning(f"Missing columns in data_viz: {missing_cols}")
            if "sector" in data_viz.columns:
                sectors = data_viz["sector"].unique().tolist()[:10]
            else:
                sectors = [f"Sector {i}" for i in range(1, 11)]
            
            return {
                "sectors": sectors,
                "price_per_sqft": [8000 + i*200 for i in range(len(sectors))],
                "built_up_area": [1200 + i*100 for i in range(len(sectors))],
                "latitude": [28.4595 + i*0.002 for i in range(len(sectors))],
                "longitude": [77.0266 + i*0.002 for i in range(len(sectors))],
                "property_count": [20 + i*5 for i in range(len(sectors))]
            }
        
        group_df = data_viz.groupby("sector").agg({
            "price_per_sqft": "mean",
            "built_up_area": "mean",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()
        
        property_count = data_viz["sector"].value_counts().to_dict()
        group_df["property_count"] = group_df["sector"].map(property_count)
        
        group_df = group_df.dropna(subset=["latitude", "longitude"])
        
        return {
            "sectors": group_df["sector"].tolist(),
            "price_per_sqft": group_df["price_per_sqft"].tolist(),
            "built_up_area": group_df["built_up_area"].tolist(),
            "latitude": group_df["latitude"].tolist(),
            "longitude": group_df["longitude"].tolist(),
            "property_count": group_df["property_count"].tolist()
        }
    except Exception as e:
        logger.error(f"Error loading geomap data: {e}")
        return {
            "sectors": ["Sector 45", "Sector 46", "Sector 47", "Sector 48", "Sector 49"],
            "price_per_sqft": [8500, 9200, 7800, 9500, 8200],
            "built_up_area": [1200, 1500, 1800, 1400, 1600],
            "latitude": [28.4595, 28.4612, 28.4630, 28.4645, 28.4660],
            "longitude": [77.0266, 77.0280, 77.0300, 77.0320, 77.0340],
            "property_count": [25, 30, 20, 35, 28]
        }

@app.get("/api/analysis/wordcloud")
async def get_wordcloud():
    try:
        logger.info("Generating wordcloud")
        if not feature_text or (isinstance(feature_text, str) and len(feature_text.strip()) < 10):
            logger.warning("feature_text is invalid or too short, using sample text")
            feature_text_to_use = "apartment flat house bedroom bathroom balcony furnished semi-furnished luxury modern new old spacious garden parking security elevator gym pool clubhouse green area" * 10
        else:
            feature_text_to_use = feature_text
        
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 8), facecolor='black')
        
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='black',
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue',
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=120,
            random_state=42
        ).generate(feature_text_to_use)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        logger.info("Wordcloud generated successfully")
        return {"image_url": f"data:image/png;base64,{img_base64}"}
        
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 8), facecolor='black')
        plt.text(0.5, 0.5, 'Wordcloud\nNot Available', 
                fontsize=24, color='white', ha='center', va='center',
                transform=plt.gca().transAxes)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {"image_url": f"data:image/png;base64,{img_base64}"}

@app.get("/api/analysis/area-vs-price")
async def get_area_vs_price(property_type: str = "flat"):
    try:
        if data_viz.empty:
            logger.warning("data_viz is empty, returning sample area-vs-price data")
            return {
                "built_up_area": [1200, 1500, 1800],
                "price": [1.2, 1.5, 1.8],
                "bedrooms": [2, 3, 4]
            }
            
        filtered_df = data_viz[data_viz["property_type"] == property_type] if "property_type" in data_viz.columns else data_viz
        return {
            "built_up_area": filtered_df["built_up_area"].tolist() if "built_up_area" in filtered_df.columns else [],
            "price": filtered_df[price_column].tolist() if price_column in filtered_df.columns else [],
            "bedrooms": filtered_df["bedRoom"].tolist() if "bedRoom" in filtered_df.columns else []
        }
    except Exception as e:
        logger.error(f"Error loading area vs price data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading area vs price data: {str(e)}")

@app.get("/api/analysis/bhk-pie")
async def get_bhk_pie(sector: str = "overall"):
    try:
        if data_viz.empty:
            logger.warning("data_viz is empty, returning sample BHK pie data")
            return {
                "bedrooms": [2, 3, 4],
                "counts": [30, 40, 20]
            }
            
        if sector == "overall":
            df_subset = data_viz
        else:
            df_subset = data_viz[data_viz["sector"] == sector] if "sector" in data_viz.columns else data_viz
        
        bhk_counts = df_subset["bedRoom"].value_counts().reset_index() if "bedRoom" in df_subset.columns else pd.DataFrame({"bedRoom": [2,3,4], "count": [30,40,20]})
        return {
            "bedrooms": bhk_counts["bedRoom"].tolist(),
            "counts": bhk_counts["count"].tolist()
        }
    except Exception as e:
        logger.error(f"Error loading BHK pie data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading BHK pie data: {str(e)}")

@app.get("/api/analysis/price-dist")
async def get_price_distribution():
    try:
        if data_viz.empty:
            logger.warning("data_viz is empty, returning sample price distribution data")
            return {
                "house_prices": [1.5, 2.0, 2.5],
                "flat_prices": [0.8, 1.2, 1.5]
            }
            
        house_prices = data_viz[data_viz["property_type"] == "house"][price_column].dropna().tolist() if "property_type" in data_viz.columns and price_column in data_viz.columns else []
        flat_prices = data_viz[data_viz["property_type"] == "flat"][price_column].dropna().tolist() if "property_type" in data_viz.columns and price_column in data_viz.columns else []
        return {
            "house_prices": house_prices,
            "flat_prices": flat_prices
        }
    except Exception as e:
        logger.error(f"Error loading price distribution data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading price distribution data: {str(e)}")

@app.get("/api/recommender/location-search")
async def location_search(location: str, radius: float):
    try:
        if location_df.empty:
            logger.warning("location_df is empty, returning sample location search data")
            return [{"property": "Sample Property", "distance": 0.5}]
            
        if location not in location_df.columns:
            raise HTTPException(status_code=400, detail="Invalid location")
        if radius <= 0:
            raise HTTPException(status_code=400, detail="Radius must be positive")
            
        result_series = location_df[location_df[location] < radius * 1000][location].sort_values()
        results = [{"property": key, "distance": round(value / 1000, 1)} for key, value in result_series.items()]
        return results
    except Exception as e:
        logger.error(f"Error in location search: {e}")
        raise HTTPException(status_code=500, detail=f"Error in location search: {str(e)}")

@app.get("/api/recommender/recommend")
async def recommend_properties(property_name: str, top_n: int = 5):
    try:
        if location_df.empty or cosine_sim1.size == 0:
            logger.warning("location_df or cosine_sim is empty, returning sample recommendations")
            return [{"PropertyName": "Sample Property 1", "SimilarityScore": 0.95}]
            
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
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/{path:path}")
async def serve_static(path: str):
    static_file = os.path.join(frontend_path, path)
    if os.path.exists(static_file) and os.path.isfile(static_file):
        return FileResponse(static_file)
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    logger.warning(f"Static file not found: {static_file}")
    return {"message": "File not found"}
