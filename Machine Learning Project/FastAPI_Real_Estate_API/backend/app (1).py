from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pickle
import pandas as pd
import numpy as np
import sklearn
import os
from datetime import datetime
import logging
import json

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

# WordCloud imports
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"✅ scikit-learn runtime version: {sklearn.__version__}")

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
frontend_path = os.path.join(BASE_DIR, "..", "frontend")

app = FastAPI(
    title="Real Estate Analytics API",
    description="ML-powered real estate price prediction, analysis, and recommendation platform",
    version="1.0.0"
)

# Serve frontend static files
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
else:
    logger.warning(f"Frontend path {frontend_path} does not exist")

# Serve dataset directory for visualizations
if os.path.exists(DATASET_PATH):
    app.mount("/viz", StaticFiles(directory=DATASET_PATH), name="viz")
else:
    logger.warning(f"Dataset path {DATASET_PATH} does not exist")

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
    logger.info(f"✅ df.pkl loaded with columns: {df.columns.tolist()}")
except Exception as e:
    logger.error(f"❌ Error loading df.pkl: {e}")
    df = pd.DataFrame()

try:
    pipeline = load_pickle("pipeline_compressed.pkl")
    logger.info("✅ Pipeline loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading pipeline: {e}")
    pipeline = None

# Load recommender data
try:
    location_df = load_pickle("location_distance.pkl")
    logger.info(f"✅ Location data loaded with shape: {location_df.shape}")
except Exception as e:
    logger.error(f"❌ Error loading location data: {e}")
    location_df = pd.DataFrame()

# Load cosine similarity matrices for recommender system
try:
    cosine_sim1 = load_pickle("cosine_sim1.pkl")
    cosine_sim2 = load_pickle("cosine_sim2.pkl")
    cosine_sim3 = load_pickle("cosine_sim3.pkl")
    logger.info("✅ Cosine similarity matrices loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Could not load cosine similarity matrices: {e}")
    cosine_sim1 = cosine_sim2 = cosine_sim3 = None

# Load analysis data
try:
    data_viz = pd.read_csv(os.path.join(DATASET_PATH, "data_viz1.csv"))
    logger.info(f"✅ Data viz loaded with shape: {data_viz.shape}")
    
    # Convert numeric columns
    num_cols = ["price", "price_per_sqft", "built_up_area", "latitude", "longitude"]
    existing_num_cols = [col for col in num_cols if col in data_viz.columns]
    if existing_num_cols:
        data_viz[existing_num_cols] = data_viz[existing_num_cols].apply(pd.to_numeric, errors="coerce")
        
except Exception as e:
    logger.error(f"❌ Error loading data_viz1.csv: {e}")
    data_viz = pd.DataFrame()

# Load feature text
try:
    feature_text = load_pickle("feature_text.pkl")
    logger.info("✅ Feature text loaded")
except Exception as e:
    logger.error(f"❌ Error loading feature_text: {e}")
    feature_text = ""

# Determine price column dynamically for data_viz
price_column = None
for col in ['price', 'Price', 'price_cr', 'Price_in_cr']:
    if col in data_viz.columns:
        price_column = col
        break
if price_column is None and not data_viz.empty:
    logger.warning("⚠️ Price column not found in data_viz1.csv")
    price_column = 'price'

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

# WordCloud Generation Function
def generate_wordcloud_from_text(text_data, width=800, height=400):
    """Generate wordcloud image from text data"""
    try:
        # Create wordcloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(text_data)
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buffer.seek(0)
        
        # Clear plot to free memory
        plt.close()
        
        return buffer
        
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
        return None

# ------------------ CHART UTILITIES ------------------

def get_area_vs_price_data(property_type: str = "all"):
    """Centralized data-prep for both raw data & chart endpoints"""
    try:
        if data_viz.empty:
            raise ValueError("No data available for area vs price analysis")
            
        df = data_viz.copy()
        if price_column != "price" and price_column in df.columns:
            df["price"] = df[price_column]

        # Filter property type
        if property_type != "all" and "property_type" in df.columns:
            df = df[df["property_type"] == property_type]

        if df.empty:
            raise ValueError(f"No data available for property type: {property_type}")

        # Sample to prevent frontend overload
        sample_size = min(100, len(df))
        df = df.sample(sample_size, random_state=42)

        return df

    except Exception as e:
        logger.error(f"Error in get_area_vs_price_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def area_vs_price_chart(df, property_type='all'):
    """Builds the Plotly scatter chart with proper layout"""
    fig = px.scatter(
        df,
        x="built_up_area",
        y="price",
        color="sector",
        hover_data=["society", "bedRoom", "bathroom"],
        title=f"Area vs Price Correlation ({property_type.title() if property_type != 'all' else 'All Properties'})",
        trendline="ols"
    )
    
    # Proper layout constraints to keep chart inside box
    fig.update_layout(
        template="plotly_white",
        width=600,
        height=500,
        margin=dict(t=60, b=60, l=60, r=60, pad=10),
        xaxis=dict(
            title="Built-up Area (sq ft)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            range=[df['built_up_area'].min() * 0.9, df['built_up_area'].max() * 1.1]
        ),
        yaxis=dict(
            title="Price (₹ Cr)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            range=[df['price'].min() * 0.8, df['price'].max() * 1.2]
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Ensure all elements are visible
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=1, color='darkgray')),
        selector=dict(mode='markers')
    )
    
    return fig


def get_property_type_data():
    """Centralized data prep for Property Type Analysis"""
    try:
        if data_viz.empty:
            raise ValueError("No data available for property type analysis")

        df = data_viz.copy()

        # Ensure price column consistency
        if price_column != "price" and price_column in df.columns:
            df["price"] = df[price_column]

        # If missing dependent columns, create them from available data
        if "price_per_sqft" not in df.columns and "built_up_area" in df.columns:
            df["price_per_sqft"] = df["price"] / df["built_up_area"].replace(0, np.nan)

        # Filter valid rows
        df = df.dropna(subset=["price", "property_type"])
        if df.empty:
            raise ValueError("No valid data available for property type analysis")

        return df

    except Exception as e:
        logger.error(f"Error preparing property type data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def property_type_analysis(df):
    """Generates Property Type Analysis Plotly Bar Chart with proper layout"""
    group_df = df.groupby('property_type', as_index=False).agg({
        'price': 'mean',
        'price_per_sqft': 'mean',
        'built_up_area': 'mean'
    })

    fig = px.bar(
        group_df,
        x='property_type',
        y='price',
        color='property_type',
        hover_data=['price_per_sqft', 'built_up_area'],
        text_auto='.2f',
        title='Average Price by Property Type'
    )
    
    # Proper layout constraints
    fig.update_layout(
        template='plotly_white',
        width=600,
        height=500,
        margin=dict(t=60, b=60, l=60, r=60, pad=10),
        xaxis=dict(
            title="Property Type",
            showgrid=False,
            tickangle=0
        ),
        yaxis=dict(
            title="Average Price (₹ Cr)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        showlegend=False
    )
    
    # Ensure text fits inside bars
    fig.update_traces(
        texttemplate='₹%{text:.2f} Cr',
        textposition='outside',
        marker=dict(line=dict(width=1, color='darkgray'))
    )
    
    return fig


# Sunburst chart with proper hierarchy structure
def sunburst_chart_simplified(df):
    """Sunburst chart with proper hierarchy structure"""
    try:
        # Ensure we have a proper hierarchical structure
        if df.empty:
            raise ValueError("No data available for sunburst chart")
            
        # Create proper parent-child relationships
        hierarchy_data = []
        
        # Add root level
        total_avg_price = df['price_per_sqft'].mean()
        hierarchy_data.append({
            'ids': 'All Properties',
            'labels': 'All Properties', 
            'parents': '',
            'values': len(df),
            'price_per_sqft': total_avg_price
        })
        
        # Add property types as children of root
        for prop_type in df['property_type'].unique():
            prop_data = df[df['property_type'] == prop_type]
            prop_avg_price = prop_data['price_per_sqft'].mean()
            
            hierarchy_data.append({
                'ids': prop_type,
                'labels': prop_type.capitalize(),
                'parents': 'All Properties',
                'values': len(prop_data),
                'price_per_sqft': prop_avg_price
            })
            
            # Add bedrooms as children of property types
            for bedroom in prop_data['bedRoom'].unique():
                bed_data = prop_data[prop_data['bedRoom'] == bedroom]
                bed_avg_price = bed_data['price_per_sqft'].mean()
                
                hierarchy_data.append({
                    'ids': f"{prop_type}_{bedroom}",
                    'labels': f"{bedroom} BHK",
                    'parents': prop_type,
                    'values': len(bed_data),
                    'price_per_sqft': bed_avg_price
                })
        
        # Convert to DataFrame for Plotly
        hierarchy_df = pd.DataFrame(hierarchy_data)
        
        fig = px.sunburst(
            hierarchy_df,
            names='labels',
            parents='parents',
            values='values',
            color='price_per_sqft',
            color_continuous_scale='Viridis',
            title='Property Type Analysis (Sunburst) — Price per Sqft by Property Type and Bedrooms'
        )
        
        # Proper layout to keep sunburst inside container
        fig.update_layout(
            template='plotly_white',
            width=600,
            height=500,
            margin=dict(t=60, b=20, l=20, r=20, pad=10),
            uniformtext=dict(minsize=12, mode='hide')
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating sunburst chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Simplified sunburst data prep
def get_sunburst_data_df(property_filter: str = "all"):
    """Simplified data prep for Property Type → Bedroom sunburst"""
    try:
        if data_viz.empty:
            raise ValueError("No data available for sunburst chart")
            
        df = data_viz.copy()
        if price_column != 'price' and price_column in df.columns:
            df["price"] = df[price_column]
        
        # Create price_per_sqft if missing
        if "price_per_sqft" not in df.columns and "built_up_area" in df.columns:
            df["price_per_sqft"] = df["price"] / df["built_up_area"].replace(0, np.nan)

        # Filter by property type only
        if property_filter != "all" and "property_type" in df.columns:
            df = df[df["property_type"] == property_filter]

        # Filter valid rows
        df = df.dropna(subset=["property_type", "bedRoom", "price_per_sqft"])
        
        if df.empty:
            raise ValueError("No valid data available for sunburst chart")

        return df

    except Exception as e:
        logger.error(f"Error preparing simplified sunburst data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ API ENDPOINTS ------------------

@app.get("/")
async def serve_frontend():
    frontend_index = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {"message": "Frontend files not found"}

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
            raise HTTPException(status_code=500, detail="No data available")
            
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

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "sklearn_version": sklearn.__version__,
        "model_loaded": pipeline is not None,
        "data_loaded": not df.empty,
        "recommender_loaded": not location_df.empty,
        "analysis_loaded": not data_viz.empty,
        "timestamp": datetime.now().isoformat()
    }

# ------------------ RECOMMENDER ENDPOINTS ------------------

@app.get("/api/recommender/options")
async def get_recommender_options():
    """Get dropdown options for recommender section"""
    try:
        if location_df.empty:
            raise HTTPException(status_code=500, detail="Location data not loaded")

        locations = sorted(location_df.columns.tolist())
        apartments = sorted(location_df.index.tolist())
        sectors = sorted(data_viz["sector"].unique().tolist()) if not data_viz.empty else []

        return {
            "locations": locations,
            "apartments": apartments,
            "sectors": sectors
        }
    except Exception as e:
        logger.error(f"Error loading recommender options: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading recommender options: {str(e)}")


@app.get("/api/recommender/location-search")
async def location_search(location: str, radius: float = 10.0):
    """Find nearby properties based on selected location and radius"""
    try:
        if location_df.empty:
            raise HTTPException(status_code=500, detail="Location data not loaded")

        if location not in location_df.columns:
            raise HTTPException(status_code=400, detail="Invalid location")

        if radius <= 0:
            raise HTTPException(status_code=400, detail="Radius must be positive")

        # Convert km to meters and filter
        radius_meters = radius * 1000
        result_series = location_df[location_df[location] < radius_meters][location].sort_values()

        results = []
        for property_name, distance_meters in result_series.items():
            results.append({
                "property": property_name,
                "distance": round(distance_meters / 1000, 2)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in location search: {e}")
        raise HTTPException(status_code=500, detail=f"Error in location search: {str(e)}")


@app.get("/api/recommender/recommend")
async def recommend_apartments(apartment: str, top_n: int = 5):
    """Recommend similar apartments using combined cosine similarity matrices"""
    try:
        if location_df.empty:
            raise HTTPException(status_code=500, detail="Location data not loaded")

        if any(sim is None for sim in [cosine_sim1, cosine_sim2, cosine_sim3]):
            raise HTTPException(status_code=500, detail="Similarity matrices not loaded")

        if apartment not in location_df.index:
            raise HTTPException(status_code=404, detail="Apartment not found in dataset")

        # Combine the three similarity matrices
        cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1.0 * cosine_sim3

        # Compute similarity scores for the selected apartment
        idx = location_df.index.get_loc(apartment)
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))

        # Sort by similarity score
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N similar apartments (excluding the same one)
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
        top_properties = location_df.index[top_indices].tolist()

        recommendations = [
            {"PropertyName": prop, "SimilarityScore": round(score, 3)}
            for prop, score in zip(top_properties, top_scores)
        ]

        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ ANALYSIS ENDPOINTS ------------------

WORDCLOUD_FILE = os.path.join(DATASET_PATH, "wordcloud.png")
wordcloud_exists = os.path.exists(WORDCLOUD_FILE)

@app.get("/api/analysis/generate-wordcloud")
async def generate_wordcloud_endpoint():
    """Generate dynamic wordcloud from feature text"""
    try:
        if not feature_text:
            raise HTTPException(status_code=500, detail="No feature text available for wordcloud generation")
        
        # Generate wordcloud
        wordcloud_buffer = generate_wordcloud_from_text(feature_text)
        
        if wordcloud_buffer:
            # Save to file for persistent access
            wordcloud_path = os.path.join(DATASET_PATH, "wordcloud_generated.png")
            with open(wordcloud_path, "wb") as f:
                f.write(wordcloud_buffer.getvalue())
            
            return {"message": "WordCloud generated successfully", "path": "/api/analysis/wordcloud"}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate wordcloud")
            
    except Exception as e:
        logger.error(f"WordCloud generation error: {e}")
        raise HTTPException(status_code=500, detail=f"WordCloud generation failed: {str(e)}")

@app.get("/api/analysis/wordcloud")
async def get_wordcloud():
    """Serve generated wordcloud image"""
    try:
        # Try generated wordcloud first
        generated_path = os.path.join(DATASET_PATH, "wordcloud_generated.png")
        if os.path.exists(generated_path):
            return FileResponse(generated_path)
        
        # Fallback to static wordcloud
        static_path = os.path.join(DATASET_PATH, "wordcloud.png")
        if os.path.exists(static_path):
            return FileResponse(static_path)
        
        # Generate new wordcloud if none exists
        await generate_wordcloud_endpoint()
        if os.path.exists(generated_path):
            return FileResponse(generated_path)
        
        raise HTTPException(status_code=404, detail="Wordcloud image not found")
        
    except Exception as e:
        logger.error(f"Error serving wordcloud: {e}")
        raise HTTPException(status_code=500, detail="Wordcloud unavailable")

@app.get("/api/analysis/area-vs-price")
async def get_area_vs_price(property_type: str = "all"):
    """Returns raw numeric data for frontend analytics"""
    try:
        df = get_area_vs_price_data(property_type)
        return {
            "area": df["built_up_area"].tolist(),
            "price": df["price"].tolist(),
            "property_type": df["property_type"].tolist(),
            "bedrooms": df["bedRoom"].tolist()
        }
    except Exception as e:
        logger.error(f"Error in /api/analysis/area-vs-price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts/area-vs-price")
async def get_dynamic_area_vs_price(property_type: str = "all"):
    """Returns full Plotly chart JSON for dynamic frontend rendering"""
    try:
        df = get_area_vs_price_data(property_type)
        fig = area_vs_price_chart(df, property_type)
        chart_json = json.loads(fig.to_json())
        return {"chart": chart_json, "property_type": property_type}
    except Exception as e:
        logger.error(f"Error generating chart /api/charts/area-vs-price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts/property-type-analysis")
async def get_property_type_analysis():
    """Property Type Analysis Chart - Average Price Comparison"""
    try:
        df = get_property_type_data()
        fig = property_type_analysis(df)
        chart_json = json.loads(fig.to_json())
        return {"chart": chart_json}

    except Exception as e:
        logger.error(f"Error generating property type analysis chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sunburst endpoint with proper hierarchy
@app.get("/api/charts/sunburst")
async def get_sunburst_chart(property_filter: str = "all"):
    """Sunburst: Proper hierarchical structure with single root"""
    try:
        df = get_sunburst_data_df(property_filter)
        fig = sunburst_chart_simplified(df)
        chart_json = json.loads(fig.to_json())
        return {"chart": chart_json, "filters": {"property_type": property_filter}}

    except Exception as e:
        logger.error(f"Error generating fixed sunburst chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Simplified sunburst data endpoint
@app.get("/api/analysis/sunburst-data")
async def get_sunburst_data(property_filter: str = "all"):
    """Returns simplified hierarchical data for property type → bedroom"""
    try:
        df = get_sunburst_data_df(property_filter)
        
        # Build proper hierarchy with single root
        hierarchy = [
            {"id": "root", "parent": "", "label": "All Properties", "value": len(df), "avg_price_per_sqft": df["price_per_sqft"].mean()}
        ]
        
        for prop_type in df["property_type"].unique():
            prop_data = df[df["property_type"] == prop_type]
            hierarchy.append({
                "id": prop_type,
                "parent": "root",
                "label": prop_type.capitalize(),
                "value": len(prop_data),
                "avg_price_per_sqft": prop_data["price_per_sqft"].mean()
            })
            
            for bedroom in sorted(prop_data["bedRoom"].unique()):
                bed_data = prop_data[prop_data["bedRoom"] == bedroom]
                hierarchy.append({
                    "id": f"{prop_type}_{bedroom}",
                    "parent": prop_type,
                    "label": f"{bedroom} BHK",
                    "value": len(bed_data),
                    "avg_price_per_sqft": bed_data["price_per_sqft"].mean()
                })
        
        return {
            "hierarchy": hierarchy,
            "filters": {"property_type": property_filter}
        }
    except Exception as e:
        logger.error(f"Error in simplified sunburst data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/bhk-pie")
async def get_bhk_pie(sector: str = "overall"):
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
        
        df_subset = data_viz
        if sector != "overall" and 'sector' in data_viz.columns:
            df_subset = data_viz[data_viz["sector"] == sector]
        
        if df_subset.empty or 'bedRoom' not in df_subset.columns:
            raise HTTPException(status_code=500, detail="No BHK data available")
            
        bhk_counts = df_subset["bedRoom"].value_counts().reset_index()
        bhk_counts.columns = ["bedRoom", "count"]
            
        return {
            "bedrooms": bhk_counts["bedRoom"].astype(int).tolist(),
            "counts": bhk_counts["count"].tolist()
        }
    except Exception as e:
        logger.error(f"Error processing BHK pie data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/price-distribution")
async def get_price_distribution():
    if data_viz.empty:
        raise HTTPException(status_code=500, detail="Analysis data not loaded")
    
    try:
        prices = data_viz[price_column].dropna().tolist() if price_column in data_viz.columns else data_viz['Price'].dropna().tolist()
        return {"prices": prices}
    except Exception as e:
        logger.error(f"Error processing price distribution data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/price-dist")
async def get_price_distribution_enhanced():
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
            
        house_prices = data_viz[data_viz["property_type"] == "house"][price_column].dropna().tolist() if "property_type" in data_viz.columns and price_column and price_column in data_viz.columns else []
        flat_prices = data_viz[data_viz["property_type"] == "flat"][price_column].dropna().tolist() if "property_type" in data_viz.columns and price_column and price_column in data_viz.columns else []
        return {
            "house_prices": house_prices,
            "flat_prices": flat_prices
        }
    except Exception as e:
        logger.error(f"Error loading price distribution data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/correlation")
async def get_correlation_heatmap():
    if data_viz.empty:
        raise HTTPException(status_code=500, detail="Analysis data not loaded")
    
    try:
        # Select numerical columns for correlation
        numerical_cols = data_viz.select_dtypes(include=[np.number]).columns
        corr_data = data_viz[numerical_cols].corr()
        
        return {
            "columns": corr_data.columns.tolist(),
            "values": corr_data.values.tolist()
        }
    except Exception as e:
        logger.error(f"Error processing correlation data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/correlation-data")
async def get_correlation_data():
    """Serve correlation data for heatmap"""
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
        
        # Select numeric columns for correlation
        numeric_cols = data_viz.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 10:  # Limit to top 10 numeric columns
            numeric_cols = numeric_cols[:10]
        
        corr_matrix = data_viz[numeric_cols].corr().round(2)
        
        return {
            "columns": numeric_cols,
            "correlation_matrix": corr_matrix.values.tolist()
        }
    except Exception as e:
        logger.error(f"Error loading correlation data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading correlation data: {str(e)}")

@app.get("/api/analysis/geomap")
async def get_geomap():
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
        
        # Check if required columns exist
        required_cols = ["sector", "price_per_sqft", "built_up_area", "latitude", "longitude"]
        missing_cols = [col for col in required_cols if col not in data_viz.columns]
        if missing_cols:
            raise HTTPException(status_code=500, detail=f"Missing required columns: {missing_cols}")
        
        # Group by sector and calculate averages
        group_df = data_viz.groupby("sector").agg({
            "price_per_sqft": "mean",
            "built_up_area": "mean",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()
        
        # Count properties per sector
        property_count = data_viz["sector"].value_counts().to_dict()
        group_df["property_count"] = group_df["sector"].map(property_count)
        
        # Filter out null coordinates
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
            
        avg_price = data_viz[price_column].mean() if price_column and price_column in data_viz.columns else 0
        sectors_covered = len(data_viz["sector"].unique()) if "sector" in data_viz.columns else 0
        
        return {
            "total_properties": len(data_viz),
            "avg_price": f"₹ {avg_price:.2f} Cr",
            "sectors_covered": sectors_covered,
            "model_accuracy": "89.2%",
            "last_updated": "2025-09-26"
        }
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/analysis-options")
async def get_analysis_options():
    """Get all available options for analysis dropdowns"""
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
        
        options = {}
        
        # Property types
        if 'property_type' in data_viz.columns:
            options['property_types'] = sorted(data_viz['property_type'].unique().tolist())
        else:
            options['property_types'] = []
        
        # Sectors
        if 'sector' in data_viz.columns:
            options['sectors'] = sorted(data_viz['sector'].unique().tolist())
        else:
            options['sectors'] = []
        
        # Bedrooms
        if 'bedRoom' in data_viz.columns:
            options['bedrooms'] = sorted(data_viz['bedRoom'].unique().tolist())
        else:
            options['bedrooms'] = []
        
        # Furnishing types
        if 'furnishing_type' in data_viz.columns:
            options['furnishing_types'] = sorted(data_viz['furnishing_type'].unique().tolist())
        else:
            options['furnishing_types'] = []
        
        # Age categories
        if 'agePossession' in data_viz.columns:
            options['age_categories'] = sorted(data_viz['agePossession'].unique().tolist())
        else:
            options['age_categories'] = []
        
        return options
        
    except Exception as e:
        logger.error(f"Error loading analysis options: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading analysis options: {str(e)}")

@app.get("/api/analysis/luxury-category")
async def get_luxury_category_data():
    """Get data for luxury category analysis"""
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
        
        if 'luxury_category' in data_viz.columns:
            luxury_data = data_viz.groupby('luxury_category').agg({
                price_column: ['count', 'mean'],
                'built_up_area': 'mean'
            }).reset_index()
            
            luxury_data.columns = ['category', 'count', 'avg_price', 'avg_area']
            
            return {
                "categories": luxury_data['category'].tolist(),
                "counts": luxury_data['count'].tolist(),
                "avg_prices": luxury_data['avg_price'].round(2).tolist(),
                "avg_areas": luxury_data['avg_area'].round(0).tolist()
            }
        else:
            raise HTTPException(status_code=500, detail="Luxury category data not available")
                
    except Exception as e:
        logger.error(f"Error generating luxury category data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/price-trend")
async def get_price_trend():
    """Get price trend data over time or sectors"""
    try:
        if data_viz.empty:
            raise HTTPException(status_code=500, detail="No analysis data available")
        
        if 'sector' in data_viz.columns and price_column in data_viz.columns:
            sector_prices = data_viz.groupby('sector')[price_column].mean().sort_values(ascending=False).head(10)
            
            return {
                "x_values": sector_prices.index.tolist(),
                "y_values": sector_prices.round(2).tolist(),
                "categories": ["Residential"] * len(sector_prices)
            }
        else:
            raise HTTPException(status_code=500, detail="Required columns not available for price trend")
                
    except Exception as e:
        logger.error(f"Error generating price trend data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/viz/{viz_name}")
async def get_visualization(viz_name: str):
    """Serve pre-generated visualization HTML files"""
    try:
        viz_path = os.path.join(DATASET_PATH, f"{viz_name}.html")
        if os.path.exists(viz_path):
            return FileResponse(viz_path)
        else:
            raise HTTPException(status_code=404, detail=f"Visualization {viz_name} not found")
    except Exception as e:
        logger.error(f"Error serving visualization {viz_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving visualization: {str(e)}")

# ------------------ DEBUG ENDPOINT ------------------

@app.get("/api/debug")
async def debug_info():
    files = os.listdir(DATASET_PATH) if os.path.exists(DATASET_PATH) else []
    return {
        "dataset_path": DATASET_PATH,
        "files_in_dataset": files,
        "df_loaded": not df.empty,
        "pipeline_loaded": pipeline is not None,
        "location_data_loaded": not location_df.empty,
        "data_viz_loaded": not data_viz.empty,
        "feature_text_loaded": bool(feature_text),
        "data_viz_columns": data_viz.columns.tolist() if not data_viz.empty else []
    }

# Serve other frontend files
@app.get("/{path:path}")
async def serve_static(path: str):
    static_file = os.path.join(frontend_path, path)
    if os.path.exists(static_file) and os.path.isfile(static_file):
        return FileResponse(static_file)
    # Fallback to index.html for SPA routing
    frontend_index = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {"message": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)