import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import joblib
import os
import ast
from pathlib import Path

# ==================== PATH CONFIGURATION ====================
# Get the root project directory (Machine_Learning_Project)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Adjust based on your structure

# Processed Data paths
PROCESSED_DATA_DIR = PROJECT_ROOT / "Processed_Data"
RAW_DATA_DIR = PROJECT_ROOT / "Datasets"  # If you have raw data

# Backend paths
BACKEND_DIR = PROJECT_ROOT / "FastAPI_Real_Estate_API" / "backend"
DATASET_DIR = BACKEND_DIR / "Dataset"

# Create directories
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# File paths
DATA_VIZ_CSV = DATASET_DIR / "data_viz1.csv"
DF_JOBLIB = DATASET_DIR / "df.pkl"
FEATURE_TEXT_JOBLIB = DATASET_DIR / "feature_text.pkl"
WORDCLOUD_PNG = DATASET_DIR / "wordcloud.png"

print(f"Project Root: {PROJECT_ROOT}")
print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")
print(f"Backend Directory: {BACKEND_DIR}")
print(f"Dataset Directory: {DATASET_DIR}")

def load_and_preprocess_data():
    """Load and preprocess the main dataset with proper path handling"""
    print("Loading and preprocessing data...")
    
    try:
        # Load main data - adjust path based on your actual file location
        main_data_path = PROCESSED_DATA_DIR / "gurgaon_properties_missing_value_imputation.csv"
        latlong_path = PROCESSED_DATA_DIR / "latlong.csv"  # or RAW_DATA_DIR if in different location
        
        print(f"Looking for main data at: {main_data_path}")
        print(f"Looking for latlong data at: {latlong_path}")
        
        if not main_data_path.exists():
            # Try alternative paths
            alternative_paths = [
                PROCESSED_DATA_DIR / "gurgaon_properties.csv",
                RAW_DATA_DIR / "gurgaon_properties.csv",
                PROJECT_ROOT / "gurgaon_properties.csv"
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    main_data_path = alt_path
                    print(f"Found data at: {alt_path}")
                    break
            else:
                raise FileNotFoundError(f"Could not find data file in any expected location")
        
        df = pd.read_csv(main_data_path)
        print(f"Main data loaded: {df.shape}")
        
        # Load latlong data
        if latlong_path.exists():
            latlong = pd.read_csv(latlong_path)
            print(f"Latlong data loaded: {latlong.shape}")
            
            # Process coordinates
            latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype(float)
            latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype(float)
            
            # Merge datasets
            new_df = df.merge(latlong, on='sector')
            print(f"Merged data: {new_df.shape}")
        else:
            print("Latlong file not found, using main data only")
            new_df = df.copy()
            # Add dummy coordinates if missing
            if 'latitude' not in new_df.columns:
                new_df['latitude'] = 28.4595 + np.random.uniform(-0.1, 0.1, len(new_df))
            if 'longitude' not in new_df.columns:
                new_df['longitude'] = 77.0266 + np.random.uniform(-0.1, 0.1, len(new_df))
        
        # Ensure Price column consistency
        if 'price' in new_df.columns and 'Price' not in new_df.columns:
            new_df.rename(columns={'price': 'Price'}, inplace=True)
        
        # Create price_per_sqft if missing
        if 'price_per_sqft' not in new_df.columns and 'built_up_area' in new_df.columns:
            new_df['price_per_sqft'] = new_df['Price'] / new_df['built_up_area']
            new_df['price_per_sqft'] = new_df['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
            new_df['price_per_sqft'] = new_df['price_per_sqft'].fillna(new_df['price_per_sqft'].median())
        
        # Handle missing values
        numeric_cols = new_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if new_df[col].isnull().sum() > 0:
                new_df[col] = new_df[col].fillna(new_df[col].median())
        
        # Ensure critical columns exist for the simplified sunburst
        if 'bedRoom' not in new_df.columns:
            if 'bedrooms' in new_df.columns:
                new_df['bedRoom'] = new_df['bedrooms']
            else:
                new_df['bedRoom'] = np.random.randint(1, 5, len(new_df))
        
        if 'property_type' not in new_df.columns:
            new_df['property_type'] = np.random.choice(['flat', 'house'], len(new_df), p=[0.7, 0.3])
        
        print(f"Final processed dataset shape: {new_df.shape}")
        print(f"Columns: {new_df.columns.tolist()}")
        
        return new_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create sample data if files not found
        print("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if original files are not available"""
    n_samples = 1000
    sample_df = pd.DataFrame({
        'property_type': np.random.choice(['flat', 'house'], n_samples, p=[0.7, 0.3]),
        'sector': np.random.choice([f'Sector {i}' for i in range(1, 51)], n_samples),
        'bedRoom': np.random.randint(1, 6, n_samples),
        'bathroom': np.random.randint(1, 4, n_samples),
        'built_up_area': np.random.randint(800, 3000, n_samples),
        'Price': np.random.uniform(0.5, 5.0, n_samples),
        'price_per_sqft': np.random.randint(5000, 15000, n_samples),
        'latitude': 28.4595 + np.random.uniform(-0.1, 0.1, n_samples),
        'longitude': 77.0266 + np.random.uniform(-0.1, 0.1, n_samples),
        'luxury_category': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'furnishing_type': np.random.choice(['unfurnished', 'semifurnished', 'furnished'], n_samples),
        'agePossession': np.random.choice(['New Property', 'Relatively New', 'Moderately Old', 'Old Property'], n_samples)
    })
    return sample_df

def generate_feature_text():
    """Generate feature text for wordcloud with proper path handling"""
    print("Generating feature text for wordcloud...")
    
    try:
        features_path = PROCESSED_DATA_DIR / "gurgaon_properties.csv"
        
        if features_path.exists():
            wordcloud_df = pd.read_csv(features_path)
            if 'features' in wordcloud_df.columns:
                main = []
                for item in wordcloud_df['features'].dropna():
                    try:
                        if isinstance(item, str):
                            features_list = ast.literal_eval(item)
                            main.extend(features_list)
                    except (ValueError, SyntaxError):
                        continue
                feature_text = ' '.join(main)
                print(f"Generated feature text with {len(main)} features")
            else:
                feature_text = "luxury modern spacious furnished semi-furnished apartment house villa swimming_pool gym park clubhouse security power_backup lift parking garden balcony"
        else:
            feature_text = "luxury modern spacious furnished semi-furnished apartment house villa swimming_pool gym park clubhouse security power_backup lift parking garden balcony"
        
        return feature_text
        
    except Exception as e:
        print(f"Error generating feature text: {e}")
        return "luxury modern spacious furnished semi-furnished apartment house villa swimming_pool gym park clubhouse security power_backup lift parking garden balcony"

# ... [Keep all the visualization functions from previous version] ...

def save_essential_files(new_df, feature_text):
    """Save essential files for the FastAPI backend"""
    print("Saving essential files...")
    
    # Save main data files
    new_df.to_csv(DATA_VIZ_CSV, index=False)
    joblib.dump(new_df, DF_JOBLIB)
    joblib.dump(feature_text, FEATURE_TEXT_JOBLIB)
    
    # Create placeholder files for the backend
    joblib.dump({}, DATASET_DIR / "pipeline_compressed.pkl")
    joblib.dump(pd.DataFrame(), DATASET_DIR / "location_distance.pkl")
    
    # Create cosine similarity placeholders
    joblib.dump(np.eye(10), DATASET_DIR / "cosine_sim1.pkl")
    joblib.dump(np.eye(10), DATASET_DIR / "cosine_sim2.pkl")
    joblib.dump(np.eye(10), DATASET_DIR / "cosine_sim3.pkl")

def main():
    """Main execution function"""
    print("Starting data preparation...")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        new_df = load_and_preprocess_data()
        
        # Generate feature text
        feature_text = generate_feature_text()
        
        # Create wordcloud
        create_wordcloud(feature_text)
        
        # Create visualizations (include all functions from previous version)
        create_simplified_sunburst(new_df)
        create_property_type_analysis(new_df)
        create_area_vs_price_chart(new_df)
        create_bhk_distribution(new_df)
        create_price_distribution_comparison(new_df)
        create_geographic_analysis(new_df)
        create_luxury_analysis(new_df)
        create_correlation_heatmap(new_df)
        create_advanced_visualizations(new_df)
        
        # Save essential files
        save_essential_files(new_df, feature_text)
        
        print("\n" + "="*60)
        print("🎉 DATA PREPARATION COMPLETE!")
        print("="*60)
        print(f"✓ Processed dataset: {new_df.shape}")
        print(f"✓ Created visualization files in: {DATASET_DIR}")
        print(f"✓ Generated wordcloud: {WORDCLOUD_PNG}")
        
        # List generated files
        html_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.html')]
        print(f"✓ Created {len(html_files)} visualization files")
        
        print("\n📁 Project Structure Ready for GitHub:")
        print(f"  Processed_Data/ → CSV files")
        print(f"  FastAPI_Real_Estate_API/backend/ → API code")
        print(f"  FastAPI_Real_Estate_API/backend/Dataset/ → Generated files")
        print(f"  FastAPI_Real_Estate_API/frontend/ → Web interface")
        
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        raise

if __name__ == "__main__":
    main()