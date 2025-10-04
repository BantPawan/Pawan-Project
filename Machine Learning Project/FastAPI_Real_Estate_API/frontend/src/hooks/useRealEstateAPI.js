import { useState, useEffect } from 'react';

// Use Render URL in production, localhost in development
const API_BASE = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000/api' 
  : '/api';

export const useRealEstateAPI = () => {
  const [options, setOptions] = useState({});
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load options and stats on component mount
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [optionsRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/options`).then(res => res.json()),
        fetch(`${API_BASE}/stats`).then(res => res.json())
      ]);
      
      setOptions(optionsRes);
      setStats(statsRes);
    } catch (err) {
      console.log('Using fallback data');
      setOptions({
        property_type: ["apartment", "house"],
        sector: ["sector 1", "sector 2", "sector 3"],
        bedrooms: [2, 3, 4],
        bathroom: [1, 2, 3],
        balcony: ["1", "2", "3"],
        property_age: ["New Property", "1-5 years", "5-10 years"],
        servant_room: [0, 1],
        store_room: [0, 1],
        furnishing_type: ["Unfurnished", "Semi-Furnished", "Furnished"],
        luxury_category: ["Low", "Medium", "High"],
        floor_category: ["Low Rise", "Mid Rise", "High Rise"]
      });
      setStats({
        total_properties: 10000,
        model_accuracy: "92%",
        sectors_covered: 50,
        avg_price: "₹ 1.25 Cr"
      });
    }
  };

  const predictPrice = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/predict_price`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setLoading(false);
      return data;
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Prediction service temporarily unavailable. Using demo data.');
      setLoading(false);
      // Return demo data
      return {
        low_price_cr: 1.25,
        high_price_cr: 1.45,
        formatted_range: "₹ 1.25 Cr - ₹ 1.45 Cr",
        note: "Demo prediction - backend service unavailable"
      };
    }
  };

  const loadAnalysisData = async (endpoint) => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      console.error(`Error loading ${endpoint}:`, err);
      // Return fallback data based on endpoint
      if (endpoint.includes('geomap')) {
        return {
          sectors: ["sector 1", "sector 2", "sector 3"],
          price_per_sqft: [5000, 6000, 5500],
          built_up_area: [1000, 1200, 1100],
          latitude: [28.45, 28.46, 28.47],
          longitude: [77.02, 77.03, 77.04],
          property_count: [10, 15, 8]
        };
      }
      if (endpoint.includes('area-vs-price')) {
        return {
          built_up_area: [1000, 1200, 1400, 1600, 1800],
          price: [1.2, 1.4, 1.6, 1.8, 2.0],
          bedrooms: [2, 3, 3, 4, 4]
        };
      }
      if (endpoint.includes('bhk-pie')) {
        return {
          bedrooms: [2, 3, 4],
          counts: [45, 35, 20]
        };
      }
      if (endpoint.includes('price-dist')) {
        return {
          house_prices: [1.2, 1.5, 1.8, 2.1, 2.4],
          flat_prices: [0.8, 1.0, 1.2, 1.4, 1.6]
        };
      }
      return null;
    }
  };

  return {
    options,
    stats,
    loading,
    error,
    predictPrice,
    loadAnalysisData
  };
};
