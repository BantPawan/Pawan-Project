import { useState, useEffect } from 'react';

const API_BASE = '/api';

export const useRealEstateAPI = () => {
  const [options, setOptions] = useState({});
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [optionsData, statsData] = await Promise.all([
        fetch(`${API_BASE}/api/options`).then(res => res.json()),
        fetch(`${API_BASE}/api/stats`).then(res => res.json())
      ]);
      setOptions(optionsData);
      setStats(statsData);
    } catch (err) {
      console.log('Using fallback data');
      setOptions({
        property_type: ["flat", "house"],
        sector: ["sector 45", "sector 46", "sector 47"],
        bedrooms: [2, 3, 4],
        bathroom: [2, 3],
        balcony: ["1", "2", "3"],
        property_age: ["New Property", "Relatively New"],
        servant_room: [0, 1],
        store_room: [0, 1],
        furnishing_type: ["unfurnished", "semifurnished", "furnished"],
        luxury_category: ["Low", "Medium", "High"],
        floor_category: ["Low Floor", "Mid Floor", "High Floor"]
      });
      setStats({
        total_properties: 10000,
        model_accuracy: "92%",
        sectors_covered: 50
      });
    }
  };

  const predictPrice = async (formData) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/api/predict_price`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) throw new Error('Prediction failed');
      
      const result = await response.json();
      return result;
    } catch (err) {
      setError(err.message);
      // Return mock prediction
      return {
        formatted_range: "₹ 1.20 Cr - ₹ 1.50 Cr",
        low_price_cr: 1.20,
        high_price_cr: 1.50,
        note: "Demo prediction - backend might be loading"
      };
    } finally {
      setLoading(false);
    }
  };

  const loadAnalysisData = async (endpoint) => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`);
      if (!response.ok) throw new Error(`Failed to load ${endpoint}`);
      return await response.json();
    } catch (err) {
      console.log(`Using mock data for ${endpoint}`);
      // Return mock data
      if (endpoint.includes('geomap')) {
        return {
          sectors: ["Sector 45", "Sector 46", "Sector 47"],
          price_per_sqft: [8500, 9200, 7800],
          built_up_area: [1200, 1500, 1800],
          latitude: [28.4595, 28.4612, 28.4630],
          longitude: [77.0266, 77.0280, 77.0300]
        };
      }
      if (endpoint.includes('area-vs-price')) {
        return {
          built_up_area: [1200, 1500, 1800],
          price: [1.2, 1.5, 1.8],
          bedrooms: [2, 3, 4]
        };
      }
      if (endpoint.includes('bhk-pie')) {
        return {
          bedrooms: [2, 3, 4],
          counts: [30, 45, 25]
        };
      }
      return {};
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
