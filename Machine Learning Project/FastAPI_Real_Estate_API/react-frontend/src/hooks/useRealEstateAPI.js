import { useState, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';

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
      await Promise.all([loadOptions(), loadStats()]);
    } catch (err) {
      setError('Failed to load initial data');
      console.error('Error loading initial data:', err);
    }
  };

  const loadOptions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/options`);
      if (!response.ok) throw new Error('Failed to load options');
      const data = await response.json();
      setOptions(data);
    } catch (err) {
      console.error('Error loading options:', err);
      throw err;
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/stats`);
      if (!response.ok) throw new Error('Failed to load stats');
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error loading stats:', err);
      throw err;
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
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }
      
      const result = await response.json();
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
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
      console.error(`Error loading ${endpoint}:`, err);
      throw err;
    }
  };

  const loadRecommenderOptions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/recommender/options`);
      if (!response.ok) throw new Error('Failed to load recommender options');
      return await response.json();
    } catch (err) {
      console.error('Error loading recommender options:', err);
      throw err;
    }
  };

  const searchLocations = async (location, radius) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/recommender/location-search?location=${encodeURIComponent(location)}&radius=${radius}`
      );
      if (!response.ok) throw new Error('Location search failed');
      return await response.json();
    } catch (err) {
      console.error('Error in location search:', err);
      throw err;
    }
  };

  const getRecommendations = async (propertyName, topN = 5) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/recommender/recommend?property_name=${encodeURIComponent(propertyName)}&top_n=${topN}`
      );
      if (!response.ok) throw new Error('Failed to get recommendations');
      return await response.json();
    } catch (err) {
      console.error('Error getting recommendations:', err);
      throw err;
    }
  };

  return {
    options,
    stats,
    loading,
    error,
    predictPrice,
    loadAnalysisData,
    loadRecommenderOptions,
    searchLocations,
    getRecommendations,
  };
};
