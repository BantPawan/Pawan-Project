import { useState, useEffect } from 'react';

// Auto-detect API base URL
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
        fetch(`${API_BASE}/options`).then(res => {
          if (!res.ok) throw new Error('Failed to load options');
          return res.json();
        }),
        fetch(`${API_BASE}/stats`).then(res => {
          if (!res.ok) throw new Error('Failed to load stats');
          return res.json();
        })
      ]);
      
      setOptions(optionsRes);
      setStats(statsRes);
    } catch (err) {
      console.error('Error loading initial data:', err);
      setError('Failed to load application data');
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
        throw new Error(`Prediction failed: ${response.status}`);
      }

      const data = await response.json();
      setLoading(false);
      return data;
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Prediction service temporarily unavailable');
      setLoading(false);
      throw err;
    }
  };

  const loadAnalysisData = async (endpoint) => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`);
      if (!response.ok) {
        throw new Error(`Failed to load data: ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      console.error(`Error loading ${endpoint}:`, err);
      throw err;
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
