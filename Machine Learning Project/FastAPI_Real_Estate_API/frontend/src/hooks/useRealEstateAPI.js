import { useState, useEffect } from 'react';

const API_BASE = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000/api' 
  : '/api';

export const useRealEstateAPI = () => {
  const [options, setOptions] = useState({});
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadInitialData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Check if backend is ready
      const healthResponse = await fetch(`${API_BASE}/health`);
      const healthData = await healthResponse.json();
      
      if (!healthData.data_loaded) {
        throw new Error('Backend is starting up, please wait...');
      }

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
      setLoading(false);
    } catch (err) {
      console.error('Error loading initial data:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

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
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setLoading(false);
      return data;
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message);
      setLoading(false);
      throw err;
    }
  };

  const loadAnalysisData = async (endpoint) => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to load data');
      }
      return await response.json();
    } catch (err) {
      console.error(`Error loading ${endpoint}:`, err);
      throw err;
    }
  };

  const retry = () => {
    loadInitialData();
  };

  return {
    options,
    stats,
    loading,
    error,
    predictPrice,
    loadAnalysisData,
    retry
  };
};
