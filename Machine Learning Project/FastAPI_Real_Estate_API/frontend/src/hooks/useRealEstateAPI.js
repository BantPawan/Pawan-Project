import { useState, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';

// Normalize to fix double /api (e.g., /api/api/options -> /api/options)
const normalizeEndpoint = (endpoint) => {
  console.log(`Normalizing: base="${API_BASE}", ep="${endpoint}"`);  // Debug log
  let base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE;
  let ep = endpoint.startsWith('/') ? endpoint : '/' + endpoint;
  if (base.endsWith('/api') && ep.startsWith('/api')) {
    ep = ep.replace(/^\/api/, '');  // Strip leading /api
  }
  const full = base + ep;
  console.log(`Normalized to: ${full}`);  // Debug
  return full;
};

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
      const [optionsRes, statsRes] = await Promise.all([
        fetch(normalizeEndpoint('/api/options')),
        fetch(normalizeEndpoint('/api/stats'))
      ]);
      if (!optionsRes.ok || !statsRes.ok) throw new Error('API fetch failed');
      const [optionsData, statsData] = await Promise.all([optionsRes.json(), statsRes.json()]);
      setOptions(optionsData);
      setStats(statsData);
      console.log('Loaded real options/stats:', optionsData, statsData);  // Debug
    } catch (err) {
      console.error('Fallback to mocks:', err);
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
      const response = await fetch(normalizeEndpoint('/api/predict_price'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();
      console.log('Real prediction:', result);  // Debug
      return result;
    } catch (err) {
      setError(err.message);
      console.error('Prediction fallback:', err);
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
      const response = await fetch(normalizeEndpoint(endpoint));
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      console.log(`Loaded real ${endpoint}:`, data);  // Debug
      return data;
    } catch (err) {
      console.error(`Mock for ${endpoint}:`, err);
      // Specific mocks only on error
      if (endpoint.includes('wordcloud')) return { image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==' };  // Tiny placeholder
      if (endpoint.includes('geomap')) return { sectors: ["Sector 45"], price_per_sqft: [8500], built_up_area: [1200], latitude: [28.4595], longitude: [77.0266], property_count: [150] };
      if (endpoint.includes('area-vs-price')) return { built_up_area: [1200], price: [1.2], bedrooms: [2] };
      if (endpoint.includes('bhk-pie')) return { bedrooms: [2], counts: [30] };
      if (endpoint.includes('price-dist')) return { house_prices: [1.5], flat_prices: [1.2] };
      return {};
    }
  };

  return { options, stats, loading, error, predictPrice, loadAnalysisData };
};
