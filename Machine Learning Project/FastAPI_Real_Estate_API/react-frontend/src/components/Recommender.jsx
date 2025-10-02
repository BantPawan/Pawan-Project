import React, { useState, useEffect } from 'react';
import { useRealEstateAPI } from '../hooks/useRealEstateAPI';

const Recommender = () => {
  const { loadRecommenderOptions, searchLocations, getRecommendations } = useRealEstateAPI();
  const [recommenderOptions, setRecommenderOptions] = useState({
    locations: [],
    apartments: [],
    sectors: []
  });
  const [locationSearch, setLocationSearch] = useState({
    location: '',
    radius: '1.0'
  });
  const [recommendation, setRecommendation] = useState({
    apartment: ''
  });
  const [locationResults, setLocationResults] = useState([]);
  const [recommendationResults, setRecommendationResults] = useState([]);
  const [loading, setLoading] = useState({
    locations: false,
    recommendations: false
  });
  const [error, setError] = useState('');

  useEffect(() => {
    loadOptions();
  }, []);

  const loadOptions = async () => {
    try {
      const options = await loadRecommenderOptions();
      setRecommenderOptions(options);
    } catch (err) {
      console.error('Error loading recommender options:', err);
    }
  };

  const handleLocationSearch = async () => {
    if (!locationSearch.location || !locationSearch.radius) {
      setError('Please select a location and enter a radius');
      return;
    }

    setLoading(prev => ({...prev, locations: true}));
    setError('');

    try {
      const results = await searchLocations(locationSearch.location, parseFloat(locationSearch.radius));
      setLocationResults(results);
    } catch (err) {
      setError('Failed to search locations. Please try again.');
    } finally {
      setLoading(prev => ({...prev, locations: false}));
    }
  };

  const handleGetRecommendations = async () => {
    if (!recommendation.apartment) {
      setError('Please select an apartment');
      return;
    }

    setLoading(prev => ({...prev, recommendations: true}));
    setError('');

    try {
      const results = await getRecommendations(recommendation.apartment);
      setRecommendationResults(results);
    } catch (err) {
      setError('Failed to get recommendations. Please try again.');
    } finally {
      setLoading(prev => ({...prev, recommendations: false}));
    }
  };

  const handleInputChange = (section, field, value) => {
    if (section === 'location') {
      setLocationSearch(prev => ({...prev, [field]: value}));
    } else if (section === 'recommendation') {
      setRecommendation(prev => ({...prev, [field]: value}));
    }
    if (error) setError('');
  };

  return (
    <section id="recommender" className="recommender-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">🏡 Apartment Recommender</h2>
          <p className="section-subtitle">Find similar apartments based on location and features</p>
        </div>

        <div className="recommender-container">
          {/* Location Search */}
          <div className="recommender-form">
            <h3>Location-based Search</h3>
            
            <div className="form-group">
              <label htmlFor="location-search">
                <i className="fas fa-map-marker-alt"></i> Location
              </label>
              <select 
                id="location-search"
                value={locationSearch.location}
                onChange={(e) => handleInputChange('location', 'location', e.target.value)}
                required
              >
                <option value="">Select...</option>
                {recommenderOptions.locations?.map(location => (
                  <option key={location} value={location}>{location}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="radius">
                <i className="fas fa-ruler"></i> Radius (Kms)
              </label>
              <input 
                type="number" 
                id="radius"
                value={locationSearch.radius}
                onChange={(e) => handleInputChange('location', 'radius', e.target.value)}
                min="0.1" 
                step="0.1" 
                required
              />
            </div>

            <button 
              className="predict-btn"
              onClick={handleLocationSearch}
              disabled={loading.locations}
            >
              <i className="fas fa-search"></i>
              <span className="btn-text">
                {loading.locations ? 'Searching...' : 'Search Locations'}
              </span>
              {loading.locations && <div className="btn-loader"></div>}
            </button>

            {locationResults.length > 0 && (
              <div className="result-container">
                <h4>Properties within {locationSearch.radius} km:</h4>
                {locationResults.map((result, index) => (
                  <div key={index} className="result-item">
                    {result.property} ({result.distance.toFixed(1)} kms)
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Apartment Recommendations */}
          <div className="recommender-form">
            <h3>Apartment Recommendations</h3>
            
            <div className="form-group">
              <label htmlFor="apartment-select">
                <i className="fas fa-building"></i> Select Apartment
              </label>
              <select 
                id="apartment-select"
                value={recommendation.apartment}
                onChange={(e) => handleInputChange('recommendation', 'apartment', e.target.value)}
                required
              >
                <option value="">Select...</option>
                {recommenderOptions.apartments?.map(apartment => (
                  <option key={apartment} value={apartment}>{apartment}</option>
                ))}
              </select>
            </div>

            <button 
              className="predict-btn"
              onClick={handleGetRecommendations}
              disabled={loading.recommendations}
            >
              <i className="fas fa-thumbs-up"></i>
              <span className="btn-text">
                {loading.recommendations ? 'Getting...' : 'Get Recommendations'}
              </span>
              {loading.recommendations && <div className="btn-loader"></div>}
            </button>

            {recommendationResults.length > 0 && (
              <div className="result-container">
                <h4>Similar Properties:</h4>
                <table className="recommend-table">
                  <thead>
                    <tr>
                      <th>Property Name</th>
                      <th>Similarity Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recommendationResults.map((result, index) => (
                      <tr key={index}>
                        <td>{result.PropertyName}</td>
                        <td>{result.SimilarityScore.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="error-message" style={{ marginTop: '20px' }}>
            <i className="fas fa-exclamation-triangle"></i>
            {error}
          </div>
        )}
      </div>
    </section>
  );
};

export default Recommender;
