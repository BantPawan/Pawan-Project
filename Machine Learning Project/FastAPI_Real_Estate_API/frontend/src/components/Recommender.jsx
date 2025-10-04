import React, { useState, useEffect } from 'react';
import { useRealEstateAPI } from '../hooks/useRealEstateAPI';

const Recommender = () => {
  const { loadAnalysisData } = useRealEstateAPI();
  const [searchType, setSearchType] = useState('location');
  const [recommenderOptions, setRecommenderOptions] = useState({});
  const [location, setLocation] = useState('');
  const [radius, setRadius] = useState(1.0);
  const [propertyName, setPropertyName] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadOptions = async () => {
      try {
        const data = await loadAnalysisData('/api/recommender/options');
        setRecommenderOptions(data);
        if (data.locations?.length > 0) {
          setLocation(data.locations[0]);
        }
        if (data.apartments?.length > 0) {
          setPropertyName(data.apartments[0]);
        }
      } catch (err) {
        console.log('Using mock recommender options');
        setRecommenderOptions({
          locations: ["Sector 45", "Sector 46", "Sector 47"],
          apartments: ["Grand Apartments", "Modern Villas", "Luxury Homes"],
          sectors: ["sector 45", "sector 46", "sector 47"]
        });
        setLocation("Sector 45");
        setPropertyName("Grand Apartments");
      }
    };
    loadOptions();
  }, [loadAnalysisData]);

  const handleLocationSearch = async () => {
    if (!location || !radius) return;
    setLoading(true);
    try {
      const endpoint = `/api/recommender/location-search?location=${encodeURIComponent(location)}&radius=${radius}`;
      const data = await loadAnalysisData(endpoint);
      setResults(data);
    } catch (err) {
      console.error(err);
      setResults([
        { property: "Grand Apartments", distance: 0.8 },
        { property: "Modern Villas", distance: 1.2 },
        { property: "Luxury Homes", distance: 0.5 },
        { property: "Green Valley Apartments", distance: 1.8 }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSimilarSearch = async () => {
    if (!propertyName) return;
    setLoading(true);
    try {
      const endpoint = `/api/recommender/recommend?property_name=${encodeURIComponent(propertyName)}&top_n=5`;
      const data = await loadAnalysisData(endpoint);
      setResults(data);
    } catch (err) {
      console.error(err);
      setResults([
        { PropertyName: "Similar Apartment A", SimilarityScore: 0.95 },
        { PropertyName: "Similar Apartment B", SimilarityScore: 0.89 },
        { PropertyName: "Similar Apartment C", SimilarityScore: 0.82 },
        { PropertyName: "Similar Apartment D", SimilarityScore: 0.78 }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="recommender" className="recommender-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">🏡 Property Recommender</h2>
          <p className="section-subtitle">Find properties based on location or similarity</p>
        </div>

        <div className="recommender-container">
          <div className="recommender-tabs">
            <button 
              className={`tab-btn ${searchType === 'location' ? 'active' : ''}`}
              onClick={() => setSearchType('location')}
            >
              📍 Location Based
            </button>
            <button 
              className={`tab-btn ${searchType === 'similar' ? 'active' : ''}`}
              onClick={() => setSearchType('similar')}
            >
              🔍 Similar Properties
            </button>
          </div>

          {searchType === 'location' && (
            <div className="recommender-form">
              <h3>Find Properties Near Location</h3>
              <div className="form-group">
                <label>Location</label>
                <select value={location} onChange={(e) => setLocation(e.target.value)}>
                  {recommenderOptions.locations?.map((loc) => (
                    <option key={loc} value={loc}>{loc}</option>
                  )) || <option>Loading...</option>}
                </select>
              </div>
              <div className="form-group">
                <label>Radius (km)</label>
                <select value={radius} onChange={(e) => setRadius(parseFloat(e.target.value))}>
                  <option value={0.5}>0.5</option>
                  <option value={1.0}>1.0</option>
                  <option value={2.0}>2.0</option>
                </select>
              </div>
              <button 
                className="predict-btn"
                onClick={handleLocationSearch}
                disabled={loading}
              >
                {loading ? 'Searching...' : 'Search Locations'}
              </button>
            </div>
          )}

          {searchType === 'similar' && (
            <div className="recommender-form">
              <h3>Find Similar Properties</h3>
              <div className="form-group">
                <label>Select Property</label>
                <select value={propertyName} onChange={(e) => setPropertyName(e.target.value)}>
                  {recommenderOptions.apartments?.map((apt) => (
                    <option key={apt} value={apt}>{apt}</option>
                  )) || <option>Loading...</option>}
                </select>
              </div>
              <button 
                className="predict-btn"
                onClick={handleSimilarSearch}
                disabled={loading}
              >
                {loading ? 'Finding...' : 'Find Similar'}
              </button>
            </div>
          )}

          {results.length > 0 && (
            <div className="results-container">
              <h3>Recommended Properties</h3>
              <div className="results-list">
                {results.map((result, index) => (
                  <div key={index} className="result-card">
                    <div className="property-name">
                      {result.property || result.PropertyName}
                    </div>
                    {result.distance && (
                      <div className="property-distance">
                        {result.distance} km away
                      </div>
                    )}
                    {result.SimilarityScore && (
                      <div className="similarity-score">
                        {(result.SimilarityScore * 100).toFixed(1)}% match
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default Recommender;
