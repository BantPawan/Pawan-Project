import React, { useState } from 'react';

const Recommender = () => {
  const [searchType, setSearchType] = useState('location');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const mockLocationSearch = () => {
    return [
      { property: "Grand Apartments", distance: 0.8 },
      { property: "Modern Villas", distance: 1.2 },
      { property: "Luxury Homes", distance: 0.5 },
      { property: "Green Valley Apartments", distance: 1.8 }
    ];
  };

  const mockSimilarProperties = () => {
    return [
      { PropertyName: "Similar Apartment A", SimilarityScore: 0.95 },
      { PropertyName: "Similar Apartment B", SimilarityScore: 0.89 },
      { PropertyName: "Similar Apartment C", SimilarityScore: 0.82 },
      { PropertyName: "Similar Apartment D", SimilarityScore: 0.78 }
    ];
  };

  const handleLocationSearch = async () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setResults(mockLocationSearch());
      setLoading(false);
    }, 1000);
  };

  const handleSimilarSearch = async () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setResults(mockSimilarProperties());
      setLoading(false);
    }, 1000);
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
                <select>
                  <option>Sector 45</option>
                  <option>Sector 46</option>
                  <option>Sector 47</option>
                </select>
              </div>
              <div className="form-group">
                <label>Radius (km)</label>
                <select>
                  <option>0.5</option>
                  <option>1.0</option>
                  <option>2.0</option>
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
                <select>
                  <option>Grand Apartments - 3BHK</option>
                  <option>Modern Villas - 4BHK</option>
                  <option>Luxury Homes - 3BHK</option>
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
