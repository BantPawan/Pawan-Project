import React, { useState } from 'react';
import { useRealEstateAPI } from '../hooks/useRealEstateAPI';

const Predictor = () => {
  const { options, loading, error, predictPrice } = useRealEstateAPI();
  const [formData, setFormData] = useState({
    property_type: '',
    sector: '',
    bedrooms: '',
    bathroom: '',
    balcony: '',
    property_age: '',
    built_up_area: '1200',
    servant_room: '0',
    store_room: '0',
    furnishing_type: '',
    luxury_category: '',
    floor_category: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [formError, setFormError] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    if (formError) setFormError('');
  };

  const validateForm = () => {
    const requiredFields = [
      'property_type', 'sector', 'bedrooms', 'bathroom', 'balcony', 
      'property_age', 'built_up_area', 'furnishing_type', 
      'luxury_category', 'floor_category'
    ];

    const missingFields = requiredFields.filter(field => !formData[field]);
    if (missingFields.length > 0) {
      setFormError('Please fill in all required fields');
      return false;
    }

    if (!formData.built_up_area || formData.built_up_area < 500) {
      setFormError('Built-up area must be at least 500 sq ft');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError('');
    setPrediction(null);

    if (!validateForm()) return;

    const processedData = {
      ...formData,
      bedrooms: parseFloat(formData.bedrooms),
      bathroom: parseFloat(formData.bathroom),
      built_up_area: parseFloat(formData.built_up_area),
      servant_room: parseFloat(formData.servant_room),
      store_room: parseFloat(formData.store_room)
    };

    const result = await predictPrice(processedData);
    setPrediction(result);
  };

  const SelectField = ({ id, label, icon, value, required = true }) => (
    <div className="form-group">
      <label htmlFor={id}>
        <i className={icon}></i> {label}
      </label>
      <select 
        id={id}
        name={id}
        value={value}
        onChange={handleInputChange}
        required={required}
      >
        <option value="">Select...</option>
        {options[id]?.map(option => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </div>
  );

  const InputField = ({ id, label, icon, type = 'text', value, required = true }) => (
    <div className="form-group">
      <label htmlFor={id}>
        <i className={icon}></i> {label}
      </label>
      <input 
        type={type}
        id={id}
        name={id}
        value={value}
        onChange={handleInputChange}
        min="500"
        max="10000"
        step="50"
        required={required}
      />
    </div>
  );

  return (
    <section id="predictor" className="predictor-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">🏠 Intelligent Price Predictor</h2>
          <p className="section-subtitle">Enter property details to get AI-powered price estimation</p>
        </div>
        
        <div className="predictor-container">
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="form-grid">
              <div className="form-column">
                <SelectField
                  id="property_type"
                  label="Property Type"
                  icon="fas fa-building"
                  value={formData.property_type}
                />
                <SelectField
                  id="sector"
                  label="Sector/Location"
                  icon="fas fa-map-marker-alt"
                  value={formData.sector}
                />
                <SelectField
                  id="bedrooms"
                  label="Bedrooms"
                  icon="fas fa-bed"
                  value={formData.bedrooms}
                />
              </div>
              
              <div className="form-column">
                <SelectField
                  id="bathroom"
                  label="Bathrooms"
                  icon="fas fa-bath"
                  value={formData.bathroom}
                />
                <SelectField
                  id="balcony"
                  label="Balconies"
                  icon="fas fa-door-open"
                  value={formData.balcony}
                />
                <InputField
                  id="built_up_area"
                  label="Built-up Area (sq ft)"
                  icon="fas fa-arrows-alt"
                  type="number"
                  value={formData.built_up_area}
                />
              </div>
              
              <div className="form-column">
                <SelectField
                  id="property_age"
                  label="Property Age"
                  icon="fas fa-calendar"
                  value={formData.property_age}
                />
                <SelectField
                  id="furnishing_type"
                  label="Furnishing Type"
                  icon="fas fa-couch"
                  value={formData.furnishing_type}
                />
                <SelectField
                  id="luxury_category"
                  label="Luxury Category"
                  icon="fas fa-gem"
                  value={formData.luxury_category}
                />
              </div>
            </div>

            <div className="form-grid">
              <div className="form-column">
                <SelectField
                  id="servant_room"
                  label="Servant Room"
                  icon="fas fa-user"
                  value={formData.servant_room}
                />
              </div>
              <div className="form-column">
                <SelectField
                  id="store_room"
                  label="Store Room"
                  icon="fas fa-box"
                  value={formData.store_room}
                />
              </div>
              <div className="form-column">
                <SelectField
                  id="floor_category"
                  label="Floor Category"
                  icon="fas fa-building"
                  value={formData.floor_category}
                />
              </div>
            </div>

            <button type="submit" className="predict-btn" disabled={loading}>
              <i className="fas fa-calculator"></i> 
              <span className="btn-text">
                {loading ? 'Predicting...' : 'Predict Price'}
              </span>
              {loading && <div className="btn-loader"></div>}
            </button>
          </form>
          
          {formError && (
            <div className="error-message">
              <i className="fas fa-exclamation-triangle"></i>
              {formError}
            </div>
          )}
          
          {error && (
            <div className="error-message">
              <i className="fas fa-exclamation-triangle"></i>
              {error}
            </div>
          )}
          
          {prediction && (
            <div className="result-container">
              <div className="result-header">
                <i className="fas fa-chart-line"></i>
                <h3>AI Prediction Result</h3>
              </div>
              <div className="price-range">{prediction.formatted_range}</div>
              <div className="confidence-meter">
                <div className="confidence-label">Model Confidence: 92%</div>
                <div className="confidence-bar">
                  <div className="confidence-fill"></div>
                </div>
              </div>
              <div className="result-details">
                <div className="detail-item">
                  <span>Low Estimate:</span>
                  <strong>₹ {prediction.low_price_cr} Cr</strong>
                </div>
                <div className="detail-item">
                  <span>High Estimate:</span>
                  <strong>₹ {prediction.high_price_cr} Cr</strong>
                </div>
              </div>
              {prediction.note && <p className="note">{prediction.note}</p>}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default Predictor;
