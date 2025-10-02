import React, { useState, useEffect } from 'react';
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

  useEffect(() => {
    if (error) {
      setFormError(error);
    }
  }, [error]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear errors when user starts typing
    if (formError) setFormError('');
  };

  const validateForm = () => {
    const requiredFields = [
      'property_type', 'sector', 'bedrooms', 'bathroom', 'balcony', 
      'property_age', 'built_up_area', 'servant_room', 'store_room',
      'furnishing_type', 'luxury_category', 'floor_category'
    ];

    const missingFields = requiredFields.filter(field => !formData[field]);
    if (missingFields.length > 0) {
      setFormError(`Please fill in all required fields: ${missingFields.join(', ')}`);
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

    try {
      const numericFields = ['bedrooms', 'bathroom', 'built_up_area', 'servant_room', 'store_room'];
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
    } catch (err) {
      // Error is handled in the hook
    }
  };

  const SelectField = ({ id, label, icon, options: fieldOptions, value, required = true }) => (
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
        {fieldOptions?.map(option => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </div>
  );

  const InputField = ({ id, label, icon, type = 'text', min, max, step, value, required = true }) => (
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
        min={min}
        max={max}
        step={step}
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
                  options={options.property_type}
                  value={formData.property_type}
                />
                <SelectField
                  id="sector"
                  label="Sector/Location"
                  icon="fas fa-map-marker-alt"
                  options={options.sector}
                  value={formData.sector}
                />
                <SelectField
                  id="bedrooms"
                  label="Bedrooms"
                  icon="fas fa-bed"
                  options={options.bedrooms}
                  value={formData.bedrooms}
                />
              </div>
              
              <div className="form-column">
                <SelectField
                  id="bathroom"
                  label="Bathrooms"
                  icon="fas fa-bath"
                  options={options.bathroom}
                  value={formData.bathroom}
                />
                <SelectField
                  id="balcony"
                  label="Balconies"
                  icon="fas fa-door-open"
                  options={options.balcony}
                  value={formData.balcony}
                />
                <InputField
                  id="built_up_area"
                  label="Built-up Area (sq ft)"
                  icon="fas fa-arrows-alt"
                  type="number"
                  min="500"
                  max="10000"
                  step="50"
                  value={formData.built_up_area}
                />
              </div>
              
              <div className="form-column">
                <SelectField
                  id="property_age"
                  label="Property Age"
                  icon="fas fa-calendar"
                  options={options.property_age}
                  value={formData.property_age}
                />
                <SelectField
                  id="furnishing_type"
                  label="Furnishing Type"
                  icon="fas fa-couch"
                  options={options.furnishing_type}
                  value={formData.furnishing_type}
                />
                <SelectField
                  id="luxury_category"
                  label="Luxury Category"
                  icon="fas fa-gem"
                  options={options.luxury_category}
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
                  options={options.servant_room}
                  value={formData.servant_room}
                />
              </div>
              <div className="form-column">
                <SelectField
                  id="store_room"
                  label="Store Room"
                  icon="fas fa-box"
                  options={options.store_room}
                  value={formData.store_room}
                />
              </div>
              <div className="form-column">
                <SelectField
                  id="floor_category"
                  label="Floor Category"
                  icon="fas fa-building"
                  options={options.floor_category}
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
                <div className="detail-item">
                  <span>Best Value:</span>
                  <strong>₹ {prediction.prediction_raw.toFixed(2)} Cr</strong>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default Predictor;
