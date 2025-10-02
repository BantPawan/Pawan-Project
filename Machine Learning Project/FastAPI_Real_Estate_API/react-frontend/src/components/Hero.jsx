import React from 'react';
import { useRealEstateAPI } from '../hooks/useRealEstateAPI';

const Hero = () => {
  const { stats } = useRealEstateAPI();

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="home" className="hero">
      <div className="hero-container">
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="gradient-text">AI-Powered</span> Real Estate Analytics
          </h1>
          <p className="hero-subtitle">
            Machine Learning project that predicts property prices, analyzes market trends, and recommends apartments with 92% accuracy using 
            <strong> FastAPI</strong>, <strong>React</strong>, and advanced data analytics
          </p>
          <div className="hero-stats">
            <div className="stat">
              <div className="stat-number" id="stat-properties">
                {stats.total_properties ? stats.total_properties.toLocaleString() : '10,000+'}
              </div>
              <div className="stat-label">Properties Analyzed</div>
            </div>
            <div className="stat">
              <div className="stat-number" id="stat-accuracy">
                {stats.model_accuracy || '92%'}
              </div>
              <div className="stat-label">Prediction Accuracy</div>
            </div>
            <div className="stat">
              <div className="stat-number" id="stat-sectors">
                {stats.sectors_covered || '50+'}
              </div>
              <div className="stat-label">Sectors Covered</div>
            </div>
          </div>
          <div className="hero-buttons">
            <button 
              className="btn btn-primary"
              onClick={() => scrollToSection('predictor')}
            >
              <i className="fas fa-rocket"></i> Try Predictor
            </button>
            <button 
              className="btn btn-secondary"
              onClick={() => scrollToSection('analysis')}
            >
              <i className="fas fa-chart-bar"></i> View Analysis
            </button>
          </div>
        </div>
        <div className="hero-visual">
          <div className="floating-card">
            <div className="card-header">
              <i className="fas fa-chart-line"></i>
              <span>Live Prediction</span>
            </div>
            <div className="card-content">
              <div className="prediction-demo">
                <div className="demo-property">3BHK Apartment • Sector 45</div>
                <div className="demo-price">₹ 1.25 - 1.45 Cr</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
