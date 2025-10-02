import React from 'react';

const Features = () => {
  const features = [
    {
      icon: 'fas fa-brain',
      title: 'Machine Learning',
      description: 'Advanced regression models trained on 10,000+ property records with 92% prediction accuracy'
    },
    {
      icon: 'fas fa-bolt',
      title: 'FastAPI Backend',
      description: 'High-performance REST API with automatic documentation and validation'
    },
    {
      icon: 'fas fa-chart-bar',
      title: 'Data Analytics',
      description: 'Interactive visualizations and market trend analysis'
    },
    {
      icon: 'fas fa-thumbs-up',
      title: 'Recommendations',
      description: 'Personalized apartment recommendations based on similarity scores'
    },
    {
      icon: 'fas fa-mobile-alt',
      title: 'Responsive Design',
      description: 'Optimized for all devices with modern UI/UX'
    },
    {
      icon: 'fas fa-cloud',
      title: 'Cloud Deployment',
      description: 'Fully deployed on Render with CI/CD pipeline'
    }
  ];

  return (
    <section id="features" className="features-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">✨ Project Features</h2>
          <p className="section-subtitle">Comprehensive real estate analytics platform</p>
        </div>
        
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-icon">
                <i className={feature.icon}></i>
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
