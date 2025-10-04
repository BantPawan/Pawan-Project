import React from 'react';

const About = () => {
  const highlights = [
    {
      icon: 'fas fa-database',
      title: 'Data Processing',
      description: '10,000+ records cleaned and analyzed'
    },
    {
      icon: 'fas fa-robot',
      title: 'ML Pipeline',
      description: 'Custom regression model deployment'
    },
    {
      icon: 'fas fa-server',
      title: 'API Development',
      description: 'RESTful endpoints with FastAPI'
    }
  ];

  return (
    <section id="about" className="about-section">
      <div className="container">
        <div className="about-grid">
          <div className="about-content">
            <h2>About This Project</h2>
            <p>
              This Real Estate Analytics platform represents a full-stack machine learning project, 
              showcasing modern development skills with cutting-edge technologies.
            </p>
            
            <div className="project-highlights">
              {highlights.map((highlight, index) => (
                <div key={index} className="highlight">
                  <i className={highlight.icon}></i>
                  <div>
                    <strong>{highlight.title}</strong>
                    <span>{highlight.description}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="tech-details">
              <h4>Technical Stack:</h4>
              <ul>
                <li><strong>Backend:</strong> Python, FastAPI, Scikit-learn, Pandas, NumPy</li>
                <li><strong>Frontend:</strong> React, JavaScript (ES6+), Vite</li>
                <li><strong>ML Models:</strong> Regression, Feature Engineering, Cross-validation</li>
                <li><strong>Deployment:</strong> Render, Netlify, Git</li>
              </ul>
            </div>

            <div className="project-links">
              <a 
                href="https://github.com/your-username/your-repo" 
                className="btn btn-outline"
                target="_blank"
                rel="noopener noreferrer"
              >
                <i className="fab fa-github"></i> View Source Code
              </a>
              <a href="mailto:your-email@example.com" className="btn btn-outline">
                <i className="fas fa-envelope"></i> Contact Me
              </a>
            </div>
          </div>
          
          <div className="about-visual">
            <div className="code-window">
              <div className="window-header">
                <div className="window-buttons">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span className="window-title">model_training.py</span>
              </div>
              <div className="code-content">
                <pre><code>{`# Machine Learning Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Feature engineering
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2
)

# Model training
model = RandomForestRegressor(n_estimators=100)
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Accuracy: {scores.mean():.2f}")`}</code></pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
