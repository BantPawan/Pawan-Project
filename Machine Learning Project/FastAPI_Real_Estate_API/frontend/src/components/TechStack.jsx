import React from 'react';

const TechStack = () => {
  const techItems = [
    { icon: 'fab fa-python', name: 'Python' },
    { icon: 'fas fa-brain', name: 'Scikit-learn' },
    { icon: 'fas fa-bolt', name: 'FastAPI' },
    { icon: 'fab fa-react', name: 'React' },
    { icon: 'fas fa-chart-bar', name: 'Pandas' },
    { icon: 'fas fa-cloud', name: 'Render' }
  ];

  return (
    <section className="tech-stack">
      <div className="container">
        <h2 className="section-title">Built With Modern Tech Stack</h2>
        <div className="tech-grid">
          {techItems.map((tech, index) => (
            <div key={index} className="tech-item">
              <i className={tech.icon}></i>
              <span>{tech.name}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TechStack;
