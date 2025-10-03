import React from 'react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-brand">
            <i className="fas fa-robot"></i>
            <span>RealEstate<span className="ai-text">AI</span></span>
          </div>
          <p>Machine Learning Project • Built with modern technologies</p>
          <div className="footer-links">
            <a 
              href="https://github.com/your-username"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i className="fab fa-github"></i>
            </a>
            <a 
              href="https://www.linkedin.com/in/your-profile"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i className="fab fa-linkedin"></i>
            </a>
            <a href="mailto:your-email@example.com">
              <i className="fas fa-envelope"></i>
            </a>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; {currentYear} Real Estate AI Analytics. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
