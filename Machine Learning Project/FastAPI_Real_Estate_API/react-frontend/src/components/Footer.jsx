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
          <p>Machine Learning Project • Built with ❤️ for the developer community</p>
          <div className="footer-links">
            <a 
              href="https://github.com/BantPawan/Pawan-Project"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i className="fab fa-github"></i>
            </a>
            <a 
              href="https://www.linkedin.com/in/pawan-bant"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i className="fab fa-linkedin"></i>
            </a>
            <a href="mailto:bantpawan@gmail.com">
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
