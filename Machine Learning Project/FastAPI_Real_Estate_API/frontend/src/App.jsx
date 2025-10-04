import React from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import TechStack from './components/TechStack';
import Predictor from './components/Predictor';
import Analysis from './components/Analysis';
import Recommender from './components/Recommender';
import Features from './components/Features';
import About from './components/About';
import Footer from './components/Footer';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Hero />
      <TechStack />
      <Predictor />
      <Analysis />
      <Recommender />
      <Features />
      <About />
      <Footer />
    </div>
  );
}

export default App;
