import React, { useState, useEffect } from 'react';
import { useRealEstateAPI } from '../hooks/useRealEstateAPI';

const Analysis = () => {
  const { loadAnalysisData } = useRealEstateAPI();
  const [analysisData, setAnalysisData] = useState({
    geomap: null,
    areaPrice: null,
    bhkPie: null
  });
  const [selectedPropertyType, setSelectedPropertyType] = useState('flat');
  const [selectedSector, setSelectedSector] = useState('overall');

  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    try {
      const [geomap, areaPrice, bhkPie] = await Promise.all([
        loadAnalysisData('/api/analysis/geomap'),
        loadAnalysisData('/api/analysis/area-vs-price?property_type=flat'),
        loadAnalysisData('/api/analysis/bhk-pie?sector=overall')
      ]);

      setAnalysisData({
        geomap,
        areaPrice,
        bhkPie
      });
    } catch (error) {
      console.log('Using mock analysis data');
    }
  };

  const GeomapDisplay = () => {
    if (!analysisData.geomap) return <div className="plot-placeholder">Loading map data...</div>;

    return (
      <div className="map-container">
        <div className="map-overview">
          <h4>Price Distribution by Sector</h4>
          <div className="sector-list">
            {analysisData.geomap.sectors.map((sector, index) => (
              <div key={sector} className="sector-item">
                <span className="sector-name">{sector}</span>
                <span className="sector-price">₹{analysisData.geomap.price_per_sqft[index]}/sqft</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const AreaPriceDisplay = () => {
    if (!analysisData.areaPrice) return <div className="plot-placeholder">Loading area vs price data...</div>;

    return (
      <div className="chart-container">
        <h4>Area vs Price Correlation</h4>
        <div className="data-points">
          {analysisData.areaPrice.built_up_area.map((area, index) => (
            <div key={index} className="data-point">
              <div className="point-details">
                <span>{area} sqft</span>
                <span>₹ {analysisData.areaPrice.price[index]} Cr</span>
                <span>{analysisData.areaPrice.bedrooms[index]} BHK</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const BhkPieDisplay = () => {
    if (!analysisData.bhkPie) return <div className="plot-placeholder">Loading BHK distribution...</div>;

    const total = analysisData.bhkPie.counts.reduce((sum, count) => sum + count, 0);

    return (
      <div className="pie-container">
        <h4>BHK Distribution</h4>
        <div className="pie-chart">
          {analysisData.bhkPie.bedrooms.map((bhk, index) => {
            const percentage = ((analysisData.bhkPie.counts[index] / total) * 100).toFixed(1);
            return (
              <div key={bhk} className="pie-segment">
                <div className="segment-label">
                  <span className="bhk">{bhk} BHK</span>
                  <span className="count">{analysisData.bhkPie.counts[index]} properties ({percentage}%)</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <section id="analysis" className="analysis-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">📊 Market Analysis</h2>
          <p className="section-subtitle">Explore market trends and property insights</p>
        </div>

        <div className="analysis-container">
          <div className="analysis-item">
            <h3>📍 Sector Price Distribution</h3>
            <div className="plot-container">
              <GeomapDisplay />
            </div>
          </div>

          <div className="analysis-item">
            <h3>📐 Area vs Price</h3>
            <select 
              className="analysis-select"
              value={selectedPropertyType}
              onChange={(e) => setSelectedPropertyType(e.target.value)}
            >
              <option value="flat">Flat</option>
              <option value="house">House</option>
            </select>
            <div className="plot-container">
              <AreaPriceDisplay />
            </div>
          </div>

          <div className="analysis-item">
            <h3>🏠 BHK Distribution</h3>
            <select 
              className="analysis-select"
              value={selectedSector}
              onChange={(e) => setSelectedSector(e.target.value)}
            >
              <option value="overall">Overall</option>
              <option value="sector 45">Sector 45</option>
              <option value="sector 46">Sector 46</option>
            </select>
            <div className="plot-container">
              <BhkPieDisplay />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Analysis;
