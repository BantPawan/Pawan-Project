import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { useRealEstateAPI } from '../hooks/useRealEstateAPI';

const Analysis = () => {
  const { loadAnalysisData } = useRealEstateAPI();
  const [analysisData, setAnalysisData] = useState({
    geomap: null,
    wordcloud: null,
    areaPrice: null,
    bhkPie: null,
    priceDist: null
  });
  const [selectedPropertyType, setSelectedPropertyType] = useState('flat');
  const [selectedSector, setSelectedSector] = useState('overall');

  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    try {
      const [geomap, wordcloud, areaPrice, bhkPie, priceDist] = await Promise.all([
        loadAnalysisData('/api/analysis/geomap'),
        loadAnalysisData('/api/analysis/wordcloud'),
        loadAnalysisData(`/api/analysis/area-vs-price?property_type=${selectedPropertyType}`),
        loadAnalysisData(`/api/analysis/bhk-pie?sector=${selectedSector}`),
        loadAnalysisData('/api/analysis/price-dist')
      ]);
      setAnalysisData({ geomap, wordcloud, areaPrice, bhkPie, priceDist });
    } catch (error) {
      console.error('Analysis load error:', error);
    }
  };

  // Re-fetch on change
  useEffect(() => {
    loadAllData();
  }, [selectedPropertyType, selectedSector]);

  if (!analysisData.geomap) return <div className="plot-placeholder">Loading analysis...</div>;

  return (
    <section id="analysis" className="analysis-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">📊 Market Analysis</h2>
          <p className="section-subtitle">Explore market trends with interactive visualizations</p>
        </div>

        <div className="analysis-container">
          {/* Geomap */}
          <div className="analysis-item">
            <h3>📍 Sector Price per Sqft Geomap</h3>
            <Plot
              data={[
                {
                  type: 'scattermapbox',
                  lat: analysisData.geomap.latitude,
                  lon: analysisData.geomap.longitude,
                  mode: 'markers',
                  marker: {
                    size: analysisData.geomap.built_up_area.map(x => Math.sqrt(x) / 10),
                    color: analysisData.geomap.price_per_sqft,
                    colorscale: 'Icefire',
                    showscale: true
                  },
                  text: analysisData.geomap.sectors,
                  hoverinfo: 'text'
                }
              ]}
              layout={{
                mapbox: { style: 'open-street-map', center: { lat: 28.46, lon: 77.03 }, zoom: 10 },
                width: 800,
                height: 500,
                margin: { t: 0, b: 0, l: 0, r: 0 }
              }}
              config={{ displayModeBar: true, toImageButtonOptions: { format: 'png' } }}
            />
          </div>

          {/* Wordcloud */}
          <div className="analysis-item">
            <h3>🌩️ Features Wordcloud</h3>
            {analysisData.wordcloud && analysisData.wordcloud.image_url ? (
              <img src={analysisData.wordcloud.image_url} alt="Wordcloud" style={{ width: '100%', height: '400px', objectFit: 'contain' }} />
            ) : <div className="plot-placeholder">Loading wordcloud...</div>}
          </div>

          {/* Area vs Price */}
          <div className="analysis-item">
            <h3>📐 Area vs Price</h3>
            <select 
              value={selectedPropertyType} 
              onChange={(e) => setSelectedPropertyType(e.target.value)} 
              className="analysis-select"
            >
              <option value="flat">Flat</option>
              <option value="house">House</option>
            </select>
            {analysisData.areaPrice && analysisData.areaPrice.built_up_area.length > 0 ? (
              <Plot
                data={[
                  {
                    type: 'scatter',
                    x: analysisData.areaPrice.built_up_area,
                    y: analysisData.areaPrice.price,
                    mode: 'markers',
                    marker: { color: analysisData.areaPrice.bedrooms, size: 10, showscale: true },
                    text: analysisData.areaPrice.bedrooms,
                    hoverinfo: 'x+y+text'
                  }
                ]}
                layout={{
                  title: `Area vs Price for ${selectedPropertyType}`,
                  xaxis: { title: 'Built-up Area (sq ft)' },
                  yaxis: { title: 'Price (Cr)' },
                  height: 500
                }}
                config={{ displayModeBar: true }}
              />
            ) : <div className="plot-placeholder">Loading area vs price...</div>}
          </div>

          {/* BHK Pie */}
          <div className="analysis-item">
            <h3>🏠 BHK Distribution</h3>
            <select 
              value={selectedSector} 
              onChange={(e) => setSelectedSector(e.target.value)} 
              className="analysis-select"
            >
              <option value="overall">Overall</option>
              {/* Real sectors loaded from options */}
            </select>
            {analysisData.bhkPie && analysisData.bhkPie.bedrooms.length > 0 ? (
              <Plot
                data={[
                  {
                    type: 'pie',
                    labels: analysisData.bhkPie.bedrooms,
                    values: analysisData.bhkPie.counts,
                    textinfo: 'percent+label'
                  }
                ]}
                layout={{ title: `BHK Distribution in ${selectedSector}`, height: 500 }}
                config={{ displayModeBar: true }}
              />
            ) : <div className="plot-placeholder">Loading BHK distribution...</div>}
          </div>

          {/* Price Dist */}
          <div className="analysis-item">
            <h3>📈 Price Distribution by Property Type</h3>
            {analysisData.priceDist && (analysisData.priceDist.house_prices.length > 0 || analysisData.priceDist.flat_prices.length > 0) ? (
              <Plot
                data={[
                  { type: 'histogram', x: analysisData.priceDist.house_prices, name: 'House', opacity: 0.5, histnorm: 'density' },
                  { type: 'histogram', x: analysisData.priceDist.flat_prices, name: 'Flat', opacity: 0.5, histnorm: 'density' }
                ]}
                layout={{ title: 'Price Distribution', xaxis: { title: 'Price (Cr)' }, yaxis: { title: 'Density' }, barmode: 'overlay', height: 500 }}
                config={{ displayModeBar: true }}
              />
            ) : <div className="plot-placeholder">Loading price distribution...</div>}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Analysis;
