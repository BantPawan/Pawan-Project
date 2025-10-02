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
    priceDist: null,
    correlation: null,
    luxuryScores: null,
    priceTrend: null
  });
  const [loading, setLoading] = useState({});
  const [selectedPropertyType, setSelectedPropertyType] = useState('flat');
  const [selectedSector, setSelectedSector] = useState('overall');

  useEffect(() => {
    loadAllData();
  }, []);

  useEffect(() => {
    loadAreaVsPrice();
  }, [selectedPropertyType]);

  useEffect(() => {
    loadBhkPie();
  }, [selectedSector]);

  const loadAllData = async () => {
    try {
      setLoading(prev => ({...prev, all: true}));
      
      const [
        geomap, 
        wordcloud, 
        areaPrice, 
        bhkPie, 
        priceDist,
        correlation,
        luxuryScores,
        priceTrend
      ] = await Promise.all([
        loadAnalysisData('/api/analysis/geomap'),
        loadAnalysisData('/api/analysis/wordcloud'),
        loadAnalysisData('/api/analysis/area-vs-price?property_type=flat'),
        loadAnalysisData('/api/analysis/bhk-pie?sector=overall'),
        loadAnalysisData('/api/analysis/price-dist'),
        loadAnalysisData('/api/analysis/correlation'),
        loadAnalysisData('/api/analysis/luxury-score'),
        loadAnalysisData('/api/analysis/price-trend')
      ]);

      setAnalysisData({
        geomap,
        wordcloud,
        areaPrice,
        bhkPie,
        priceDist,
        correlation,
        luxuryScores,
        priceTrend
      });
    } catch (error) {
      console.error('Error loading analysis data:', error);
    } finally {
      setLoading(prev => ({...prev, all: false}));
    }
  };

  const loadAreaVsPrice = async () => {
    setLoading(prev => ({...prev, areaPrice: true}));
    try {
      const data = await loadAnalysisData(`/api/analysis/area-vs-price?property_type=${selectedPropertyType}`);
      setAnalysisData(prev => ({...prev, areaPrice: data}));
    } catch (error) {
      console.error('Error loading area vs price:', error);
    } finally {
      setLoading(prev => ({...prev, areaPrice: false}));
    }
  };

  const loadBhkPie = async () => {
    setLoading(prev => ({...prev, bhkPie: true}));
    try {
      const data = await loadAnalysisData(`/api/analysis/bhk-pie?sector=${selectedSector}`);
      setAnalysisData(prev => ({...prev, bhkPie: data}));
    } catch (error) {
      console.error('Error loading BHK pie:', error);
    } finally {
      setLoading(prev => ({...prev, bhkPie: false}));
    }
  };

  const GeomapChart = () => {
    if (!analysisData.geomap) return <div className="plot-placeholder">Loading map...</div>;

    const { geomap } = analysisData;
    const markerSizes = geomap.built_up_area.map(area => 
      Math.max(10, Math.min(30, Math.sqrt(area) / 5))
    );

    return (
      <Plot
        data={[{
          type: 'scattermapbox',
          lat: geomap.latitude,
          lon: geomap.longitude,
          mode: 'markers',
          marker: {
            size: markerSizes,
            color: geomap.price_per_sqft,
            colorscale: 'Viridis',
            colorbar: { title: 'Price per Sqft', titleside: 'right' },
            opacity: 0.8,
            sizemode: 'diameter'
          },
          text: geomap.sectors.map((sector, i) => 
            `${sector}<br>Price: ₹${geomap.price_per_sqft[i]?.toLocaleString() || 'N/A'}/sqft<br>Area: ${geomap.built_up_area[i] || 'N/A'} sqft`
          ),
          hoverinfo: 'text'
        }]}
        layout={{
          mapbox: {
            style: 'open-street-map',
            center: { 
              lat: geomap.latitude.reduce((a, b) => a + b, 0) / geomap.latitude.length,
              lon: geomap.longitude.reduce((a, b) => a + b, 0) / geomap.longitude.length
            },
            zoom: 11
          },
          margin: { t: 0, b: 0, l: 0, r: 0 },
          height: 500,
          showlegend: false
        }}
        style={{ width: '100%', height: '100%' }}
        config={{ responsive: true }}
      />
    );
  };

  const WordcloudDisplay = () => {
    if (!analysisData.wordcloud) return <div className="plot-placeholder">Loading wordcloud...</div>;

    return (
      <div style={{ textAlign: 'center', padding: '10px' }}>
        <img 
          src={analysisData.wordcloud.image_url} 
          alt="Property Features Wordcloud"
          style={{ 
            maxWidth: '100%', 
            height: 'auto', 
            borderRadius: '8px',
            background: analysisData.wordcloud.image_url.includes('placeholder') ? '#f8fafc' : 'black'
          }}
        />
        {analysisData.wordcloud.message && (
          <p style={{ color: '#6b7280', fontSize: '0.9rem', marginTop: '10px' }}>
            {analysisData.wordcloud.message}
          </p>
        )}
      </div>
    );
  };

  const AreaPriceChart = () => {
    if (!analysisData.areaPrice) return <div className="plot-placeholder">Loading chart...</div>;

    return (
      <Plot
        data={[{
          type: 'scatter',
          x: analysisData.areaPrice.built_up_area,
          y: analysisData.areaPrice.price,
          mode: 'markers',
          marker: {
            color: analysisData.areaPrice.bedrooms,
            size: 10,
            colorscale: 'Viridis',
            showscale: true
          },
          text: analysisData.areaPrice.bedrooms?.map(b => `${b} BHK`),
          hoverinfo: 'x+y+text'
        }]}
        layout={{
          title: `Area vs Price for ${selectedPropertyType.charAt(0).toUpperCase() + selectedPropertyType.slice(1)}`,
          xaxis: { title: 'Built-up Area (sq ft)' },
          yaxis: { title: 'Price (Cr)' },
          height: 400
        }}
      />
    );
  };

  const BhkPieChart = () => {
    if (!analysisData.bhkPie) return <div className="plot-placeholder">Loading chart...</div>;

    return (
      <Plot
        data={[{
          type: 'pie',
          labels: analysisData.bhkPie.bedrooms.map(b => `${b} BHK`),
          values: analysisData.bhkPie.counts,
          textinfo: 'percent+label',
          hoverinfo: 'label+percent+value',
          hole: 0.4
        }]}
        layout={{
          title: `BHK Distribution in ${selectedSector === 'overall' ? 'Overall Data' : selectedSector}`,
          height: 400,
          showlegend: true
        }}
      />
    );
  };

  const PriceDistributionChart = () => {
    if (!analysisData.priceDist) return <div className="plot-placeholder">Loading chart...</div>;

    return (
      <Plot
        data={[
          {
            type: 'histogram',
            x: analysisData.priceDist.house_prices,
            name: 'House',
            opacity: 0.7,
            nbinsx: 20
          },
          {
            type: 'histogram',
            x: analysisData.priceDist.flat_prices,
            name: 'Flat',
            opacity: 0.7,
            nbinsx: 20
          }
        ]}
        layout={{
          title: 'Price Distribution by Property Type',
          xaxis: { title: 'Price (Cr)' },
          yaxis: { title: 'Count' },
          barmode: 'overlay',
          height: 400
        }}
      />
    );
  };

  const CorrelationHeatmap = () => {
    if (!analysisData.correlation) return <div className="plot-placeholder">Loading heatmap...</div>;

    return (
      <Plot
        data={[{
          z: analysisData.correlation.correlation_matrix,
          x: analysisData.correlation.features,
          y: analysisData.correlation.features,
          type: 'heatmap',
          colorscale: 'RdBu',
          zmin: -1,
          zmax: 1,
          text: analysisData.correlation.correlation_matrix.map(row => row.map(val => val.toFixed(2))),
          texttemplate: '%{text}',
          textfont: { size: 12, color: 'black' }
        }]}
        layout={{
          title: 'Feature Correlation Heatmap',
          height: 400,
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
      />
    );
  };

  const LuxuryScoresChart = () => {
    if (!analysisData.luxuryScores) return <div className="plot-placeholder">Loading chart...</div>;

    return (
      <Plot
        data={[{
          type: 'bar',
          x: analysisData.luxuryScores.sectors,
          y: analysisData.luxuryScores.scores,
          marker: {
            color: analysisData.luxuryScores.scores,
            colorscale: 'Viridis'
          }
        }]}
        layout={{
          title: 'Top Sectors by Average Luxury Score',
          xaxis: { title: 'Sector', tickangle: -45 },
          yaxis: { title: 'Luxury Score' },
          height: 400
        }}
      />
    );
  };

  const PriceTrendChart = () => {
    if (!analysisData.priceTrend) return <div className="plot-placeholder">Loading chart...</div>;

    return (
      <Plot
        data={[{
          type: 'scatter',
          x: analysisData.priceTrend.age_categories,
          y: analysisData.priceTrend.avg_prices,
          mode: 'lines+markers',
          line: { shape: 'spline', width: 3 },
          marker: { size: 8 }
        }]}
        layout={{
          title: 'Price Trend by Property Age',
          xaxis: { title: 'Property Age Category' },
          yaxis: { title: 'Average Price (Cr)' },
          height: 400
        }}
      />
    );
  };

  return (
    <section id="analysis" className="analysis-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">📊 Market Analysis</h2>
          <p className="section-subtitle">Explore market trends and property insights with interactive visualizations</p>
        </div>

        <div className="analysis-container">
          {/* Geomap */}
          <div className="analysis-item">
            <h3>Sector Price per Sqft Geomap</h3>
            <div className="plot-container">
              <GeomapChart />
            </div>
          </div>

          {/* Wordcloud */}
          <div className="analysis-item">
            <h3>Features Wordcloud</h3>
            <div className="plot-container">
              <WordcloudDisplay />
            </div>
          </div>

          {/* Area vs Price */}
          <div className="analysis-item">
            <h3>Area vs Price</h3>
            <select 
              className="analysis-select"
              value={selectedPropertyType}
              onChange={(e) => setSelectedPropertyType(e.target.value)}
            >
              <option value="flat">Flat</option>
              <option value="house">House</option>
            </select>
            <div className="plot-container">
              <AreaPriceChart />
            </div>
          </div>

          {/* BHK Distribution */}
          <div className="analysis-item">
            <h3>BHK Distribution</h3>
            <select 
              className="analysis-select"
              value={selectedSector}
              onChange={(e) => setSelectedSector(e.target.value)}
            >
              <option value="overall">Overall</option>
              <option value="sector 45">Sector 45</option>
              <option value="sector 46">Sector 46</option>
              <option value="sector 47">Sector 47</option>
            </select>
            <div className="plot-container">
              <BhkPieChart />
            </div>
          </div>

          {/* Price Distribution */}
          <div className="analysis-item">
            <h3>Price Distribution by Property Type</h3>
            <div className="plot-container">
              <PriceDistributionChart />
            </div>
          </div>

          {/* Correlation Heatmap */}
          <div className="analysis-item">
            <h3>Feature Correlation Heatmap</h3>
            <div className="plot-container">
              <CorrelationHeatmap />
            </div>
          </div>

          {/* Luxury Scores */}
          <div className="analysis-item">
            <h3>Top Sectors by Luxury Score</h3>
            <div className="plot-container">
              <LuxuryScoresChart />
            </div>
          </div>

          {/* Price Trend */}
          <div className="analysis-item">
            <h3>Price Trend by Property Age</h3>
            <div className="plot-container">
              <PriceTrendChart />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Analysis;
