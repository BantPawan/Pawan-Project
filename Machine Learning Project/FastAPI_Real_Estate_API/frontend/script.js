class RealEstatePortfolio {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.isLoading = false;
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupSmoothScrolling();
        this.setupAnimations();
        await this.loadOptions();
        await this.loadStats();
        await this.loadAnalysisData();
        await this.loadRecommenderOptions();
        
        // Make app globally available for retry functions
        window.realEstateApp = this;
    }

    setupEventListeners() {
        // Navigation
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (navToggle) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
            });
        }

        // Prediction form
        const form = document.getElementById('prediction-form');
        if (form) {
            form.addEventListener('submit', (e) => this.handlePrediction(e));
        }

        // Analysis controls
        const propertyTypeSelect = document.getElementById('property-type-analysis');
        if (propertyTypeSelect) {
            propertyTypeSelect.addEventListener('change', () => this.loadAreaVsPrice());
        }

        const sectorSelect = document.getElementById('sector-analysis');
        if (sectorSelect) {
            sectorSelect.addEventListener('change', () => this.loadBhkPie());
        }

        // Recommender controls
        const locationSearchBtn = document.getElementById('location-search-btn');
        if (locationSearchBtn) {
            locationSearchBtn.addEventListener('click', () => this.handleLocationSearch());
        }

        const recommendBtn = document.getElementById('recommend-btn');
        if (recommendBtn) {
            recommendBtn.addEventListener('click', () => this.handleRecommendation());
        }

        // Nav link active state
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                navLinks.forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
            });
        });

        // Window scroll for navbar effect
        window.addEventListener('scroll', this.handleScroll.bind(this));
    }

    handleScroll() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    }

    setupSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    setupAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.feature-card, .tech-item, .about-content, .analysis-item').forEach(el => {
            observer.observe(el);
        });
    }

    async loadOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/options`);
            if (!response.ok) throw new Error('Failed to load options');
            
            const options = await response.json();
            this.populateSelect('property_type', options.property_type);
            this.populateSelect('sector', options.sector);
            this.populateSelect('bedrooms', options.bedrooms);
            this.populateSelect('bathroom', options.bathroom);
            this.populateSelect('balcony', options.balcony);
            this.populateSelect('property_age', options.property_age);
            this.populateSelect('servant_room', options.servant_room);
            this.populateSelect('store_room', options.store_room);
            this.populateSelect('furnishing_type', options.furnishing_type);
            this.populateSelect('luxury_category', options.luxury_category);
            this.populateSelect('floor_category', options.floor_category);
            
        } catch (error) {
            console.error('Error loading options:', error);
            this.showNotification('Failed to load form options. Please refresh the page.', 'error');
        }
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/stats`);
            if (response.ok) {
                const stats = await response.json();
                this.updateStats(stats);
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    async loadAnalysisData() {
        try {
            // Remove loading states
            this.removePlotLoadingStates();
            
            // Load all analysis components in parallel with better error handling
            await Promise.allSettled([
                this.loadGeomap().catch(e => {
                    console.error('Geomap error:', e);
                    this.showPlotError('geomap');
                }),
                this.loadWordcloud().catch(e => {
                    console.error('Wordcloud error:', e);
                    this.showPlotError('wordcloud');
                }),
                this.loadAreaVsPrice().catch(e => console.error('Area vs Price error:', e)),
                this.loadBhkPie().catch(e => console.error('BHK Pie error:', e)),
                this.loadPriceDistribution().catch(e => console.error('Price Distribution error:', e))
            ]);
            
            this.showNotification('Analysis data loaded successfully!', 'success');
        } catch (error) {
            console.error('Error loading analysis data:', error);
            this.showNotification('Some analysis components failed to load. Please try refreshing.', 'warning');
        }
    }

    async loadRecommenderOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/recommender/options`);
            if (!response.ok) throw new Error('Failed to load recommender options');
            const options = await response.json();
            this.populateSelect('location-search', options.locations);
            this.populateSelect('apartment-select', options.apartments);
            this.populateSelect('sector-analysis', ['overall', ...options.sectors]);
        } catch (error) {
            console.error('Error loading recommender options:', error);
            this.showNotification('Failed to load recommender options. Please refresh the page.', 'error');
        }
    }

    updateStats(stats) {
        const propertiesEl = document.getElementById('stat-properties');
        const accuracyEl = document.getElementById('stat-accuracy');
        const sectorsEl = document.getElementById('stat-sectors');

        if (propertiesEl) propertiesEl.textContent = stats.total_properties.toLocaleString();
        if (accuracyEl) accuracyEl.textContent = stats.model_accuracy;
        if (sectorsEl) sectorsEl.textContent = stats.sectors_covered;
    }

    populateSelect(selectId, options) {
        const select = document.getElementById(selectId);
        if (!select) return;

        select.innerHTML = '<option value="">Select...</option>';
        
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            select.appendChild(optionElement);
        });
    }

    async handlePrediction(e) {
        e.preventDefault();
        if (this.isLoading) return;

        const formData = new FormData(e.target);
        const inputData = {
            property_type: formData.get('property_type'),
            sector: formData.get('sector'),
            bedrooms: parseFloat(formData.get('bedrooms')) || 0,
            bathroom: parseFloat(formData.get('bathroom')) || 0,
            balcony: formData.get('balcony') || '0',
            property_age: formData.get('property_age'),
            built_up_area: parseFloat(formData.get('built_up_area')) || 0,
            servant_room: parseFloat(formData.get('servant_room')) || 0,
            store_room: parseFloat(formData.get('store_room')) || 0,
            furnishing_type: formData.get('furnishing_type'),
            luxury_category: formData.get('luxury_category'),
            floor_category: formData.get('floor_category')
        };

        // Validate required fields
        const requiredFields = ['property_type', 'sector', 'bedrooms', 'bathroom', 'balcony', 'property_age', 'built_up_area', 'servant_room', 'store_room', 'furnishing_type', 'luxury_category', 'floor_category'];
        const missingFields = requiredFields.filter(field => !inputData[field] || inputData[field] === '');
        if (missingFields.length > 0) {
            this.showNotification(`Please fill in: ${missingFields.join(', ')}`, 'error');
            return;
        }

        await this.predictPrice(inputData);
    }

    async predictPrice(inputData) {
        this.setLoading(true);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/predict_price`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResult(result);
            this.showNotification('Price prediction generated successfully!', 'success');
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification(error.message || 'Failed to get prediction. Please try again.', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    async loadGeomap() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/geomap`);
            if (!response.ok) throw new Error('Failed to load geomap data');
            const data = await response.json();
            
            // Remove loading state
            const geomapContainer = document.getElementById('geomap');
            geomapContainer.classList.remove('loading');
            
            // Calculate marker sizes based on built_up_area
            const markerSizes = data.built_up_area.map(area => {
                const size = Math.sqrt(area) / 5;
                return Math.max(10, Math.min(30, size));
            });

            // Create the map trace
            const trace = {
                type: 'scattermapbox',
                lat: data.latitude,
                lon: data.longitude,
                mode: 'markers',
                marker: {
                    size: markerSizes,
                    color: data.price_per_sqft,
                    colorscale: 'Viridis',
                    colorbar: {
                        title: 'Price per Sqft',
                        titleside: 'right'
                    },
                    opacity: 0.8,
                    sizemode: 'diameter'
                },
                text: data.sectors.map((sector, i) => 
                    `${sector}<br>Price: ₹${data.price_per_sqft[i]?.toLocaleString() || 'N/A'}/sqft<br>Avg Area: ${data.built_up_area[i] || 'N/A'} sqft<br>Properties: ${data.property_count?.[i] || 'N/A'}`
                ),
                hoverinfo: 'text',
                name: 'Sectors'
            };

            // Calculate center of the map
            const centerLat = data.latitude.reduce((a, b) => a + b, 0) / data.latitude.length;
            const centerLon = data.longitude.reduce((a, b) => a + b, 0) / data.longitude.length;

            const layout = {
                mapbox: {
                    style: 'open-street-map',
                    center: { lat: centerLat, lon: centerLon },
                    zoom: 11
                },
                margin: { t: 0, b: 0, l: 0, r: 0 },
                height: 500,
                showlegend: false
            };

            Plotly.newPlot('geomap', [trace], layout, { responsive: true });
            
        } catch (error) {
            console.error('Error loading geomap:', error);
            this.showPlotError('geomap');
            
            // Create a fallback map with sample data
            const fallbackData = [{
                type: 'scattermapbox',
                lat: [28.4595, 28.4612, 28.4630],
                lon: [77.0266, 77.0280, 77.0300],
                mode: 'markers',
                marker: {
                    size: [20, 25, 18],
                    color: [8500, 9200, 7800],
                    colorscale: 'Viridis',
                    showscale: true
                },
                text: ['Sector 45: ₹8,500/sqft', 'Sector 46: ₹9,200/sqft', 'Sector 47: ₹7,800/sqft']
            }];

            const fallbackLayout = {
                mapbox: {
                    style: 'open-street-map',
                    center: { lat: 28.4595, lon: 77.0266 },
                    zoom: 11
                },
                margin: { t: 0, b: 0, l: 0, r: 0 },
                height: 500
            };

            Plotly.newPlot('geomap', fallbackData, fallbackLayout);
        }
    }

    async loadWordcloud() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/wordcloud`);
            if (!response.ok) throw new Error('Failed to load wordcloud data');
            const data = await response.json();
            
            // Remove loading state
            const wordcloudDiv = document.getElementById('wordcloud');
            wordcloudDiv.classList.remove('loading');
            
            if (data.image_url) {
                wordcloudDiv.innerHTML = `
                    <div style="text-align: center;">
                        <img src="${data.image_url}" 
                             style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" 
                             alt="Property Features Wordcloud"
                             onload="this.style.opacity='1'"
                             onerror="this.showPlotError('wordcloud')">
                    </div>
                `;
                
                // Add fade-in effect
                const img = wordcloudDiv.querySelector('img');
                if (img) {
                    img.style.transition = 'opacity 0.5s ease-in';
                    img.style.opacity = '0';
                    setTimeout(() => { 
                        if (img) img.style.opacity = '1'; 
                    }, 100);
                }
            }
        } catch (error) {
            console.error('Error loading wordcloud:', error);
            this.showPlotError('wordcloud');
        }
    }

    async loadAreaVsPrice() {
        try {
            const propertyType = document.getElementById('property-type-analysis').value;
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/area-vs-price?property_type=${propertyType}`);
            if (!response.ok) throw new Error('Failed to load area vs price data');
            const data = await response.json();
            
            Plotly.newPlot('area-vs-price', [{
                type: 'scatter',
                x: data.built_up_area,
                y: data.price,
                mode: 'markers',
                marker: {
                    color: data.bedrooms,
                    size: 10,
                    showscale: true
                },
                text: data.bedrooms,
                hoverinfo: 'x+y+text'
            }], {
                title: `Area vs Price for ${propertyType.charAt(0).toUpperCase() + propertyType.slice(1)}`,
                xaxis: { title: 'Built-up Area (sq ft)' },
                yaxis: { title: 'Price (Cr)' },
                height: 500
            });
        } catch (error) {
            console.error('Error loading area vs price:', error);
            this.showNotification('Failed to load area vs price plot. Please try again.', 'error');
        }
    }

    async loadBhkPie() {
        try {
            const sector = document.getElementById('sector-analysis').value;
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/bhk-pie?sector=${sector}`);
            if (!response.ok) throw new Error('Failed to load BHK pie data');
            const data = await response.json();
            
            Plotly.newPlot('bhk-pie', [{
                type: 'pie',
                labels: data.bedrooms.map(b => `${b} BHK`),
                values: data.counts,
                textinfo: 'percent+label'
            }], {
                title: `BHK Distribution in ${sector === 'overall' ? 'Overall Data' : sector}`,
                height: 500
            });
        } catch (error) {
            console.error('Error loading BHK pie:', error);
            this.showNotification('Failed to load BHK distribution. Please try again.', 'error');
        }
    }

    async loadPriceDistribution() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/price-dist`);
            if (!response.ok) throw new Error('Failed to load price distribution data');
            const data = await response.json();
            
            Plotly.newPlot('price-dist', [
                {
                    type: 'histogram',
                    x: data.house_prices,
                    name: 'House',
                    opacity: 0.5,
                    histnorm: 'density'
                },
                {
                    type: 'histogram',
                    x: data.flat_prices,
                    name: 'Flat',
                    opacity: 0.5,
                    histnorm: 'density'
                }
            ], {
                title: 'Price Distribution by Property Type',
                xaxis: { title: 'Price (Cr)' },
                yaxis: { title: 'Density' },
                barmode: 'overlay',
                height: 500
            });
        } catch (error) {
            console.error('Error loading price distribution:', error);
            this.showNotification('Failed to load price distribution. Please try again.', 'error');
        }
    }

    async handleLocationSearch() {
        if (this.isLoading) return;
        this.setLoading(true, 'location-search-btn');

        try {
            const location = document.getElementById('location-search').value;
            const radius = parseFloat(document.getElementById('radius').value);
            if (!location || isNaN(radius)) {
                this.showNotification('Please select a location and valid radius', 'error');
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/api/recommender/location-search?location=${encodeURIComponent(location)}&radius=${radius}`);
            if (!response.ok) throw new Error('Failed to search locations');
            const results = await response.json();
            
            const resultContainer = document.getElementById('location-results');
            resultContainer.innerHTML = results.length > 0 
                ? results.map(r => `<div class="result-item">${r.property} (${r.distance.toFixed(1)} kms)</div>`).join('')
                : '<p>No properties found within the specified radius.</p>';
            resultContainer.style.display = 'block';
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            this.showNotification('Location search completed successfully!', 'success');
        } catch (error) {
            console.error('Location search error:', error);
            this.showNotification(error.message || 'Failed to search locations. Please try again.', 'error');
        } finally {
            this.setLoading(false, 'location-search-btn');
        }
    }

    async handleRecommendation() {
        if (this.isLoading) return;
        this.setLoading(true, 'recommend-btn');

        try {
            const apartment = document.getElementById('apartment-select').value;
            if (!apartment) {
                this.showNotification('Please select an apartment', 'error');
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/api/recommender/recommend?property_name=${encodeURIComponent(apartment)}`);
            if (!response.ok) throw new Error('Failed to get recommendations');
            const results = await response.json();
            
            const resultContainer = document.getElementById('recommend-results');
            resultContainer.innerHTML = results.length > 0 
                ? `<table class="recommend-table">
                    <thead><tr><th>Property Name</th><th>Similarity Score</th></tr></thead>
                    <tbody>${results.map(r => `<tr><td>${r.PropertyName}</td><td>${r.SimilarityScore.toFixed(3)}</td></tr>`).join('')}</tbody>
                    </table>`
                : '<p>No recommendations available.</p>';
            resultContainer.style.display = 'block';
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            this.showNotification('Recommendations generated successfully!', 'success');
        } catch (error) {
            console.error('Recommendation error:', error);
            this.showNotification(error.message || 'Failed to get recommendations. Please try again.', 'error');
        } finally {
            this.setLoading(false, 'recommend-btn');
        }
    }

    displayResult(result) {
        const resultContainer = document.getElementById('result');
        const priceRange = document.getElementById('price-range');
        const lowPrice = document.getElementById('low-price');
        const highPrice = document.getElementById('high-price');
        const midPrice = document.getElementById('mid-price');

        if (priceRange) priceRange.textContent = result.formatted_range;
        if (lowPrice) lowPrice.textContent = `₹ ${result.low_price_cr} Cr`;
        if (highPrice) highPrice.textContent = `₹ ${result.high_price_cr} Cr`;
        if (midPrice) midPrice.textContent = `₹ ${result.prediction_raw.toFixed(2)} Cr`;

        if (resultContainer) {
            resultContainer.style.display = 'block';
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        this.celebrateSuccess();
    }

    setLoading(loading, buttonId = 'predict-btn') {
        this.isLoading = loading;
        const button = document.querySelector(`#${buttonId}`);
        const loader = button ? button.querySelector('.btn-loader') : null;
        const buttonText = button ? button.querySelector('.btn-text') : null;
        const loadingOverlay = document.getElementById('loading');

        if (loading) {
            if (buttonText) buttonText.style.opacity = '0.5';
            if (loader) loader.style.display = 'block';
            if (loadingOverlay) loadingOverlay.style.display = 'flex';
        } else {
            if (buttonText) buttonText.style.opacity = '1';
            if (loader) loader.style.display = 'none';
            if (loadingOverlay) loadingOverlay.style.display = 'none';
        }
    }

    celebrateSuccess() {
        const resultContainer = document.getElementById('result');
        if (resultContainer) {
            resultContainer.style.animation = 'none';
            setTimeout(() => {
                resultContainer.style.animation = 'pulse 0.6s ease-in-out';
            }, 10);
        }
    }

    removePlotLoadingStates() {
        const plots = ['geomap', 'wordcloud', 'area-vs-price', 'bhk-pie', 'price-dist'];
        plots.forEach(plotId => {
            const plot = document.getElementById(plotId);
            if (plot) {
                plot.classList.remove('loading');
                const errorDiv = plot.querySelector('.plot-error');
                if (errorDiv) errorDiv.style.display = 'none';
            }
        });
    }

    showPlotError(plotId) {
        const plot = document.getElementById(plotId);
        if (plot) {
            plot.classList.remove('loading');
            const errorDiv = plot.querySelector('.plot-error');
            if (errorDiv) errorDiv.style.display = 'block';
        }
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        document.querySelectorAll('.notification').forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;

        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            animation: slideInRight 0.3s ease-out;
            max-width: 400px;
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }, 4000);

        if (!document.querySelector('#notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOutRight {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.02); }
                    100% { transform: scale(1); }
                }
                .recommend-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 1rem;
                }
                .recommend-table th, .recommend-table td {
                    border: 1px solid var(--gray-light);
                    padding: 0.5rem;
                    text-align: left;
                }
                .recommend-table th {
                    background: var(--primary);
                    color: white;
                }
            `;
            document.head.appendChild(style);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new RealEstatePortfolio();
});
