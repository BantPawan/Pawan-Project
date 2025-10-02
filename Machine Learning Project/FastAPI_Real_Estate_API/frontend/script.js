class RealEstatePortfolio {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.isLoading = false;
        this.currentFilter = 'all';
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
        
        window.realEstateApp = this;
        console.log('Real Estate Portfolio App initialized');
    }

    setupEventListeners() {
        // Navigation
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (navToggle) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
                navToggle.classList.toggle('active');
            });
        }

        // Prediction form
        const form = document.getElementById('prediction-form');
        if (form) {
            form.addEventListener('submit', (e) => this.handlePrediction(e));
        }

        // Analysis filters
        const analysisFilters = document.querySelectorAll('.analysis-filter');
        analysisFilters.forEach(filter => {
            filter.addEventListener('click', (e) => {
                analysisFilters.forEach(f => f.classList.remove('active'));
                e.target.classList.add('active');
                this.currentFilter = e.target.dataset.filter;
                this.filterAnalysisItems(this.currentFilter);
            });
        });

        // Analysis controls
        const propertyTypeSelect = document.getElementById('property-type-analysis');
        if (propertyTypeSelect) {
            propertyTypeSelect.addEventListener('change', () => this.loadAreaVsPrice());
        }

        const sectorSelect = document.getElementById('sector-analysis');
        if (sectorSelect) {
            sectorSelect.addEventListener('change', () => this.loadBhkPie());
        }

        // Recommender
        const locationSearchBtn = document.getElementById('location-search-btn');
        if (locationSearchBtn) {
            locationSearchBtn.addEventListener('click', () => this.handleLocationSearch());
        }

        const recommendBtn = document.getElementById('recommend-btn');
        if (recommendBtn) {
            recommendBtn.addEventListener('click', () => this.handleRecommendation());
        }

        // Radius slider
        const radiusSlider = document.getElementById('radius');
        if (radiusSlider) {
            radiusSlider.addEventListener('input', (e) => {
                document.getElementById('radius-value').textContent = `${e.target.value} km`;
            });
        }

        // Navigation links
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                navLinks.forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
                
                // Close mobile menu
                if (window.innerWidth <= 768) {
                    navMenu.classList.remove('active');
                    navToggle.classList.remove('active');
                }
            });
        });

        // Scroll handling for navbar
        window.addEventListener('scroll', this.handleScroll.bind(this));

        // Error handling for images
        document.addEventListener('error', (e) => {
            if (e.target.tagName === 'IMG') {
                console.warn('Image failed to load:', e.target.src);
                e.target.style.display = 'none';
            }
        }, true);
    }

    handleScroll() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Update active nav link based on scroll position
        this.updateActiveNavLink();
    }

    updateActiveNavLink() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.clientHeight;
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    }

    setupSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    const offsetTop = target.offsetTop - 80;
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
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
                    
                    // Load data for analysis items when they come into view
                    if (entry.target.classList.contains('analysis-item')) {
                        const plotId = entry.target.querySelector('.plot-container')?.id;
                        if (plotId && !this.isPlotLoaded(plotId)) {
                            this.lazyLoadPlot(plotId);
                        }
                    }
                }
            });
        }, { 
            threshold: 0.1,
            rootMargin: '50px'
        });

        document.querySelectorAll('.feature-card, .tech-item, .about-content, .analysis-item, .recommender-card').forEach(el => {
            observer.observe(el);
        });
    }

    isPlotLoaded(plotId) {
        const plot = document.getElementById(plotId);
        return plot && !plot.classList.contains('loading') && plot.querySelector('.js-plotly-plot');
    }

    lazyLoadPlot(plotId) {
        // Implement lazy loading for plots if needed
        console.log('Lazy loading plot:', plotId);
    }

    filterAnalysisItems(filter) {
        const items = document.querySelectorAll('.analysis-item');
        items.forEach(item => {
            if (filter === 'all' || item.dataset.type === filter) {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        });
    }

    async loadOptions() {
        try {
            this.showNotification('Loading form options...', 'info');
            const response = await fetch(`${this.apiBaseUrl}/api/options`);
            if (!response.ok) throw new Error(`Failed to load options: ${response.status}`);
            
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
            
            this.showNotification('Form options loaded successfully!', 'success');
        } catch (error) {
            console.error('Error loading options:', error);
            this.showNotification('Failed to load form options. Using default values.', 'error');
            this.populateDefaultOptions();
        }
    }

    populateDefaultOptions() {
        const defaultOptions = {
            property_type: ['flat', 'house', 'villa'],
            sector: ['Sector 45', 'Sector 46', 'Sector 47', 'Sector 48', 'Sector 49'],
            bedrooms: [1, 2, 3, 4, 5],
            bathroom: [1, 2, 3, 4],
            balcony: ['0', '1', '2', '3', '3+'],
            property_age: ['New Property', 'Relatively New', 'Moderately Old', 'Old Property'],
            servant_room: [0, 1],
            store_room: [0, 1],
            furnishing_type: ['unfurnished', 'semifurnished', 'furnished'],
            luxury_category: ['Low', 'Medium', 'High'],
            floor_category: ['Low Floor', 'Mid Floor', 'High Floor']
        };

        Object.keys(defaultOptions).forEach(key => {
            this.populateSelect(key, defaultOptions[key]);
        });
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/stats`);
            if (response.ok) {
                const stats = await response.json();
                this.updateStats(stats);
            } else {
                throw new Error(`Failed to load stats: ${response.status}`);
            }
        } catch (error) {
            console.error('Error loading stats:', error);
            this.showNotification('Failed to load statistics. Using fallback data.', 'warning');
        }
    }

    async loadAnalysisData() {
        try {
            this.removePlotLoadingStates();
            
            // Load all visualizations
            await Promise.allSettled([
                this.loadGeomap(),
                this.loadWordcloud(),
                this.loadAreaVsPrice(),
                this.loadBhkPie(),
                this.loadPriceDistribution(),
                this.loadCorrelationHeatmap(),
                this.loadLuxuryScores(),
                this.loadPriceTrend(),
                this.loadPropertyTypes()
            ]);
            
            this.showNotification('All visualizations loaded successfully!', 'success');
        } catch (error) {
            console.error('Error loading analysis data:', error);
            this.showNotification('Some visualizations failed to load. Please try refreshing.', 'warning');
        }
    }

    async loadRecommenderOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/recommender/options`);
            if (!response.ok) throw new Error(`Failed to load recommender options: ${response.status}`);
            const options = await response.json();
            
            this.populateSelect('location-search', options.locations);
            this.populateSelect('apartment-select', options.apartments);
            
            // Populate sector analysis dropdown
            const sectorSelect = document.getElementById('sector-analysis');
            if (sectorSelect && options.sectors) {
                options.sectors.forEach(sector => {
                    const option = document.createElement('option');
                    option.value = sector;
                    option.textContent = sector;
                    sectorSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading recommender options:', error);
            this.showNotification('Failed to load recommender options. Using sample data.', 'error');
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

        // Clear existing options except the first one
        while (select.options.length > 1) {
            select.remove(1);
        }
        
        if (Array.isArray(options)) {
            options.forEach(option => {
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = option;
                select.appendChild(optionElement);
            });
        }
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
        const requiredFields = ['property_type', 'sector', 'bedrooms', 'bathroom', 'balcony', 
                               'property_age', 'built_up_area', 'servant_room', 'store_room', 
                               'furnishing_type', 'luxury_category', 'floor_category'];
        
        const missingFields = requiredFields.filter(field => {
            const value = inputData[field];
            const numericFields = ['servant_room', 'store_room', 'bedrooms', 'bathroom', 'built_up_area'];
            
            if (numericFields.includes(field)) {
                return value === '' || value === null || value === undefined || isNaN(value);
            }
            
            return !value || value === '';
        });

        if (missingFields.length > 0) {
            this.showNotification(`Please fill in all required fields: ${missingFields.join(', ')}`, 'error');
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
            if (!response.ok) throw new Error(`Failed to load geomap data: ${response.status}`);
            const data = await response.json();
            
            const geomapContainer = document.getElementById('geomap');
            geomapContainer.classList.remove('loading');
            
            if (!data.latitude || !data.longitude || data.latitude.length === 0) {
                throw new Error('Invalid or empty geomap data');
            }

            const markerSizes = data.built_up_area.map(area => {
                const size = Math.sqrt(area) / 5;
                return Math.max(8, Math.min(25, size));
            });

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
                    `${sector}<br>Price: ₹${data.price_per_sqft[i]?.toLocaleString() || 'N/A'}/sqft<br>Area: ${data.built_up_area[i] || 'N/A'} sqft<br>Properties: ${data.property_count?.[i] || 'N/A'}`
                ),
                hoverinfo: 'text',
                name: 'Sectors'
            };

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

            Plotly.newPlot('geomap', [trace], layout, { 
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            });
            
        } catch (error) {
            console.error('Error loading geomap:', error);
            this.showPlotError('geomap');
        }
    }

    async loadWordcloud() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/wordcloud`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            const wordcloudDiv = document.getElementById('wordcloud');
            wordcloudDiv.classList.remove('loading');
            
            if (data.image_url) {
                const imgContainer = document.createElement('div');
                imgContainer.style.cssText = 'text-align: center; padding: 10px;';
                
                const img = document.createElement('img');
                img.src = data.image_url;
                img.alt = data.message || 'Property Features Wordcloud';
                img.style.cssText = 'max-width: 100%; height: auto; border-radius: 8px; opacity: 0; transition: opacity 0.5s ease-in;';
                
                img.onload = () => {
                    img.style.opacity = '1';
                };
                
                img.onerror = () => {
                    console.error('Failed to load wordcloud image');
                    this.showPlaceholderWordcloud(wordcloudDiv, 'Image failed to load');
                };
                
                imgContainer.appendChild(img);
                wordcloudDiv.innerHTML = '';
                wordcloudDiv.appendChild(imgContainer);
                
                if (data.message) {
                    const messageDiv = document.createElement('div');
                    messageDiv.style.cssText = 'text-align: center; color: #6b7280; font-size: 0.9rem; margin-top: 10px;';
                    messageDiv.textContent = data.message;
                    wordcloudDiv.appendChild(messageDiv);
                }
            } else {
                throw new Error('No image_url in wordcloud response');
            }
        } catch (error) {
            console.error('Error loading wordcloud:', error);
            this.showPlaceholderWordcloud(document.getElementById('wordcloud'), 'Wordcloud temporarily unavailable');
        }
    }

    showPlaceholderWordcloud(container, message) {
        if (!container) return;
        
        container.classList.remove('loading');
        container.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 2rem; text-align: center; background: #f8fafc; border-radius: 10px;">
                <i class="fas fa-cloud" style="font-size: 4rem; color: #d1d5db; margin-bottom: 1rem;"></i>
                <h3 style="color: #6b7280; margin-bottom: 0.5rem;">Feature Overview</h3>
                <p style="color: #9ca3af; margin-bottom: 1.5rem;">${message}</p>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; max-width: 400px;">
                    <span style="background: #e0e7ff; color: #3730a3; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Apartment</span>
                    <span style="background: #f0f9ff; color: #0c4a6e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Luxury</span>
                    <span style="background: #fef2f2; color: #991b1b; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Modern</span>
                    <span style="background: #f0fdf4; color: #166534; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Spacious</span>
                    <span style="background: #fff7ed; color: #9a3412; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Furnished</span>
                    <span style="background: #faf5ff; color: #7c3aed; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Balcony</span>
                    <span style="background: #f1f5f9; color: #475569; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Parking</span>
                    <span style="background: #fef7cd; color: #854d0e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">Security</span>
                </div>
            </div>
        `;
    }

    async loadAreaVsPrice() {
        try {
            const propertyType = document.getElementById('property-type-analysis').value;
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/area-vs-price?property_type=${encodeURIComponent(propertyType)}`);
            if (!response.ok) throw new Error(`Failed to load area vs price data: ${response.status}`);
            const data = await response.json();
            
            const container = document.getElementById('area-vs-price');
            container.classList.remove('loading');
            
            if (!data.built_up_area || !data.price || data.built_up_area.length === 0) {
                throw new Error('Invalid or empty area vs price data');
            }

            Plotly.newPlot('area-vs-price', [{
                type: 'scatter',
                x: data.built_up_area,
                y: data.price,
                mode: 'markers',
                marker: {
                    color: data.bedrooms,
                    size: 8,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {
                        title: 'BHK'
                    }
                },
                text: data.bedrooms?.map(b => `${b} BHK`),
                hoverinfo: 'x+y+text'
            }], {
                title: `Area vs Price - ${propertyType.charAt(0).toUpperCase() + propertyType.slice(1)}s`,
                xaxis: { title: 'Built-up Area (sq ft)' },
                yaxis: { title: 'Price (₹ Cr)' },
                height: 400,
                showlegend: false
            }, {
                responsive: true,
                displayModeBar: true
            });
        } catch (error) {
            console.error('Error loading area vs price:', error);
            this.showPlotError('area-vs-price');
        }
    }

    async loadBhkPie() {
        try {
            const sector = document.getElementById('sector-analysis').value;
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/bhk-pie?sector=${encodeURIComponent(sector)}`);
            if (!response.ok) throw new Error(`Failed to load BHK pie data: ${response.status}`);
            const data = await response.json();
            
            const container = document.getElementById('bhk-pie');
            container.classList.remove('loading');
            
            if (!data.bedrooms || !data.counts || data.bedrooms.length === 0) {
                throw new Error('Invalid or empty BHK pie data');
            }

            Plotly.newPlot('bhk-pie', [{
                type: 'pie',
                labels: data.bedrooms.map(b => `${b} BHK`),
                values: data.counts,
                textinfo: 'percent+label',
                hoverinfo: 'label+percent+value',
                hole: 0.4,
                marker: {
                    colors: ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899']
                }
            }], {
                title: `BHK Distribution - ${sector === 'overall' ? 'All Sectors' : sector}`,
                height: 400,
                showlegend: true
            }, {
                responsive: true
            });
        } catch (error) {
            console.error('Error loading BHK pie:', error);
            this.showPlotError('bhk-pie');
        }
    }

    async loadPriceDistribution() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/price-dist`);
            if (!response.ok) throw new Error(`Failed to load price distribution data: ${response.status}`);
            const data = await response.json();
            
            const container = document.getElementById('price-dist');
            container.classList.remove('loading');
            
            if (!data.house_prices && !data.flat_prices) {
                throw new Error('Invalid or empty price distribution data');
            }

            Plotly.newPlot('price-dist', [
                {
                    type: 'histogram',
                    x: data.house_prices,
                    name: 'Houses',
                    opacity: 0.7,
                    nbinsx: 20,
                    marker: {
                        color: '#8b5cf6'
                    }
                },
                {
                    type: 'histogram',
                    x: data.flat_prices,
                    name: 'Flats',
                    opacity: 0.7,
                    nbinsx: 20,
                    marker: {
                        color: '#6366f1'
                    }
                }
            ], {
                title: 'Price Distribution by Property Type',
                xaxis: { title: 'Price (₹ Cr)' },
                yaxis: { title: 'Count' },
                barmode: 'overlay',
                height: 400
            }, {
                responsive: true
            });
        } catch (error) {
            console.error('Error loading price distribution:', error);
            this.showPlotError('price-dist');
        }
    }

    // NEW VISUALIZATION METHODS
    async loadCorrelationHeatmap() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/correlation`);
            if (!response.ok) throw new Error(`Failed to load correlation data: ${response.status}`);
            const data = await response.json();
            
            const container = document.getElementById('correlation-heatmap');
            container.classList.remove('loading');

            Plotly.newPlot('correlation-heatmap', [{
                z: data.correlation_matrix,
                x: data.features,
                y: data.features,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmin: -1,
                zmax: 1,
                hoverongaps: false,
                text: data.correlation_matrix.map(row => row.map(val => val.toFixed(2))),
                texttemplate: '%{text}',
                textfont: { size: 12, color: 'black' }
            }], {
                title: 'Feature Correlation Matrix',
                height: 400,
                margin: { t: 50, b: 50, l: 50, r: 50 }
            }, {
                responsive: true
            });
        } catch (error) {
            console.error('Error loading correlation heatmap:', error);
            this.showPlotError('correlation-heatmap');
        }
    }

    async loadLuxuryScores() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/luxury-score`);
            if (!response.ok) throw new Error(`Failed to load luxury scores: ${response.status}`);
            const data = await response.json();
            
            const container = document.getElementById('luxury-scores');
            container.classList.remove('loading');

            Plotly.newPlot('luxury-scores', [{
                type: 'bar',
                x: data.sectors,
                y: data.scores,
                marker: {
                    color: data.scores,
                    colorscale: 'Viridis'
                }
            }], {
                title: 'Top Sectors by Average Luxury Score',
                xaxis: { 
                    title: 'Sector',
                    tickangle: -45
                },
                yaxis: { title: 'Luxury Score' },
                height: 400
            }, {
                responsive: true
            });
        } catch (error) {
            console.error('Error loading luxury scores:', error);
            this.showPlotError('luxury-scores');
        }
    }

    async loadPriceTrend() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/price-trend`);
            if (!response.ok) throw new Error(`Failed to load price trend: ${response.status}`);
            const data = await response.json();
            
            const container = document.getElementById('price-trend');
            container.classList.remove('loading');

            Plotly.newPlot('price-trend', [{
                type: 'scatter',
                x: data.age_categories,
                y: data.avg_prices,
                mode: 'lines+markers',
                line: { 
                    shape: 'spline', 
                    width: 3,
                    color: '#6366f1'
                },
                marker: { 
                    size: 8,
                    color: '#8b5cf6'
                }
            }], {
                title: 'Price Trend by Property Age',
                xaxis: { title: 'Property Age Category' },
                yaxis: { title: 'Average Price (₹ Cr)' },
                height: 400
            }, {
                responsive: true
            });
        } catch (error) {
            console.error('Error loading price trend:', error);
            this.showPlotError('price-trend');
        }
    }

    async loadPropertyTypes() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/property-types`);
            if (!response.ok) throw new Error(`Failed to load property types: ${response.status}`);
            const data = await response.json();
            
            const container = document.getElementById('property-types');
            container.classList.remove('loading');

            Plotly.newPlot('property-types', [{
                type: 'pie',
                labels: data.types,
                values: data.counts,
                textinfo: 'percent+label',
                hoverinfo: 'label+percent+value',
                marker: {
                    colors: ['#6366f1', '#8b5cf6', '#a855f7']
                }
            }], {
                title: 'Property Type Distribution',
                height: 400
            }, {
                responsive: true
            });
        } catch (error) {
            console.error('Error loading property types:', error);
            this.showPlotError('property-types');
        }
    }

    async handleLocationSearch() {
        if (this.isLoading) return;
        this.setLoading(true, 'location-search-btn');

        try {
            const location = document.getElementById('location-search').value;
            const radius = parseFloat(document.getElementById('radius').value);
            
            if (!location) {
                this.showNotification('Please select a location', 'error');
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/api/recommender/location-search?location=${encodeURIComponent(location)}&radius=${radius}`);
            if (!response.ok) throw new Error(`Failed to search locations: ${response.status}`);
            const results = await response.json();
            
            const resultContainer = document.getElementById('location-results');
            if (results.length > 0) {
                resultContainer.innerHTML = `
                    <h4>Properties within ${radius} km of ${location}:</h4>
                    <div class="results-list">
                        ${results.map(r => `
                            <div class="result-item">
                                <strong>${r.property}</strong>
                                <span class="distance">${r.distance} km away</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                resultContainer.innerHTML = '<p>No properties found within the specified radius. Try increasing the search radius.</p>';
            }
            resultContainer.style.display = 'block';
            
            this.showNotification(`Found ${results.length} properties near ${location}`, 'success');
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
            const topN = parseInt(document.getElementById('recommendation-count').value);
            
            if (!apartment) {
                this.showNotification('Please select an apartment', 'error');
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/api/recommender/recommend?property_name=${encodeURIComponent(apartment)}&top_n=${topN}`);
            if (!response.ok) throw new Error(`Failed to get recommendations: ${response.status}`);
            const results = await response.json();
            
            const resultContainer = document.getElementById('recommend-results');
            if (results.length > 0) {
                resultContainer.innerHTML = `
                    <h4>Similar properties to ${apartment}:</h4>
                    <table class="recommend-table">
                        <thead>
                            <tr>
                                <th>Property Name</th>
                                <th>Similarity Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${results.map(r => `
                                <tr>
                                    <td>${r.PropertyName}</td>
                                    <td>${r.SimilarityScore.toFixed(3)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            } else {
                resultContainer.innerHTML = '<p>No recommendations available for this property.</p>';
            }
            resultContainer.style.display = 'block';
            
            this.showNotification(`Generated ${results.length} recommendations`, 'success');
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
            if (button) button.disabled = true;
        } else {
            if (buttonText) buttonText.style.opacity = '1';
            if (loader) loader.style.display = 'none';
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            if (button) button.disabled = false;
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
        const plots = ['geomap', 'wordcloud', 'area-vs-price', 'bhk-pie', 'price-dist', 
                      'correlation-heatmap', 'luxury-scores', 'price-trend', 'property-types'];
        plots.forEach(plotId => {
            const plot = document.getElementById(plotId);
            if (plot) {
                plot.classList.remove('loading');
            }
        });
    }

    showPlotError(plotId) {
        const plot = document.getElementById(plotId);
        if (plot) {
            plot.classList.remove('loading');
            plot.innerHTML = `
                <div class="plot-error" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 2rem; text-align: center;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: #6b7280; margin-bottom: 1rem;"></i>
                    <p style="color: #6b7280; margin-bottom: 1rem;">Failed to load ${plotId.replace('-', ' ')}</p>
                    <button onclick="realEstateApp.retryLoad('${plotId}')" class="btn btn-outline" style="margin-top: 1rem;">
                        <i class="fas fa-redo"></i> Retry
                    </button>
                </div>
            `;
        }
    }

    retryLoad(plotId) {
        switch(plotId) {
            case 'wordcloud':
                this.loadWordcloud();
                break;
            case 'geomap':
                this.loadGeomap();
                break;
            case 'area-vs-price':
                this.loadAreaVsPrice();
                break;
            case 'bhk-pie':
                this.loadBhkPie();
                break;
            case 'price-dist':
                this.loadPriceDistribution();
                break;
            case 'correlation-heatmap':
                this.loadCorrelationHeatmap();
                break;
            case 'luxury-scores':
                this.loadLuxuryScores();
                break;
            case 'price-trend':
                this.loadPriceTrend();
                break;
            case 'property-types':
                this.loadPropertyTypes();
                break;
        }
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) return;

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;

        container.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RealEstatePortfolio();
});

// Export for global access
window.RealEstatePortfolio = RealEstatePortfolio;
