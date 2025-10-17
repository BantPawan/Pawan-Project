class RealEstatePortfolio {
    constructor() {
        // Use relative path for production
        this.apiBaseUrl = window.location.origin;
        this.isLoading = false;
        this.init();
    }

    async init() {
        try {
            this.setupEventListeners();
            this.setupSmoothScrolling();
            this.setupAnimations();
            
            // Load data sequentially with proper error handling
            await Promise.allSettled([
                this.loadOptions(),
                this.loadStats(),
                this.loadAnalysisOptions(),
                this.loadRecommenderOptions()
            ]);
            
            // Load visualizations with fallback handling
            await this.loadAnalysisData();
            await this.loadDynamicCharts();
            await this.loadAdvancedVisualizations();
            await this.loadInteractiveVisualizations();
            
            window.realEstateApp = this;
            this.initEnhancedFeatures();
            
            this.showNotification('Real Estate AI Portfolio loaded successfully! 🎉', 'success');
        } catch (error) {
            console.error('Initialization error:', error);
            this.showNotification('App loaded with some features unavailable', 'warning');
        }
    }

    // 🔥 ENHANCED: Load Dynamic Charts with Comprehensive Error Handling
    async loadDynamicCharts() {
        try {
            this.showNotification('Loading dynamic charts...', 'info');
            
            await Promise.allSettled([
                this.loadAreaVsPriceChart().catch(e => {
                    console.error('Area vs Price Chart error:', e);
                    this.showPlotError('areaChart', 'Area vs Price chart failed to load');
                }),
                this.loadPropertyTypeAnalysis().catch(e => {
                    console.error('Property Type Analysis error:', e);
                    this.showPlotError('typeChart', 'Property Type analysis failed to load');
                }),
                this.loadSunburstChart().catch(e => {
                    console.error('Sunburst Chart error:', e);
                    this.showPlotError('sunburstChart', 'Sunburst chart failed to load');
                })
            ]);
            
            this.showNotification('Dynamic charts loaded successfully! 🎉', 'success');
        } catch (error) {
            console.error('Error loading dynamic charts:', error);
            this.showNotification('Some charts failed to load. Please try refreshing.', 'warning');
        }
    }

    // 🔥 1. Area vs Price - Dynamic Updates
    async loadAreaVsPriceChart() {
        try {
            const propertyTypeSelect = document.getElementById('property-type-analysis');
            const selectedType = propertyTypeSelect ? propertyTypeSelect.value || 'all' : 'all';
            
            const response = await fetch(`${this.apiBaseUrl}/api/charts/area-vs-price?property_type=${encodeURIComponent(selectedType)}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}: Failed to load area vs price chart`);
            
            const data = await response.json();
            const container = document.getElementById('areaChart');
            if (!container) throw new Error('Chart container not found');
            
            this.setContainerLoading(container, false);
            container.innerHTML = '';
            
            // Validate chart data
            if (!data.chart?.data || !data.chart?.layout) {
                throw new Error('Invalid chart data structure');
            }

            await Plotly.newPlot('areaChart', data.chart.data, data.chart.layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            });
            
            container.classList.add('plot-success');
            this.addChartActions(container);
            
        } catch (error) {
            console.error('Error loading area vs price chart:', error);
            this.showPlotError('areaChart', `Area vs Price analysis error: ${error.message}`);
        }
    }

    // 🔥 2. Property Type Analysis
    async loadPropertyTypeAnalysis() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/charts/property-type-analysis`);
            if (!response.ok) throw new Error(`HTTP ${response.status}: Failed to load property type analysis`);
            
            const data = await response.json();
            const container = document.getElementById('typeChart');
            if (!container) throw new Error('Chart container not found');
            
            this.setContainerLoading(container, false);
            container.innerHTML = '';
            
            if (!data.chart?.data || !data.chart?.layout) {
                throw new Error('Invalid chart data structure');
            }

            await Plotly.newPlot('typeChart', data.chart.data, data.chart.layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            });
            
            container.classList.add('plot-success');
            this.addChartActions(container);
            
        } catch (error) {
            console.error('Error loading property type analysis:', error);
            this.showPlotError('typeChart', `Property Type analysis error: ${error.message}`);
        }
    }

    // 🔥 3. Enhanced Sunburst Chart
    async loadSunburstChart() {
        try {
            const propertyFilterSelect = document.getElementById('sunburst-property-type');
            const propertyFilter = propertyFilterSelect ? propertyFilterSelect.value || 'all' : 'all';
            
            const response = await fetch(`${this.apiBaseUrl}/api/charts/sunburst?property_filter=${encodeURIComponent(propertyFilter)}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}: Failed to load sunburst chart`);
            
            const data = await response.json();
            const container = document.getElementById('sunburstChart');
            if (!container) throw new Error('Sunburst chart container not found');
            
            this.setContainerLoading(container, false);
            container.innerHTML = '';
            
            if (!data.chart?.data || !data.chart?.layout) {
                throw new Error('Invalid sunburst data structure');
            }

            await Plotly.newPlot('sunburstChart', data.chart.data, data.chart.layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            });
            
            container.classList.add('plot-success');
            this.addChartActions(container);
            
        } catch (error) {
            console.error('Error loading sunburst chart:', error);
            this.showPlotError('sunburstChart', `Sunburst chart error: ${error.message}`);
            await this.loadFallbackSunburst();
        }
    }

    // 🔥 Fallback Sunburst Implementation
    async loadFallbackSunburst() {
        try {
            const container = document.getElementById('sunburstChart');
            if (!container) return;
            
            const fallbackData = [{
                type: "sunburst",
                labels: ["All Properties", "Flats", "Houses", "2 BHK", "3 BHK", "4 BHK", "1 BHK", "5+ BHK"],
                parents: ["", "All Properties", "All Properties", "Flats", "Flats", "Flats", "Houses", "Houses"],
                values: [1000, 600, 400, 250, 200, 150, 180, 220],
                branchvalues: "total",
                marker: {
                    colors: ['#6366f1', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#ef4444']
                },
                textinfo: 'label+value',
                hovertemplate: '%{label}<br>Properties: %{value}<extra></extra>'
            }];

            const layout = {
                title: "Property Hierarchy (Fallback Data)",
                height: 500,
                margin: { t: 40, b: 20, l: 20, r: 20 }
            };

            await Plotly.newPlot('sunburstChart', fallbackData, layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            });
            
            container.classList.add('plot-success');
            this.addChartActions(container);
            
        } catch (fallbackError) {
            console.error('Fallback sunburst failed:', fallbackError);
            this.showPlotError('sunburstChart', 'Unable to display sunburst visualization');
        }
    }

    // 🔥 Enhanced Error Display
    showPlotError(plotId, message = 'Chart failed to load') {
        const plot = document.getElementById(plotId);
        if (!plot) return;
        
        this.setContainerLoading(plot, false);
        plot.innerHTML = `
            <div class="plot-error">
                <i class="fas fa-chart-line fa-3x" style="color: #ef4444; margin-bottom: 1rem;"></i>
                <h4>Visualization Unavailable</h4>
                <p style="color: #6b7280; margin-bottom: 1rem;">${message}</p>
                <button onclick="window.realEstateApp?.retryLoad('${plotId}')" 
                        class="btn btn-outline" 
                        style="margin-top: 1rem; padding: 0.75rem 1.5rem;">
                    <i class="fas fa-redo"></i> Retry Loading
                </button>
                <button onclick="window.realEstateApp?.retryLoad('${plotId}', 999)" 
                        class="btn btn-secondary" 
                        style="margin-left: 0.5rem; padding: 0.75rem 1.5rem;">
                    <i class="fas fa-database"></i> Show Sample Data
                </button>
            </div>
        `;
    }

    // 🔥 Retry with Exponential Backoff
    async retryLoad(plotId, retryCount = 0, forceFallback = false) {
        const maxRetries = 3;
        
        if (retryCount >= maxRetries && !forceFallback) {
            this.showNotification('Maximum retry attempts reached. Showing sample data...', 'warning');
            return this.retryLoad(plotId, retryCount, true);
        }
        
        const delay = 1000 * Math.pow(2, Math.min(retryCount, 3));
        this.showNotification(`Loading ${plotId.replace(/([A-Z])/g, ' $1').trim()}... (${retryCount + 1})`, 'info');
        
        await new Promise(resolve => setTimeout(resolve, delay));
        
        try {
            const loadMethods = {
                'sunburstChart': () => this.loadSunburstChart(),
                'areaChart': () => this.loadAreaVsPriceChart(),
                'typeChart': () => this.loadPropertyTypeAnalysis(),
                'geomap': () => this.loadGeomap(),
                'wordcloud': () => this.loadWordcloud(),
                'bhk-pie': () => this.loadBhkPie(),
                'price-dist': () => this.loadPriceDistribution(),
                'correlation-heatmap': () => this.loadCorrelationHeatmap()
            };
            
            const method = loadMethods[plotId] || (() => Promise.reject(new Error(`Unknown plot: ${plotId}`)));
            await method();
            
            this.showNotification(`${plotId.replace(/-/g, ' ')} reloaded successfully!`, 'success');
        } catch (error) {
            console.error(`Retry ${retryCount + 1} failed for ${plotId}:`, error);
            if (forceFallback) {
                this.showPlotError(plotId, 'Sample data unavailable');
            } else {
                await this.retryLoad(plotId, retryCount + 1);
            }
        }
    }

    // 🔥 Utility Methods
    setContainerLoading(container, loading = true) {
        if (!container) return;
        if (loading) {
            container.classList.add('loading');
        } else {
            container.classList.remove('loading');
        }
    }

    addChartActions(container) {
        if (container.querySelector('.chart-actions')) return;
        
        const actions = document.createElement('div');
        actions.className = 'chart-actions';
        actions.innerHTML = `
            <button class="chart-action-btn" onclick="window.realEstateApp.downloadChart('${container.id}')" title="Download PNG">
                <i class="fas fa-download"></i> Export
            </button>
            <button class="chart-action-btn" onclick="window.realEstateApp.refreshChart('${container.id}')" title="Refresh Data">
                <i class="fas fa-sync-alt"></i> Refresh
            </button>
            <button class="chart-action-btn fullscreen-btn" onclick="window.realEstateApp.fullscreenChart('${container.id}')" title="Fullscreen">
                <i class="fas fa-expand"></i>
            </button>
        `;
        container.parentNode.insertBefore(actions, container.nextSibling);
    }

    async downloadChart(chartId) {
        try {
            const chart = document.getElementById(chartId);
            if (chart && Plotly) {
                await Plotly.downloadImage(chart, {
                    format: 'png',
                    width: 1200,
                    height: 800,
                    filename: `${chartId}-${Date.now()}`
                });
                this.showNotification('Chart downloaded successfully!', 'success');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification('Download failed. Please try again.', 'error');
        }
    }

    async refreshChart(chartId) {
        await this.retryLoad(chartId);
        this.showNotification('Chart refresh initiated!', 'info');
    }

    async fullscreenChart(chartId) {
        const chart = document.getElementById(chartId);
        if (!chart) return;
        
        try {
            if (chart.requestFullscreen) {
                await chart.requestFullscreen();
            } else if (chart.webkitRequestFullscreen) {
                await chart.webkitRequestFullscreen();
            } else {
                this.showNotification('Fullscreen not supported in this browser', 'warning');
                return;
            }
            this.showNotification('Press ESC to exit fullscreen', 'info');
        } catch (error) {
            console.error('Fullscreen error:', error);
            this.showNotification('Fullscreen request failed', 'error');
        }
    }

    // 🔥 Enhanced Options Loading
    async loadAnalysisOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/analysis-options`);
            if (!response.ok) throw new Error(`HTTP ${response.status}: Failed to load analysis options`);
            
            const options = await response.json();
            
            // Populate dropdowns with validation
            const dropdowns = {
                'property-type-analysis': options.property_types || [],
                'sector-analysis': ['overall', ...(options.sectors || [])],
                'sunburst-property-type': ['all', ...(options.property_types || [])]
            };
            
            Object.entries(dropdowns).forEach(([id, values]) => {
                if (Array.isArray(values) && values.length > 0) {
                    this.populateSelect(id, values);
                } else {
                    console.warn(`No options for ${id}, using fallback`);
                    this.populateSelect(id, id.includes('property') ? ['all', 'flat', 'house'] : ['overall']);
                }
            });
            
            // Setup event listeners after population
            this.setupAnalysisEventListeners();
            
        } catch (error) {
            console.error('Error loading analysis options:', error);
            this.populateFallbackAnalysisOptions();
        }
    }

    setupAnalysisEventListeners() {
        const selectors = {
            'property-type-analysis': () => this.loadAreaVsPriceChart(),
            'sector-analysis': () => this.loadBhkPie(),
            'sunburst-property-type': () => this.loadSunburstChart()
        };
        
        Object.entries(selectors).forEach(([selectorId, callback]) => {
            const select = document.getElementById(selectorId);
            if (select) {
                // Remove existing listeners to prevent duplicates
                select.replaceWith(select.cloneNode(true));
                const newSelect = document.getElementById(selectorId);
                newSelect.addEventListener('change', this.debounce(callback, 300));
            }
        });
    }

    populateFallbackAnalysisOptions() {
        this.populateSelect('property-type-analysis', ['all', 'flat', 'house']);
        this.populateSelect('sector-analysis', ['overall']);
        this.populateSelect('sunburst-property-type', ['all', 'flat', 'house']);
    }

    // 🔥 Utility: Debounce Function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // 🔥 Enhanced Form Options Loading
    async loadOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/options`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const options = await response.json();
            
            if (!options || typeof options !== 'object') {
                throw new Error('Invalid options format');
            }
            
            const dropdowns = {
                'property_type': options.property_type || [],
                'sector': options.sector || [],
                'bedrooms': options.bedrooms || [],
                'bathroom': options.bathroom || [],
                'balcony': options.balcony || [],
                'property_age': options.property_age || [],
                'servant_room': options.servant_room || [],
                'store_room': options.store_room || [],
                'furnishing_type': options.furnishing_type || [],
                'luxury_category': options.luxury_category || [],
                'floor_category': options.floor_category || []
            };

            Object.entries(dropdowns).forEach(([id, values]) => {
                if (Array.isArray(values) && values.length > 0) {
                    this.populateSelect(id, values);
                } else {
                    const fallback = this.getFallbackOptions(id);
                    this.populateSelect(id, fallback);
                    console.warn(`Using fallback options for ${id}`);
                }
            });
            
        } catch (error) {
            console.error('Error loading options:', error);
            this.showNotification('Using default form options', 'warning');
            this.loadFallbackOptions();
        }
    }

    getFallbackOptions(field) {
        const fallbacks = {
            'property_type': ['Flat', 'House'],
            'sector': ['Sector 1', 'Sector 2', 'Sector 45', 'Sector 46', 'Sector 47'],
            'bedrooms': [1, 2, 3, 4, 5],
            'bathroom': [1, 2, 3, 4],
            'balcony': ['0', '1', '2', '3', '3+'],
            'property_age': ['New Property', 'Relatively New', 'Moderately Old', 'Old Property'],
            'servant_room': [0, 1],
            'store_room': [0, 1],
            'furnishing_type': ['Unfurnished', 'Semi-Furnished', 'Furnished'],
            'luxury_category': ['Low', 'Medium', 'High', 'Ultra'],
            'floor_category': ['Low Floor', 'Mid Floor', 'High Floor']
        };
        return fallbacks[field] || ['Option 1', 'Option 2'];
    }

    loadFallbackOptions() {
        const fallbackOptions = {
            'property_type': ['Flat', 'House'],
            'sector': ['Sector 1', 'Sector 2', 'Sector 45', 'Sector 46', 'Sector 47'],
            'bedrooms': [1, 2, 3, 4, 5],
            'bathroom': [1, 2, 3, 4],
            'balcony': ['0', '1', '2', '3', '3+'],
            'property_age': ['New Property', 'Relatively New', 'Moderately Old', 'Old Property'],
            'servant_room': [0, 1],
            'store_room': [0, 1],
            'furnishing_type': ['Unfurnished', 'Semi-Furnished', 'Furnished'],
            'luxury_category': ['Low', 'Medium', 'High', 'Ultra'],
            'floor_category': ['Low Floor', 'Mid Floor', 'High Floor']
        };

        Object.entries(fallbackOptions).forEach(([id, values]) => {
            this.populateSelect(id, values);
        });
    }

    // 🔥 Event Listeners Setup
    setupEventListeners() {
        // Navigation
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
            });
        }

        // Form submission
        const form = document.getElementById('prediction-form');
        if (form) {
            form.addEventListener('submit', (e) => this.handlePrediction(e));
        }

        // Location and recommendation buttons
        const locationSearchBtn = document.getElementById('location-search-btn');
        const recommendBtn = document.getElementById('recommend-btn');
        
        if (locationSearchBtn) {
            locationSearchBtn.addEventListener('click', () => this.handleLocationSearch());
        }
        
        if (recommendBtn) {
            recommendBtn.addEventListener('click', () => this.handleRecommendation());
        }

        // Navigation links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                e.currentTarget.classList.add('active');
                navMenu?.classList.remove('active'); // Close mobile menu
            });
        });

        // Tab functionality
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        // Global scroll listener
        window.addEventListener('scroll', this.handleScroll.bind(this), { passive: true });
        
        // Window resize for responsive charts
        window.addEventListener('resize', this.debounce(() => {
            Plotly.Plots.resize(document.querySelectorAll('.plotly'));
        }, 250));
    }

    handleScroll() {
        const navbar = document.querySelector('.navbar');
        if (!navbar) return;
        
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    }

    // 🔥 Tab Switching with Loading States
    switchTab(tabName) {
        this.trackEvent('Tab Switch', { tab: tabName });
        
        // Update active tab
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            btn.style.transform = 'translateY(0)';
        });
        
        const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
            activeBtn.style.transform = 'translateY(-2px)';
        }

        // Switch content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
            content.style.opacity = '0';
            content.style.transform = 'translateY(20px)';
        });
        
        const activeContent = document.getElementById(`${tabName}-tab`);
        if (activeContent) {
            setTimeout(() => {
                activeContent.classList.add('active');
                activeContent.style.opacity = '1';
                activeContent.style.transform = 'translateY(0)';
            }, 150);
        }

        // Load tab-specific content
        setTimeout(() => {
            switch(tabName) {
                case 'advanced':
                    this.loadAdvancedVisualizations();
                    break;
                case 'interactive':
                    this.loadInteractiveVisualizations();
                    break;
                case 'dynamic':
                    this.loadDynamicCharts();
                    break;
            }
        }, 300);
    }

    // 🔥 Enhanced Features Initialization
    initEnhancedFeatures() {
        this.initScrollAnimations();
        this.addGlobalChartActions();
        this.setupKeyboardShortcuts();
    }

    addGlobalChartActions() {
        // Add actions to existing charts
        document.querySelectorAll('.plot-container').forEach(container => {
            this.addChartActions(container);
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F11') return; // Don't interfere with browser fullscreen
            
            // Ctrl/Cmd + R for refresh all charts
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                this.refreshAllCharts();
            }
            
            // Escape to exit fullscreen
            if (e.key === 'Escape') {
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                }
            }
        });
    }

    async refreshAllCharts() {
        this.showNotification('Refreshing all charts...', 'info');
        await Promise.allSettled([
            this.loadAreaVsPriceChart(),
            this.loadPropertyTypeAnalysis(),
            this.loadSunburstChart(),
            this.loadGeomap(),
            this.loadBhkPie(),
            this.loadPriceDistribution(),
            this.loadCorrelationHeatmap()
        ]);
        this.showNotification('All charts refreshed!', 'success');
    }

    initScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target); // Animate only once
                }
            });
        }, observerOptions);

        document.querySelectorAll('.analysis-item, .feature-card, .tech-item, .recommend-card').forEach(el => {
            observer.observe(el);
        });
    }

    // 🔥 Existing Methods (Enhanced)
    populateSelect(selectId, options) {
        const select = document.getElementById(selectId);
        if (!select) {
            console.warn(`Select element not found: ${selectId}`);
            return;
        }

        select.innerHTML = '<option value="">Select...</option>';
        
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = typeof option === 'object' ? option.label || option : option;
            select.appendChild(optionElement);
        });
        
        // Trigger change event for initial load
        select.dispatchEvent(new Event('change', { bubbles: true }));
    }

    async loadAdvancedVisualizations() {
        try {
            const vizIframes = {
                'luxury-viz': 'luxury_bar.html',
                '3d-viz': '3d_scatter.html',
                'trend-viz': 'price_trend.html',
                'age-viz': 'price_age_line.html'
            };

            for (const [iframeId, vizFile] of Object.entries(vizIframes)) {
                const iframe = document.getElementById(iframeId);
                if (iframe) {
                    iframe.src = `${this.apiBaseUrl}/viz/${vizFile}?t=${Date.now()}`;
                    iframe.onload = () => console.log(`${iframeId} loaded`);
                    iframe.onerror = () => {
                        console.error(`Failed to load ${vizFile}`);
                        iframe.style.display = 'none';
                    };
                }
            }
        } catch (error) {
            console.error('Error loading advanced visualizations:', error);
        }
    }

    async loadInteractiveVisualizations() {
        // Implementation similar to advanced visualizations
        await this.loadAdvancedVisualizations();
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
        // Legacy method - now handled by initScrollAnimations
        console.log('Animations initialized via IntersectionObserver');
    }

    // 🔥 Prediction and Form Handling
    async handlePrediction(e) {
        e.preventDefault();
        if (this.isLoading) return;

        const formData = new FormData(e.target);
        const inputData = Object.fromEntries(formData).reduce((acc, [key, value]) => {
            if (['bedrooms', 'bathroom', 'built_up_area', 'servant_room', 'store_room'].includes(key)) {
                acc[key] = parseFloat(value) || 0;
            } else {
                acc[key] = value || '';
            }
            return acc;
        }, {});

        // Enhanced validation
        const errors = this.validatePredictionInput(inputData);
        if (errors.length > 0) {
            this.showNotification(`Please fix: ${errors.join(', ')}`, 'error');
            return;
        }

        await this.predictPrice(inputData);
    }

    validatePredictionInput(data) {
        const required = ['property_type', 'sector', 'bedrooms', 'bathroom', 'built_up_area'];
        const errors = [];
        
        required.forEach(field => {
            if (!data[field] || data[field] === '') {
                errors.push(field.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()));
            }
        });
        
        if (data.built_up_area <= 0) {
            errors.push('built_up_area');
        }
        
        return errors;
    }

    async predictPrice(inputData) {
        this.setLoading(true);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/predict_price`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            this.displayResult(result);
            this.trackEvent('Price Prediction', { 
                property_type: inputData.property_type,
                sector: inputData.sector 
            });
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification(error.message || 'Prediction failed', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    displayResult(result) {
        const elements = {
            'result': document.getElementById('result'),
            'price-range': document.getElementById('price-range'),
            'low-price': document.getElementById('low-price'),
            'high-price': document.getElementById('high-price'),
            'mid-price': document.getElementById('mid-price')
        };

        Object.entries(elements).forEach(([key, el]) => {
            if (el && result[key.replace('-', '_')]) {
                el.textContent = result[key.replace('-', '_')];
            }
        });

        if (elements.result) {
            elements.result.style.display = 'block';
            elements.result.scrollIntoView({ behavior: 'smooth', block: 'center' });
            this.celebrateSuccess();
        }
    }

    setLoading(loading, buttonId = 'predict-btn') {
        this.isLoading = loading;
        const button = document.getElementById(buttonId);
        if (!button) return;

        const loader = button.querySelector('.btn-loader');
        const text = button.querySelector('.btn-text') || button;

        if (loading) {
            button.disabled = true;
            button.style.opacity = '0.7';
            if (loader) loader.style.display = 'inline-block';
            if (text) text.style.display = 'none';
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        } else {
            button.disabled = false;
            button.style.opacity = '1';
            if (loader) loader.style.display = 'none';
            button.innerHTML = 'Predict Price';
        }
    }

    celebrateSuccess() {
        const result = document.getElementById('result');
        if (result) {
            result.style.animation = 'pulse 1s ease-in-out';
            setTimeout(() => {
                result.style.animation = 'none';
            }, 1000);
        }
    }

    // 🔥 Analysis Visualizations (Enhanced)
    async loadGeomap() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/geomap`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            const container = document.getElementById('geomap');
            if (!container) throw new Error('Geomap container missing');
            
            this.setContainerLoading(container, false);
            container.innerHTML = '';

            // Validate data
            if (!data.latitude?.length || !data.longitude?.length) {
                throw new Error('Invalid geographic data');
            }

            const trace = {
                type: 'scattermapbox',
                lat: data.latitude,
                lon: data.longitude,
                mode: 'markers',
                marker: {
                    size: data.built_up_area?.map(a => Math.max(5, Math.min(30, Math.sqrt(a) / 10))) || 10,
                    color: data.price_per_sqft || '#6366f1',
                    colorscale: 'Viridis',
                    colorbar: { title: 'Price/Sqft' },
                    opacity: 0.7
                },
                text: data.sectors?.map((s, i) => `${s}<br>₹${data.price_per_sqft?.[i] || 'N/A'}/sqft`) || [],
                hoverinfo: 'text'
            };

            const layout = {
                mapbox: {
                    style: 'open-street-map',
                    zoom: 11,
                    center: {
                        lat: data.latitude.reduce((a, b) => a + b, 0) / data.latitude.length,
                        lon: data.longitude.reduce((a, b) => a + b, 0) / data.longitude.length
                    }
                },
                height: 500,
                margin: { t: 0, b: 0, l: 0, r: 0 }
            };

            await Plotly.newPlot('geomap', [trace], layout, { 
                responsive: true, 
                mapboxAccessToken: null 
            });
            container.classList.add('plot-success');
            
        } catch (error) {
            console.error('Geomap error:', error);
            this.showPlotError('geomap', error.message);
        }
    }

    async loadWordcloud() {
        try {
            const container = document.getElementById('wordcloud');
            if (!container) throw new Error('Wordcloud container missing');
            
            this.setContainerLoading(container, false);
            container.innerHTML = '<div class="loading">Generating wordcloud...</div>';
            
            const img = document.createElement('img');
            img.src = `${this.apiBaseUrl}/api/analysis/wordcloud?t=${Date.now()}`;
            img.alt = 'Property Features Word Cloud';
            img.className = 'wordcloud-img';
            img.onerror = () => this.showPlaceholderWordcloud(container);
            img.onload = () => {
                container.classList.add('plot-success');
                container.innerHTML = '';
                container.appendChild(img);
            };
            
            container.appendChild(img);
            
        } catch (error) {
            console.error('Wordcloud error:', error);
            this.showPlaceholderWordcloud(document.getElementById('wordcloud'));
        }
    }

    showPlaceholderWordcloud(container) {
        if (!container) return;
        container.innerHTML = `
            <div class="plot-placeholder">
                <i class="fas fa-cloud-word"></i>
                <h4>Word Cloud</h4>
                <p>Feature importance visualization temporarily unavailable</p>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center;">
                    ${['Luxury', 'Spacious', 'Modern', 'Balcony', 'Furnished', 'Prime Location']
                        .map(word => `<span style="background: var(--gradient); color: white; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.8rem;">${word}</span>`).join('')}
                </div>
            </div>
        `;
    }

    // 🔥 Other analysis methods remain similar but with enhanced error handling
    async loadBhkPie() {
        try {
            const sector = document.getElementById('sector-analysis')?.value || 'overall';
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/bhk-pie?sector=${encodeURIComponent(sector)}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            const container = document.getElementById('bhk-pie');
            if (!container) throw new Error('BHK pie container missing');
            
            this.setContainerLoading(container, false);
            
            const trace = {
                type: 'pie',
                labels: data.bedrooms?.map(b => `${b} BHK`) || [],
                values: data.counts || [],
                textinfo: 'percent+label',
                hoverinfo: 'label+percent+value',
                hole: 0.4,
                marker: { colors: ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'] }
            };

            await Plotly.newPlot('bhk-pie', [trace], {
                title: `BHK Distribution - ${sector === 'overall' ? 'All Sectors' : sector}`,
                height: 400
            }, { responsive: true });
            
            container.classList.add('plot-success');
        } catch (error) {
            console.error('BHK Pie error:', error);
            this.showPlotError('bhk-pie', error.message);
        }
    }

    async loadPriceDistribution() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/price-dist`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            const container = document.getElementById('price-dist');
            if (!container) throw new Error('Price dist container missing');
            
            this.setContainerLoading(container, false);
            
            const traces = [];
            if (data.house_prices?.length) {
                traces.push({
                    type: 'histogram',
                    x: data.house_prices,
                    name: 'Houses',
                    opacity: 0.7,
                    marker: { color: '#3b82f6' }
                });
            }
            if (data.flat_prices?.length) {
                traces.push({
                    type: 'histogram',
                    x: data.flat_prices,
                    name: 'Flats',
                    opacity: 0.7,
                    marker: { color: '#10b981' }
                });
            }

            await Plotly.newPlot('price-dist', traces, {
                title: 'Price Distribution by Property Type',
                xaxis: { title: 'Price (₹ Cr)' },
                yaxis: { title: 'Count' },
                barmode: 'overlay'
            }, { responsive: true });
            
            container.classList.add('plot-success');
        } catch (error) {
            console.error('Price distribution error:', error);
            this.showPlotError('price-dist', error.message);
        }
    }

    async loadCorrelationHeatmap() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analysis/correlation-data`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            const container = document.getElementById('correlation-heatmap');
            if (!container) throw new Error('Heatmap container missing');
            
            this.setContainerLoading(container, false);
            
            const trace = {
                type: 'heatmap',
                z: data.correlation_matrix || [],
                x: data.columns || [],
                y: data.columns || [],
                colorscale: 'RdBu',
                zmin: -1,
                zmax: 1
            };

            await Plotly.newPlot('correlation-heatmap', [trace], {
                title: 'Feature Correlation Matrix',
                height: 500
            }, { responsive: true });
            
            container.classList.add('plot-success');
        } catch (error) {
            console.error('Correlation heatmap error:', error);
            this.showPlotError('correlation-heatmap', error.message);
        }
    }

    // 🔥 Recommender Features
    async loadRecommenderOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/recommender/options`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const options = await response.json();
            this.populateSelect('location-search', options.locations || []);
            this.populateSelect('apartment-select', options.apartments || []);
            this.populateSelect('sector-analysis', ['overall', ...(options.sectors || [])]);
        } catch (error) {
            console.error('Recommender options error:', error);
            this.showNotification('Using default recommendation options', 'warning');
        }
    }

    async handleLocationSearch() {
        if (this.isLoading) return;
        this.setLoading(true, 'location-search-btn');

        try {
            const location = document.getElementById('location-search')?.value;
            const radius = parseFloat(document.getElementById('radius')?.value) || 10;
            
            if (!location) {
                this.showNotification('Please select a location', 'error');
                return;
            }

            const response = await fetch(
                `${this.apiBaseUrl}/api/recommender/location-search?location=${encodeURIComponent(location)}&radius=${radius}`
            );
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const results = await response.json();
            
            const container = document.getElementById('location-results');
            if (container) {
                if (results?.length > 0) {
                    container.innerHTML = results.map(r => `
                        <div class="recommend-card">
                            <strong>${r.property || 'Property'}</strong>
                            <span style="color: var(--gray);">Distance: ${r.distance || 'N/A'} km</span>
                            <div style="margin-top: 0.5rem;">
                                <span class="recommend-score">${(r.score * 100 || 0).toFixed(1)}%</span>
                            </div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<p class="plot-placeholder">No properties found nearby</p>';
                }
                container.style.display = 'block';
                container.scrollIntoView({ behavior: 'smooth' });
            }
            
            this.showNotification(`Found ${results?.length || 0} properties`, 'success');
        } catch (error) {
            console.error('Location search error:', error);
            this.showNotification(error.message, 'error');
        } finally {
            this.setLoading(false, 'location-search-btn');
        }
    }

    async handleRecommendation() {
        if (this.isLoading) return;
        this.setLoading(true, 'recommend-btn');

        try {
            const apartment = document.getElementById('apartment-select')?.value;
            const topN = parseInt(document.getElementById('top-n')?.value) || 5;
            
            if (!apartment) {
                this.showNotification('Please select an apartment', 'error');
                return;
            }

            const response = await fetch(
                `${this.apiBaseUrl}/api/recommender/recommend?apartment=${encodeURIComponent(apartment)}&top_n=${topN}`
            );
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const results = await response.json();
            
            const container = document.getElementById('recommend-results');
            if (container) {
                if (results?.length > 0) {
                    container.innerHTML = `
                        <div style="margin-bottom: 1rem;">
                            <h4>Top ${topN} Recommendations</h4>
                        </div>
                        <table class="recommend-table">
                            <thead>
                                <tr><th>Property</th><th>Similarity</th><th>Score</th></tr>
                            </thead>
                            <tbody>
                                ${results.map(r => `
                                    <tr>
                                        <td>${r.PropertyName || r.property || 'N/A'}</td>
                                        <td>${r.SimilarityScore ? (r.SimilarityScore * 100).toFixed(1) + '%' : 'N/A'}</td>
                                        <td><span class="recommend-score">${(r.score * 100 || 0).toFixed(1)}%</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                } else {
                    container.innerHTML = '<p class="plot-placeholder">No recommendations available</p>';
                }
                container.style.display = 'block';
                container.scrollIntoView({ behavior: 'smooth' });
            }
            
            this.showNotification(`Generated ${results?.length || 0} recommendations`, 'success');
        } catch (error) {
            console.error('Recommendation error:', error);
            this.showNotification(error.message, 'error');
        } finally {
            this.setLoading(false, 'recommend-btn');
        }
    }

    // 🔥 Stats and Tracking
    async loadStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/stats`);
            if (response.ok) {
                const stats = await response.json();
                this.updateStats(stats);
            }
        } catch (error) {
            console.error('Stats error:', error);
            this.updateStats({ total_properties: 0, model_accuracy: 'N/A', sectors_covered: 0 });
        }
    }

    updateStats(stats) {
        const mappings = {
            'stat-properties': stats.total_properties,
            'stat-accuracy': stats.model_accuracy,
            'stat-sectors': stats.sectors_covered
        };
        
        Object.entries(mappings).forEach(([id, value]) => {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = typeof value === 'number' ? value.toLocaleString() : value;
            }
        });
    }

    trackEvent(eventName, properties = {}) {
        // Enhanced tracking with timestamp
        console.log(`[Tracking] ${eventName}`, { 
            ...properties, 
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent.slice(0, 100)
        });
        
        // Could integrate with analytics services here
        if (window.gtag) {
            window.gtag('event', eventName.toLowerCase().replace(/\s+/g, '_'), properties);
        }
    }

    // 🔥 Loading and Analysis Data
    async loadAnalysisData() {
        this.removePlotLoadingStates();
        
        await Promise.allSettled([
            this.loadGeomap(),
            this.loadWordcloud(),
            this.loadBhkPie(),
            this.loadPriceDistribution(),
            this.loadCorrelationHeatmap()
        ]);
    }

    removePlotLoadingStates() {
        ['geomap', 'wordcloud', 'bhk-pie', 'price-dist', 'correlation-heatmap', 
         'areaChart', 'typeChart', 'sunburstChart'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('loading');
        });
    }

    // 🔥 Notification System
    showNotification(message, type = 'info') {
        // Remove existing notifications
        document.querySelectorAll('.notification').forEach(n => {
            n.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => n.remove(), 300);
        });
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${{
                success: 'check-circle',
                error: 'exclamation-triangle',
                warning: 'exclamation-circle',
                info: 'info-circle'
            }[type] || 'info-circle'}"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.remove()" style="margin-left: auto; background: none; border: none; color: inherit; cursor: pointer; font-size: 1.2rem;">×</button>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// 🔥 Enhanced DOM Ready Handler
document.addEventListener('DOMContentLoaded', () => {
    try {
        // Initialize app
        window.realEstateApp = new RealEstatePortfolio();
        
        // Global error handling
        window.addEventListener('error', (event) => {
            if (event.error && !event.error.message?.includes('willReadFrequently')) {
                console.error('Global error:', event.error);
                window.realEstateApp?.showNotification('An error occurred', 'error');
            }
        });
        
        window.addEventListener('unhandledrejection', (event) => {
            const reason = event.reason;
            if (reason && !reason.message?.includes('willReadFrequently')) {
                console.error('Unhandled promise rejection:', reason);
                window.realEstateApp?.showNotification('Network operation failed', 'warning');
            }
            event.preventDefault();
        });
        
        // Service Worker registration (optional)
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').catch(console.error);
        }
        
    } catch (initError) {
        console.error('Critical initialization failure:', initError);
        
        const errorOverlay = document.createElement('div');
        errorOverlay.style.cssText = `
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8); z-index: 99999;
            display: flex; align-items: center; justify-content: center;
            color: white; flex-direction: column; padding: 2rem;
        `;
        errorOverlay.innerHTML = `
            <h2>⚠️ Application Error</h2>
            <p>Failed to initialize. Please refresh the page.</p>
            <button onclick="location.reload()" 
                    style="background: #6366f1; color: white; border: none; 
                           padding: 1rem 2rem; border-radius: 8px; 
                           font-size: 1.1rem; cursor: pointer; margin-top: 1rem;">
                🔄 Refresh Page
            </button>
        `;
        document.body.appendChild(errorOverlay);
    }
});

// 🔥 Inject Essential Styles
const essentialStyles = document.createElement('style');
essentialStyles.textContent = `
    .notification { 
        position: fixed; top: 20px; right: 20px; 
        background: white; color: #374151;
        padding: 1rem 1.5rem; border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        display: flex; align-items: center; gap: 0.75rem;
        z-index: 10000; max-width: 400px;
        border-left: 4px solid #3b82f6;
    }
    .notification.success { border-left-color: #10b981; }
    .notification.error { border-left-color: #ef4444; }
    .notification.warning { border-left-color: #f59e0b; }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .plot-error, .plot-placeholder {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 300px; text-align: center; padding: 2rem;
        color: #6b7280; background: #f8fafc;
    }
    .plot-error { background: #fef2f2; color: #dc2626; }
    
    .chart-actions {
        display: flex; gap: 0.5rem; margin-top: 1rem;
        justify-content: center; flex-wrap: wrap;
    }
    .chart-action-btn {
        padding: 0.5rem 1rem; border: 1px solid #e5e7eb;
        background: white; border-radius: 6px; cursor: pointer;
        display: flex; align-items: center; gap: 0.25rem;
        font-size: 0.875rem; transition: all 0.2s;
    }
    .chart-action-btn:hover {
        background: #6366f1; color: white; border-color: #6366f1;
        transform: translateY(-1px);
    }
    
    .recommend-table {
        width: 100%; border-collapse: collapse; margin-top: 1rem;
        background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .recommend-table th { 
        background: #f8fafc; padding: 1rem; text-align: left;
        font-weight: 600; color: #374151; border-bottom: 2px solid #e5e7eb;
    }
    .recommend-table td { padding: 0.75rem 1rem; border-bottom: 1px solid #f3f4f6; }
    .recommend-table tr:hover { background: #f9fafb; }
    
    .recommend-score {
        font-weight: 700; color: #10b981; font-size: 1.1rem;
    }
    
    .loading { 
        display: flex; align-items: center; justify-content: center;
        height: 200px; color: #6b7280;
    }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }
    
    /* Fade-in animation */
    .fade-in {
        animation: fadeIn 0.8s ease-out forwards;
        opacity: 0; transform: translateY(30px);
    }
    @keyframes fadeIn {
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(essentialStyles);

// 🔥 Global Functions for Button onclick Handlers
window.retryLoadChart = (chartId) => window.realEstateApp?.retryLoad(chartId);
window.refreshAll = () => window.realEstateApp?.refreshAllCharts();
window.exportData = () => window.realEstateApp?.showNotification('Export feature coming soon!', 'info');