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
    }

    setupEventListeners() {
        // Navigation
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (navToggle) {
            navToggle.addEventListener('click', () => {
                navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
            });
        }

        // Prediction form
        const form = document.getElementById('prediction-form');
        if (form) {
            form.addEventListener('submit', (e) => this.handlePrediction(e));
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
        // Intersection Observer for scroll animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.feature-card, .tech-item, .about-content').forEach(el => {
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
            bedrooms: parseFloat(formData.get('bedrooms')),
            bathroom: parseFloat(formData.get('bathroom')),
            balcony: formData.get('balcony'),
            property_age: formData.get('property_age'),
            built_up_area: parseFloat(formData.get('built_up_area')),
            servant_room: parseFloat(formData.get('servant_room')),
            store_room: parseFloat(formData.get('store_room')),
            furnishing_type: formData.get('furnishing_type'),
            luxury_category: formData.get('luxury_category'),
            floor_category: formData.get('floor_category')
        };

        // Validate required fields
        if (Object.values(inputData).some(value => !value)) {
            this.showNotification('Please fill in all fields', 'error');
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

        // Add celebration effect
        this.celebrateSuccess();
    }

    setLoading(loading) {
        this.isLoading = loading;
        const button = document.querySelector('.predict-btn');
        const loader = document.querySelector('.btn-loader');
        const buttonText = document.querySelector('.btn-text');
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
        // Simple celebration effect
        const resultContainer = document.getElementById('result');
        if (resultContainer) {
            resultContainer.style.animation = 'none';
            setTimeout(() => {
                resultContainer.style.animation = 'pulse 0.6s ease-in-out';
            }, 10);
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : 'exclamation-triangle'}"></i>
            <span>${message}</span>
        `;

        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: ${type === 'success' ? '#10b981' : '#ef4444'};
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

        // Remove after 4 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }, 4000);

        // Add keyframe animations
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
            `;
            document.head.appendChild(style);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RealEstatePortfolio();
});

// Add CSS for notifications
const notificationStyles = `
.notification {
    transition: all 0.3s ease;
}
`;
