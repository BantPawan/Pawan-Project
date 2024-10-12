# Real Estate Data Science Application

## Capstone Project Overview:
This capstone project centers on applying advanced data science techniques to generate comprehensive insights, accurate predictions, and personalized recommendations within the real estate sector. The project was organized through a structured approach, covering data preparation, modeling, and deployment of an interactive, user-friendly web application.

## Data Preparation and Cleaning:
- The project began with a thorough cleaning and consolidation of the dataset, which unified information on houses and flats.
- Handling missing values, resolving inconsistencies, and detecting outliers were key steps to ensure reliable and accurate analysis in subsequent phases.

## Feature Engineering:
- The dataset was enriched by engineering new features that captured intricate details about real estate properties.
- Additional room counts, property age, furnishing types, luxury scores, and area classifications were introduced to provide a more nuanced understanding of the data and to improve the predictive power of the models.

## Exploratory Data Analysis (EDA):
- Univariate and multivariate techniques were used in the EDA phase to reveal trends and relationships within the data.
- Tools like Pandas Profiling provided a deeper understanding of the dataset’s distribution and structure, aiding in the feature selection and model-building process.

## Handling Missing Data and Outliers:
- Targeted imputation techniques addressed missing values in key areas, such as property area and bedroom count.
- Outliers that could potentially impact model accuracy were identified and removed to ensure the models remained robust and reliable.

## Feature Selection:
- Various feature selection methods, including correlation analysis, random forest importance, gradient boosting, LASSO, and recursive feature elimination, were applied to identify the most impactful features.
- SHAP (SHapley Additive exPlanations) values provided interpretability by showing how each feature influenced the predictions.

## Model Selection and Deployment:
- Several regression models were evaluated for their ability to predict property prices, with each assessed based on accuracy, generalization capability, and computational efficiency.
  
  The models tested included:
  1. Linear Regression – Baseline model assuming linear relationships.
  2. Support Vector Regression (SVR) – Captures non-linear relationships in the data.
  3. Random Forest Regressor – An ensemble method that averages predictions from decision trees.
  4. Multi-layer Perceptron (MLP) – A neural network capable of learning complex patterns.
  5. LASSO Regression – Utilizes L1 regularization to enforce feature sparsity.
  6. Ridge Regression – Applies L2 regularization to handle multicollinearity.
  7. Gradient Boosting Regressor – Sequentially builds trees to improve model performance, an ensemble learning method that builds trees sequentially with each tree correcting the errors of the previous errors.
  8. Decision Tree Regressor – Non-linear model that splits data based on key features.
  9. K-Nearest Neighbors Regressor – Predicts the target variable by averaging the values of its k-nearest neighbors.
  10. ElasticNet Regression – Combines L1 and L2 regularization for optimal performance.

- After testing and assessing the performance of each model on relevant evaluation metrics, considering factors such as accuracy, precision, and recall. The most effective model was selected and integrated into a full property price prediction pipeline.
- The pipeline included preprocessing, feature encoding, and final predictions, and it was deployed via Streamlit to create a dynamic and user-friendly web interface to predict the prices.
- This allowed the end-user to make informed decisions in real estate.

## Building the Analytics Module:
- An analytics module was built to visually present insights, including geographical heatmaps, word clouds for amenities, scatter plots, box plots, and pie charts.
- These visualizations help users understand market trends and property features with greater clarity and comprehensive understanding of the market.

## Recommender System Development:
- A recommender system was designed to provide personalized property suggestions based on user preferences.
- This system focused on recommending properties based on top amenities, price details, and location advantages.
- A Streamlit-based interface allowed users to interact with the recommendation engine and make informed property decisions.

# Streamlit Application:

## Home Page:
![image](https://github.com/user-attachments/assets/b47077b2-ed75-441d-845a-8d6f510e8a9b)

## Price Predictor Page:
![image](https://github.com/user-attachments/assets/1a698567-52ab-40a6-9d52-d690b37bd94c)
![image](https://github.com/user-attachments/assets/8c632163-800b-432b-afc7-b64352bd75ac)
![image](https://github.com/user-attachments/assets/a46d2bc3-7dee-4fbf-be8e-043f73d1beba)

## Analysis Page:
![image](https://github.com/user-attachments/assets/b689a978-59cb-4d81-8220-aabae0d177e4)
![image](https://github.com/user-attachments/assets/28073241-b9a9-424e-987a-114bb2e9ca2f)
![image](https://github.com/user-attachments/assets/6c25d15a-5745-49a5-9967-655671aa3523)
![image](https://github.com/user-attachments/assets/c93bc597-9d73-4d96-8bf5-20840d9adbfe)
![image](https://github.com/user-attachments/assets/b3d4d007-d603-4b88-8b93-df2b8a753a3d)
![image](https://github.com/user-attachments/assets/24dcba30-1ce2-4fc5-95fa-39e0ec70cdf7)

## Recommend Apartments Page:
![image](https://github.com/user-attachments/assets/bc07fadc-b931-4a00-94fe-c16b09991cf1)
![image](https://github.com/user-attachments/assets/baba1b50-b4a7-4824-93b8-f47f1ab801d2)
