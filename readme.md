### Data Analysis & Machine Learning with Tabular and Time-Series Datasets

This repository demonstrates end-to-end machine learning workflows on two real-world datasets:

##### NYC Taxi Trip Data (tabular dataset)
##### Delhi Air Quality Data (time-series dataset)
Each workflow covers data preprocessing, EDA, feature engineering, clustering & anomaly detection, imputation, and model building using AutoML and advanced ML techniques.

#### Project 1: NYC Taxi Trip Data (Tabular Dataset)
##### Dataset
NYC Taxi Trip Records — containing trip details like timestamps, distances, fare amounts, and passenger counts.
##### Data Preprocessing & Cleaning
1) Handle missing values & duplicates
2) Convert timestamps to datetime objects
3) Normalize numerical features (distances, fares, durations)
##### Exploratory Data Analysis (EDA)
1) Distribution plots: trip duration, fare amount, distance
2) Correlation heatmaps & pairplots
3) Time-series visualizations for hourly/daily trip demand
##### Feature Engineering
1) Extract features: hour of day, day of week, month
2) Calculate geospatial distance features (Haversine, Manhattan distance)
3) Feature selection based on importance (RF/XGBoost)
##### Clustering & Anomaly Detection
1) K-Means / DBSCAN to group trips
2) Identify outliers in trip fare or duration
##### Data Imputation
1) Mean/median/mode imputation for missing values
2) Predictive imputation using regression models
##### Model Building (Fare Prediction)
1) Models: Random Forest, XGBoost, Auto-ViML
2) Evaluation Metrics: RMSE, MAE, R²
3) Ensemble methods for improved accuracy

#### Project 2: Delhi Air Quality Prediction (Time-Series Dataset)
##### Dataset
Delhi air pollution dataset with daily/hourly measurements of PM2.5 and other pollutants.
##### Data Preprocessing & Cleaning
1) Handle missing timestamps & values
2) Convert strings to datetime format
3) Ensure stationarity for time-series modeling
##### Exploratory Data Analysis (EDA)
1) Line plots of pollutant levels over time
2) Seasonal decomposition (trend, seasonality, residuals)
3) Anomaly detection in spikes of pollutants
##### Feature Engineering
1) Time-based features: day, month, week, season
2) Lag features & rolling averages
3) Pollution level categories (low/medium/high)
##### Clustering & Anomaly Detection
1) Isolation Forest & Z-score for outlier detection
2) DBSCAN to group temporal patterns
##### Data Imputation
1) Linear interpolation for continuous values
2) Predictive imputation with regression models
##### Model Building (Forecasting)
1) Forecast pollutant levels using:
- Prophet (trend/seasonality-based forecasting)
-  Auto-ViML for AutoML-based regression
-   Ensemble techniques (bagging & boosting)
2) Evaluation Metrics: RMSE, MAE, MAPE

#### Setup Environment
##### Clone repository
git clone https://github.com/yourusername/data-analysis-ml.git
cd data-analysis-ml

##### Create environment & install requirements
pip install -r requirements.txt

##### Run Jupyter Notebooks
Open any notebook from eda_visualization/ or modeling/ to explore results.
##### Results & Artifacts
1) NYC Taxi Project: Predictive model for trip fare amounts
2) Delhi Air Quality Project: Forecasting model for PM2.5 levels

#### Artifacts include:
1) Visualizations (heatmaps, plots, decompositions)
2) Trained ML models
3) Evaluation metrics (RMSE, MAE, R², etc.)
4) Resources


📘 Auto-ViML Documentation

📖 AutoGluon for Time-Series
