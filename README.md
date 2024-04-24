# Heart Rate Prediction from Running Metrics

Welcome to my GitHub repository where I present a machine-learning project focused on predicting heart rate from running data. The project includes data analysis, data preprocessing, feature engineering, and model development.

## Overview

This analysis is based on 31 of my running activities, recorded from September 2023 to March 2024.  The data includes heart rate, speed, elevation, and running power measurements collected with a Garmin Fenix 7 Pro. 
To ensure data quality, I manually labeled periods where the heart rate sensor readings were unreliable and used the other periods for training a machine learning model.

### Key Aspects

1. **Data Preprocessing**: Cleaned and imputed data, smoothed and computed features (speed, elevation gain, grade), and computed gradient-adjusted speed.
2. **Feature Engineering**: Explored different aggregation methods for gradient-adjusted speed and power.
3. **Models**: Utilized LightGBM and Ridge Regression for heart rate prediction.
4. **Results**: Achieved a Mean Absolute Error of approximately 6 beats per minute on validation running activities.

## Web App Deployment
For a hands-on experience with the heart rate predictions and to delve into more technical details, visit the interactive Streamlit web app [https://garmin-data-analysis-cytltvwaipppccwsskzvnb.streamlit.app/](https://garmin-data-analysis-cytltvwaipppccwsskzvnb.streamlit.app/). The app also offers a user-friendly interface for data labeling.

Feel free to reach out with any questions or feedback!
