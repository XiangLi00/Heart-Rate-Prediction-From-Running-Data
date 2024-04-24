# Heart Rate Prediction from Running Metrics

Welcome to my GitHub repository where I present a machine learning project focused on predicting heart rate my personal running data. The project includes data analysis, data preprocessing, feature engineering, and model development.

## Overview

I've analyzed 31 running activities recorded between September 2023 and March 2024. The data includes heart rate, speed, elevation, and running power measurements collected with a Garmin Fenix 7 Pro. 
Since the heart rate sensor can be very inaccurate at certain times, I also manually labeled the time ranges in which the heart rate measurements are trustworthy.

### Key Aspects

1. **Data Preprocessing**: Cleaned and imputed data, smoothed and computed features (speed, elevation gain, grade), and computed gradient-adjusted speed.
2. **Feature Engineering**: Explored different aggregation methods for gradient-adjusted speed and power.
3. **Models**: Utilized Ridge Regression and LightGBM for heart rate prediction.
4. **Results**: Achieved a MAE of approximately 6 beats per minute on validation running activities.

## Web App Deployment

Explore the interactive Streamlit web app at [https://garmin-data-analysis-cytltvwaipppccwsskzvnb.streamlit.app/](https://garmin-data-analysis-cytltvwaipppccwsskzvnb.streamlit.app/)
. The website visualizes the heart rate predictions, includes more detailed technical explanations, and provides a user interface for data labeling.
Feel free to reach out with any questions or feedback!
