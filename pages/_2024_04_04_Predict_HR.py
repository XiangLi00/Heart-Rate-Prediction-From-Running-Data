from datetime import datetime, timedelta
import importlib
import math
import os
import sqlite3
import sys
import time

import bokeh
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import sklearn
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

sys.path.append(os.path.join(os.getcwd(), 'streamlit_pages'))  
# from streamlit_pages._2024_03_18_annotate_fit_helper_v2 import test1
from streamlit_pages import _2024_04_04_predict_hr_helper_v1
from utils import helper_pandas, helper_streamlit, helper_sklearn, helper_load_specific_df
# from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report

project_path = os.getcwd()


def do_stuff_on_page_load():
    # Make sure that page elements use the full page width
    st.set_page_config(layout="wide")
    
    # Retrieve screen heigh, width in pixels
    st.session_state.screen_height, st.session_state.screen_width = helper_streamlit.get_screen_height_and_width()  
do_stuff_on_page_load()


st.title("Machine Learning Project: Predicting heart rate from speed, elevation, and running power")

# Display explanation of this page
str_markdown_explanation = """
## Data
- 31 running activities by the author from Sep 2023 to Mar 2024 from the (only selected activities covering at least 5km)
- Data was recorded once per second with a Garmin Fenix 7 Pro. Measurements include heart rate, speed, elevation, and running power.

## Data labeling
- Data labeling: In some segments, the measured heart rate was completely unrealistic. Using expert judgement, these segments were manually labeled and not used for training (See page "Labeling inaccurate heart rate measurements")

## Data preprocessing
- Data was imputed and cleaned (e.g., unrealistic sudden elevation changes were ignored)
- Intermediate feature **"speed"**(for the previous second) is smoothed by **convolution with a Gaussian kernel** (std=3s) because the GPS data is noisy
- Similarly, the intermediate feature **"elevation_change"** (within the previous second) is smoothed (std=2s). Reason: The watch rounds the elevation to multiples of 20cm, which is too coarse when trying to compute the current grade/slope of the surface.
- The **current grade**/slope/gradient is computed as 'elevation_change in the last x seconds' / 'horizontal distance run in the last x seconds', where x is the time needed to cover the last 5 horizontal meters. This approach is favoured over setting x=1s to avoid noise in the gradient calculation when the (horizontal) speed is very low.
- If the speed is less than 4km/h or missing (resp. power is less than 100W or missing), then it is imputed with 4km/h (resp. 100W). Justification: The watch does not record any data when the activity is paused by the runner. In these cases, he usually temporarily stopped running and did something else (standing, walking). Since standing and slow-walking have a similarly low effect on the heart rate, we just assume slow walking at 4km/h and 100W at all paused times.
- The **gradient adjusted speed** is computed with Strava's formula (https://medium.com/strava-engineering/an-improved-gap-model-8b07ae8886c3). This is the speed that the runner would have on a flat surface if he had the same effort level as on the current gradient.

## Feature Engineering
- Two different ways of aggregrating the **gradient adjusted speed and power** were tested:
-- a) Smoothen the time series with the **exponentially weighted moving average**. Both variables had the highest correlation with heart rate with the parameters span=100s. 
-- b) For each variable used the following average values as features: Last 10s, the 20s before that (i.e. from 29s earlier to 10s earlier), the 30s before that, the 1min before that, the 2min before that, the 3min before that, the 4min before that, and the 5 min before that. 
- The **cumulative power** since the beginning of the run is added as a feature because heart rate is known to increase in the course of a constant-speed run (see cardiac drift).

## Model
- The activities were randomly split into training activities (70%) and validation activities (30%). This avoids "cheating" by using the heart rate at the beginning of the run to predict the heart rate at later times in the same run.
- **Models: Ridge** and **LightGBM**. 
- Hyperparameters were tuned with Gridsearch and repeated 70-30-train-validation-activity-splits (10 times).

## Results
- **Validation MAE** is around **6** beats per minute for both models and both feature engineering methods.


## Possible improvements
- Get more and more accurate data (e.g., from a chest strap)
- Use a **more sophisticated model** (e.g., LSTM, Transformers) to capture the time dependencies in the data
- Include **more features** such as temperature, exercises before the run
- Instead of using the mean (for power and gradient adjusted speed), experiment with other functions. For cycling power, the generalized mean with p=4 (Normalized Power®) is often favored over the arithmetic mean to assess perceived effort. For example, this means that an interval session session at 200W has higher Normalized Power® than a constant 200W effort.
- Use a more sophisticated formula for gradient adjusted speed (e.g., https://medium.com/strava-engineering/an-improved-gap-model-8b07ae8886c3)
- Include test activities to evaluate test performance

"""
with st.expander("Explanation of the project"):
    st.markdown(str_markdown_explanation)


# Read data (only at the beginning of the session)
if not "df" in st.session_state:
    st.session_state.df = _2024_04_04_predict_hr_helper_v1.load_df_all_activities_labeled_and_processed()
if not "df_activities" in st.session_state:
    # DataFrame with summarized information about each activity
    st.session_state.df_activities = helper_load_specific_df.load_df_activities(root_path_db=os.path.join(project_path, 'data')).astype({"activity_id": 'int64'}).set_index('activity_id')

# Train model and predict heart rate (only at the beginning of the session)
if not "list_train_activity_ids" in st.session_state or "train_val_or_unused_due_to_unreliable_hr" not in st.session_state.df.columns:
    _2024_04_04_predict_hr_helper_v1.train_model_and_predict_hr_v2()

# Compute and display mae
st.header("Train the regression model")
mae_train = st.session_state.df.query("train_val_or_unused_due_to_unreliable_hr == 'train'")["hr_mae"].mean()
mae_val = st.session_state.df.query("train_val_or_unused_due_to_unreliable_hr == 'val'")["hr_mae"].mean()
cols = st.columns([1, 1, 3])
cols[0].metric("MAE Train", f'{mae_train: .1f} bpm')
cols[1].metric("MAE Validation", f'{mae_val: .1f} bpm')
with cols[2]:   
    # Add button so that this (train-val split of activities, training regression model, predicting hr) can be manually rerun again
    st.button("Retrain regression model on a new random train-validation split", key="button_rerun_train_model_and_predict_hr", on_click=_2024_04_04_predict_hr_helper_v1.train_model_and_predict_hr_v2)



st.header("View predictions for a specified activity")

# Choose specific activity
_2024_04_04_predict_hr_helper_v1.input_ui_choose_activity_id()

# Display summarized information about this activity
_2024_04_04_predict_hr_helper_v1.ui_display_activity_overview()

df_specific_activity = st.session_state.df.query("activity_id == @st.session_state.activity_id")
_2024_04_04_predict_hr_helper_v1.ui_get_plotly_timeseries_fig_v5(df_specific_activity, plot_fig=True)

if False:
    # Print the training and validation activity ids
    st.write("Training activity ids:", str(st.session_state.list_train_activity_ids))
    st.write("Validation activity ids:", str(st.session_state.list_val_activity_ids))

    st.dataframe(st.session_state.df_train)


    st.write(X_train.shape, X_val.shape)

    st.write(df.shape, df_only_correct_hr_rows.shape)

    st.dataframe(df)



    st.write(list_of_all_activity_ids)


    st.write(st.session_state)