from datetime import datetime, timedelta
import importlib
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
import streamlit as st
from streamlit_js_eval import streamlit_js_eval


project_path = os.getcwd()

#sys.path.append(os.path.join(project_path, 'streamlit_pages'))  

from streamlit_pages._2024_03_10_annotate_fit_helper import test1
from utils import helper_load_fit_file, helper_load_specific_df, helper_pandas, helper_streamlit
# from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report




st.header("List of running activities")
# Load the overview of all activities
df_activities = helper_load_specific_df.load_df_activities(root_path_db=os.path.join(project_path, 'data'))

# Filter only running activities
# Reason: Add this because otherwise helper_load_fit_file.load_fit_file() otherwise expects columns that don't exist (e.g. {'activity_type', 'power', 'step_length'} for walking/hiking activities and  ('cadence', 'fractional_cadence'), additionally for other activities)
df_running_activities = df_activities.query('sport == "running"')  

# Add filtering UI
with st.expander("Search all activities"):
    df_running_activities_filtered = helper_streamlit.add_df_activities_filtering_ui(df_running_activities)
    
    st.write(f'Showing {len(df_running_activities_filtered)} out of {len(df_activities)} running activities')
    st.dataframe(df_running_activities_filtered)


st.header("View specific activity")
# Select specific activity
activity_id = st.text_input("Enter activity id", value="14161114490")
if False:
    list_activity_ids = df_activities["activity_id"].unique().tolist()
    activity_id = st.selectbox("Select activity id", list_activity_ids)
if activity_id not in df_activities["activity_id"].values:
    st.error(f"Activity id '{activity_id}' not found")

# Display general information about the activity
st.dataframe(df_activities.query('activity_id == @activity_id'))

# Load the fit file for this activity
path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{activity_id}_ACTIVITY.fit')
df = helper_load_fit_file.load_fit_file(path_fit_file)

# View df
st.dataframe(df)

# ––––––––––––––– Start Plotting

fig = make_subplots(rows=2, cols=1, 
                    shared_xaxes=True, shared_yaxes=False, 
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}],],
                    vertical_spacing=0.03
                    )

# Add HR trace
fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["hr"], mode='lines', name='HR', line=dict(color='crimson')), 
            row=1, col=1, secondary_y=False)
# Add Pace trace
fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["pace"], mode='lines', name='Pace', line=dict(color='deepskyblue')), 
            row=1, col=1, secondary_y=True)
# Add Altitude trace
fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["altitude"], mode='lines', name='Altitude', line=dict(color='green')),                 row=2, col=1, secondary_y=False)
# Add Cadence trace to the same subplot as Altitude
fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["real_cadence"], mode='lines', name='Cadence'), 
            row=2, col=1, secondary_y=True)

# Update layout settings
fig.update_layout(
    dragmode='pan',  # zoom, pan, select, lasso
    hovermode='x unified'  # Enable unified hover mode across all traces
)
fig = update_screen_height_of_fig(fig)

# Set y-axis titles
fig.update_yaxes(title_text="HR", row=1, col=1, secondary_y=False)
fig.update_yaxes(range=[math.log10(15), math.log10(3)], title_text="Pace",  type="log", row=1, col=1,secondary_y=True) # , type="log",autorange="reversed",  ,autorange="reversed"
fig.update_yaxes(title_text="Altitude", secondary_y=False, row=2, col=1)
fig.update_yaxes(range=[150, 200], title_text="Cadence", secondary_y=True, row=2, col=1)
fig.update_yaxes(fixedrange=True)

# Your existing code to configure and display the figure
config = {'scrollZoom': True}

st.plotly_chart(fig, use_container_width=True, config=config)


test1(df)

