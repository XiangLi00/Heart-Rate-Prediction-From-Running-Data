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

from utils import helper_load_fit_file, helper_load_specific_df, helper_pandas, helper_streamlit
# from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report


def section_running_activities_show_and_filter(project_path: str = os.getcwd()):
    st.header("List of running activities")
    # Load the overview of all activities
    df_activities = helper_load_specific_df.load_df_activities(root_path_db=os.path.join(project_path, 'data'))

    # Filter only running activities
    # Reason: Add this because otherwise helper_load_fit_file.load_fit_file() otherwise expects columns that don't exist (e.g. {'activity_type', 'power', 'step_length'} for walking/hiking activities and  ('cadence', 'fractional_cadence'), additionally for other activities)
    df_running_activities = df_activities.query('sport == "running"')  

    # Add filtering UI
    with st.expander("Search all activities"):
        df_running_activities_filtered = helper_streamlit.add_df_activities_filtering_ui(df_running_activities)
        
        st.write(f'Showing {len(df_running_activities_filtered)} out of {len(df_running_activities)} running activities')
        st.dataframe(df_running_activities_filtered)
    
    return df_activities, df_running_activities, df_running_activities_filtered 

def section_select_activity_and_retrieve_df(df_activities: pd.DataFrame, project_path: str = os.getcwd()):
    st.header("View specific activity")
    # Select specific activity
    activity_id = st.text_input("Enter activity id", value="14161114490")

    if False:  # Alternative to use drpdown menu
        list_activity_ids = df_activities["activity_id"].unique().tolist()
        activity_id = st.selectbox("Select activity id", list_activity_ids)
    if activity_id not in df_activities["activity_id"].values:
        st.error(f"Activity id '{activity_id}' not found")

    # Display general information about the activity
    st.dataframe(df_activities.query('activity_id == @activity_id'))

    # Load the fit file for this activity
    path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{activity_id}_ACTIVITY.fit')
    df = helper_load_fit_file.load_fit_file(path_fit_file)

    return df


def test1(df: pd.DataFrame):
    st.write("streamlit_pages.annotate_fit_helper.test1() v4")


