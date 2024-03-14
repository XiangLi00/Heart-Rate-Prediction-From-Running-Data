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
    activity_id = st.text_input("Enter activity id", value="14057922527")

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


def section_show_plotly_timeseries_plot_v2(df: pd.DataFrame):
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, shared_yaxes=False, 
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}],],
                        vertical_spacing=0.03
                        )

    ## First subplot
    # Add HR trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hr"], mode='lines', name='HR', line=dict(color='crimson')), 
                row=1, col=1, secondary_y=False)
    # Add gaspeed4
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed4_ew_10s"], mode='lines', name='gaspeed4_ew_10s', line=dict(color='lightgreen')), 
                row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed4_ew_120s"], mode='lines', name='gaspeed4_ew_120s', line=dict(color='deepskyblue')), 
            row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["speed"]-4, mode='lines', name='speed', line=dict(color='darkorange')), 
        row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["uphill_grade_ew_10s"], mode='lines', name='uphill_grade_ew_10s', line=dict(color='blueviolet')), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["uphill_grade_ew_120s"], mode='lines', name='uphill_grade_ew_120s', line=dict(color='black')), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade_ew_120s"], mode='lines', name='grade_ew_120s', line=dict(color='black')), row=1, col=1, secondary_y=True)
    
    
    ## Second subplot
    # Add Elevation trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["elevation"], mode='lines', name='elevation', line=dict(color='green')),                 row=2, col=1, secondary_y=False)
    # Add Power trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power100"], mode='lines', name='power100'), 
                row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power100_ew_120s"], mode='lines', name='power100_ew_120s'), 
                row=2, col=1, secondary_y=True)
    

    ## Third subplot
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["elevation_change"], mode='lines', name='elevation_change', line=dict(color='green')), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade_ew_10s"], mode='lines', name='grade_ew_10s'), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade_ew_120s"], mode='lines', name='grade_ew_120s', line=dict(color='black')), row=3, col=1, secondary_y=True)

    # Update layout settings
    fig = helper_streamlit.update_screen_height_of_fig_v2(fig, height_factor=0.9, debug=False)

    if False: 
        # Set y-axis titles
        fig.update_yaxes(title_text="HR", row=1, col=1, secondary_y=False)
        fig.update_yaxes(range=[math.log10(15), math.log10(3)], title_text="Pace",  type="log", row=1, col=1,secondary_y=True) # , type="log",autorange="reversed",  ,autorange="reversed"
        fig.update_yaxes(title_text="Altitude", secondary_y=False, row=2, col=1)
        fig.update_yaxes(range=[150, 200], title_text="Cadence", secondary_y=True, row=2, col=1)
    fig.update_yaxes(range=[-20,20], title_text="(GA)Speed and Grade", row=1, col=1, secondary_y=True) 
    fig.update_yaxes(fixedrange=True)

    # Set interactive behaviour
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
        hovermode='x unified'  # Enable unified hover mode across all traces
    )
    config = {'scrollZoom': True}

    # Show plot
    st.plotly_chart(fig, use_container_width=True, config=config)

def section_show_plotly_timeseries_plot_v1(df: pd.DataFrame):

    

    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, shared_yaxes=False, 
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}],],
                        vertical_spacing=0.03
                        )

    # Add HR trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hr"], mode='lines', name='HR', line=dict(color='crimson')), 
                row=1, col=1, secondary_y=False)
    # Add Pace trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["speed"], mode='lines', name='sped', line=dict(color='deepskyblue')), 
                row=1, col=1, secondary_y=True)
    # Add Altitude trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["elevation"], mode='lines', name='elevation', line=dict(color='green')),                 row=2, col=1, secondary_y=False)
    # Add Cadence trace to the same subplot as Altitude
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["steps_per_min"], mode='lines', name='Cadence'), 
                row=2, col=1, secondary_y=True)

    # Update layout settings

    # fig = update_screen_height_of_fig(fig)

    # Set y-axis titles
    fig.update_yaxes(title_text="HR", row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[math.log10(15), math.log10(3)], title_text="Pace",  type="log", row=1, col=1, secondary_y=True)  # , type="log",autorange="reversed",  ,autorange="reversed"
    fig.update_yaxes(title_text="Altitude", secondary_y=False, row=2, col=1)
    fig.update_yaxes(range=[150, 200], title_text="Cadence", secondary_y=True, row=2, col=1)
    fig.update_yaxes(fixedrange=True)

    # Set interactive behaviour
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
        hovermode='x unified'  # Enable unified hover mode across all traces
    )
    config = {'scrollZoom': True}

    # Show plot
    st.plotly_chart(fig, use_container_width=True, config=config)



    



def test1(df: pd.DataFrame):
    st.write("streamlit_pages.annotate_fit_helper.test1() v4")


