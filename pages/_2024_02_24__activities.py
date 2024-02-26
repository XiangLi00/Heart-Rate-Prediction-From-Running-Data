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
# from ydata_profiling import ProfileReport

from utils.helper_load_df import load_df, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report

# Append project path to system path
# print(f"Project root folder: {os.getcwd()}")
sys.path.append(os.getcwd())  


def page():
    """
        table: monitoring_hr

        First version, trying out plotly, altair, vega-lite, and bokeh.
        
    """

    activity_id_selected = st.text_input("Enter activity_id", "14057922527")

    df_specific_activity = load_df('garmin_activities.db', 'activity_records', root_path_db=os.path.join(os.getcwd(), 'data'),
                            sql_selected_columns="*",
                            sql_condition=f"activity_id={activity_id_selected}",)

    st.dataframe(df_specific_activity)
    st.write(f"Shape: {df_specific_activity.shape}")

    df_specific_activity["pace"] = 60 / (df_specific_activity["speed"])
    df_specific_activity["real_cadence"] = 2*(df_specific_activity["cadence"])

    plot_specific_activity6(df_specific_activity)

def plot_specific_activity6(df_specific_activity: pd.DataFrame):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=False, 
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    # Add HR trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["hr"], mode='lines', name='HR'), 
                row=1, col=1, secondary_y=False)
    # Add Pace trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["pace"], mode='lines', name='Pace'), 
                row=1, col=1, secondary_y=True)
    # Add Altitude trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["altitude"], mode='lines', name='Altitude', line=dict(color='green')), 
                row=2, col=1, secondary_y=False)
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
    fig.update_yaxes(range=[3, 10], title_text="Pace", row=1, col=1,secondary_y=True)
    fig.update_yaxes(title_text="Altitude", secondary_y=False, row=2, col=1)
    fig.update_yaxes(range=[150, 200], title_text="Cadence", secondary_y=True, row=2, col=1)
    fig.update_yaxes(fixedrange=True)

    # Your existing code to configure and display the figure
    config = {'scrollZoom': True}

    st.plotly_chart(fig, use_container_width=True, config=config)

def plot_specific_activity5(df_specific_activity: pd.DataFrame):
    """
        4 columns in 2 plots. works

        - automatic screen height
    """
    st.write("plot_specific_activity5")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=False, 
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    

    # Add HR trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["hr"], mode='lines', name='HR'), 
                row=1, col=1, secondary_y=False)
    # Add Pace trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["pace"], mode='lines', name='Pace'), 
                row=1, col=1, secondary_y=True)
    # Add Altitude trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["altitude"], mode='lines', name='Altitude'), 
                row=2, col=1, secondary_y=False)
    # Add Cadence trace to the same subplot as Altitude
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["real_cadence"], mode='lines', name='Cadence'), 
                row=2, col=1, secondary_y=True)

    # Set y-axis titles
    fig.update_yaxes(title_text="HR", row=1, col=1, secondary_y=False,)
    fig.update_yaxes(range=[3, 10], title_text="Pace", row=1, col=1,secondary_y=True)
    fig.update_yaxes(title_text="Altitude", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Cadence", secondary_y=True, row=2, col=1)
    fig.update_yaxes(fixedrange=True)

    # Update layout settings
    # fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
    )
    config = {'scrollZoom': True}

    if True:
        screen_height, screen_width = get_screen_height_and_width()
        st.write(f"screen_height: {screen_height}, screen_width: {screen_width}, type_screen_type: {type(screen_height)}")

        if screen_height is not None:
            try:
                fig.update_layout(height=screen_height*0.9)
            except TypeError as e:
                st.write(f"TypeError. screeen_height = {screen_height}")


    st.plotly_chart(fig, use_container_width=True, config=config)


def update_screen_height_of_fig(fig: plotly.graph_objs.Figure) -> plotly.graph_objs.Figure:
    screen_height, screen_width = get_screen_height_and_width()
    st.write(f"screen_height: {screen_height}, screen_width: {screen_width}, type_screen_height: {type(screen_height)}")

    if screen_height is not None:
        try:
            fig.update_layout(height=screen_height*0.9)
        except TypeError as e:
            st.write(f"TypeError. screeen_height = {screen_height}")
    return fig


def get_screen_height_and_width():
    screen_height=streamlit_js_eval(js_expressions='screen.height', key='get_screen_height_javascript')
    screen_width = streamlit_js_eval(js_expressions='screen.width', key='get_screen_width_javascript')
    return screen_height, screen_width

def plot_specific_activity4(df_specific_activity: pd.DataFrame):
    """
        4 columns in 2 plots. works
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=False, 
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    

    # Add HR trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["hr"], mode='lines', name='HR'), 
                row=1, col=1, secondary_y=False)

    # Add Pace trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["pace"], mode='lines', name='Pace'), 
                row=1, col=1, secondary_y=True)



    # Add Altitude trace
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["altitude"], mode='lines', name='Altitude'), 
                row=2, col=1, secondary_y=False)

    # Add Cadence trace to the same subplot as Altitude
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["real_cadence"], mode='lines', name='Cadence'), 
                row=2, col=1, secondary_y=True)

    # Set y-axis titles
    fig.update_yaxes(title_text="HR", row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[3, 10], title_text="Pace", row=1, col=1,secondary_y=True)
    fig.update_yaxes(title_text="Altitude", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Cadence", secondary_y=True, row=2, col=1)
    fig.update_yaxes(fixedrange=True)

    # Update layout settings
    # fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
    )
    config = {'scrollZoom': True}
    # fig.update_layout(height=1200)


    st.plotly_chart(fig, use_container_width=True, config=config)



def plot_specific_activity3(df_specific_activity: pd.DataFrame):
    """Works. Plots Pace and HR overlaying each other. Pace is on the secondary y-axis.

    Args:
        df_specific_activity (pd.DataFrame): _description_
    """

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=False, specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["hr"], mode='lines', name='HR'), 
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["pace"], mode='lines', name='Pace'), 
                  row=1, col=1, secondary_y=True)
    fig.update_yaxes(range=[3, 10], title_text="Pace", secondary_y=True)

    
    fig.update_yaxes(title_text="HR", secondary_y=False)
    fig.update_yaxes(title_text="Pace", secondary_y=True)


    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    # fig.update_yaxes(fixedrange=True)  # Lock the y-axis
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
    )
    config = {'scrollZoom': True}

    st.plotly_chart(fig, use_container_width=True, config=config)
def plot_specific_activity2(df_specific_activity: pd.DataFrame):

    """
        Trying multiple y axes does not work here
    """

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=False)
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["hr"], mode='lines', name='HR'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["pace"], mode='lines', name='Pace', yaxis='y2'), row=1, col=1)

    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_layout(yaxis=dict(title=f'Y-axis 1'), yaxis2=dict(title=f'Y-axis 2', anchor="free", overlaying="y", side="right"))
    # fig.update_yaxes(fixedrange=True)  # Lock the y-axis
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
    )
    config = {'scrollZoom': True}

    st.plotly_chart(fig, use_container_width=True, config=config)


def plot_specific_activity(df_specific_activity: pd.DataFrame):
    fig = px.line(
        df_specific_activity, x="timestamp", y="hr",
        title="Specific activity")
    fig.add_trace(go.Scatter(x=df_specific_activity["timestamp"], y=df_specific_activity["pace"], mode='lines', name='Pace'))


    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_yaxes(fixedrange=True)  # Lock the y-axis
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
    )
    config = {'scrollZoom': True}
    
    st.plotly_chart(fig, use_container_width=True, config=config)

  