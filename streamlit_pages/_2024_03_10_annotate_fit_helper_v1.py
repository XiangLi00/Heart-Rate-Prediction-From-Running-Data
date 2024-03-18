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
from streamlit_plotly_events import plotly_events


from utils import helper_load_fit_file_v1, helper_load_specific_df, helper_pandas, helper_streamlit
from utils.Activity import Activity

# from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report


def section_running_activities_show_and_filter(project_path: str = os.getcwd()):
    st.header("List of running activities")
    # Load the overview of all activities
    df_activities = helper_load_specific_df.load_df_activities(root_path_db=os.path.join(project_path, 'data'))

    # Filter only running activities
    # Reason: Add this because otherwise helper_load_fit_file_v1.load_fit_file() otherwise expects columns that don't exist (e.g. {'activity_type', 'power', 'step_length'} for walking/hiking activities and  ('cadence', 'fractional_cadence'), additionally for other activities)
    df_running_activities = df_activities.query('sport == "running"')  

    # Add filtering UI
    with st.expander("Search all activities"):
        df_running_activities_filtered = helper_streamlit.add_df_activities_filtering_ui(df_running_activities)
        
        st.write(f'Showing {len(df_running_activities_filtered)} out of {len(df_running_activities)} running activities')
        st.dataframe(df_running_activities_filtered)
    
    return df_activities, df_running_activities, df_running_activities_filtered 

def section_select_activity_and_retrieve_df(
        df_activities: pd.DataFrame, 
        project_path: str = os.getcwd(), 
        display_df_overview_option = ["no", "single_line_df", "pretty_selected_information"][2],
        df_loading_method = ["load_fit_file_v1", "activity_init"][0]
    ):
    st.header("View specific activity")
    # Select specific activity
    activity_id = st.text_input("Enter activity id", value="14361204813", key="activity_id") # hill reps=14057922527

    if False:  # Alternative to use drpdown menu
        list_activity_ids = df_activities["activity_id"].unique().tolist()
        activity_id = st.selectbox("Select activity id", list_activity_ids)
    if activity_id not in df_activities["activity_id"].values:
        st.error(f"Activity id '{activity_id}' not found")

    if True:
        # ser_activity_overview = df_activities.query('activity_id == @activity_id').squeeze()
        ser_activity_overview = df_activities.query('activity_id == @activity_id').squeeze()

        # Display general information about the activity
        if display_df_overview_option == "single_line_df":
            st.dataframe(df_activities.query('activity_id == @activity_id'))
        elif display_df_overview_option == "pretty_selected_information":
            cols_activity_overview = st.columns(4)
            
            # cols_activity_overview[0].metric("Type", f'{ser_activity_overview.sport}: {ser_activity_overview.sub_sport}')
            cols_activity_overview[0].metric("Distance", f'{ser_activity_overview.distance: .2f} km')
            cols_activity_overview[1].metric("Moving Time", 
                                             f'{ser_activity_overview.moving_time.total_seconds()//3600:.0f}h {(ser_activity_overview.moving_time.total_seconds()% 3600) // 60:.0f}min ')
            cols_activity_overview[2].metric("Avg Speed", f'{ser_activity_overview.avg_speed: .2f} km/h')
            cols_activity_overview[3].metric("Avg HR", f'{ser_activity_overview.avg_hr: .0f} bpm')


    # Load the fit file for this activity
    if df_loading_method == "load_fit_file_v1":
        path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{activity_id}_ACTIVITY.fit')
        df = helper_load_fit_file_v1.load_fit_file(path_fit_file)
    elif df_loading_method == "activity_init":
        config = dict()
        config["df__elevation_change_interm__threshold_for_setting_sudden_change_to_zero"] = 2  # [m]. if elevation changes by 2m within 1s, then we assume it's noise and set it to 0
        config["df__elevation_change__gaussian_kernel_sigma"] = 4  # [s], Applies Gaussian kernel to elevation_change_interm column with this standard deviation
        config["df__distance__gaussian_kernel_sigma"] = 5  # [s], Applies Gaussian kernel to distance_raw column with this standard deviation. Affects speed
        config["df__grade__delta_distance_for_grade_computation"] = 10  # [m], Computes grade := (delta elevation_change / delta distance), over the last e.g. >=5m distance. Can't make it too large (100m) because then the grade informtion will lag behind ~100m/2. Need it to be >0.5 because otherwise small delta distance can create a lot of noisy/huge grades.
        config["df__keep_raw_elevation_distance_columns_for_debugging"] = True

        df = Activity(id=activity_id, project_path=project_path, config=config).df 

    return df


def display_df(df, option="column_names"):
    if option == "df_and_column_names":
        st.dataframe(df)
        st.write("df columns: " + str(list(df.columns)))
    if option == "column_names":
        st.write("df columns: " + str(list(df.columns)))

@st.cache_data
def section_get_plotly_timeseries_fig_v4(df: pd.DataFrame, activity_id: str):
    fig = make_subplots(rows=1, cols=1, 
                        shared_xaxes=True, shared_yaxes=False, 
                        specs=[[{"secondary_y": True}],],
                        vertical_spacing=0.03
                        )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hr"], mode='lines', name='HR', line=dict(color='red', width=3)),
                  row=1,col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed"], mode='lines', name='GAP', line=dict(color='blue', dash="solid")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power"]/25, mode='lines', name='Power', line=dict(color='lightpink')),
                  row=1, col=1, secondary_y=True)

    # Ticks and lines for each tick
    array_pace_ticks = np.array([4,5,6,7,8,10])  # paces for which to display ticks
    array_hr_ticks = np.array([120, 140, 150, 160, 170, 180])  # hr values for which to display ticks
    fig.update_yaxes(
        gridcolor='deeppink',
        griddash=["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"][1],
        gridwidth=2,
        tickvals=array_hr_ticks,  # display ticks at these values
        secondary_y=False,
    )
    fig.update_yaxes(
        gridcolor='green',
        griddash = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"][3],
        gridwidth=2,
        tickvals=60/array_pace_ticks,  # convert pace to km/h
        ticktext=array_pace_ticks,
        secondary_y=True,
    )

    # Set y-axis titles/range
    fig.update_yaxes(range=[100, 200], title_text="HR", row=1, col=1, secondary_y=False) 
    fig.update_yaxes(range=[0, 17], title_text="Pace", row=1, col=1, secondary_y=True) 
    fig.update_yaxes(fixedrange=True)

    # Set interactive behaviour and layout
    # fig = helper_streamlit.update_screen_height_of_fig_v2(fig, height_factor=0.8, debug=False) # Does not work well with ptloyl_events. glitches
    fig = helper_streamlit.update_screen_height_of_fig_v3(
        fig=fig,
        screen_height=st.session_state.screen_height,
        screen_width=st.session_state.screen_width,
        height_factor=0.8,
        debug=False
    )
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
        hovermode='x unified',  # Enable unified hover mode across all traces
        # legend=dict(orientation="h")
        legend=dict(x=0.80, y=0.02)
    )
    # st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

    return fig
def section_show_plotly_timeseries_plot_v3_5(df: pd.DataFrame):
    # Purpose: Experimental. trying out some widget interactions as well

    fig = make_subplots(rows=1, cols=1, 
                        shared_xaxes=True, shared_yaxes=False, 
                        specs=[[{"secondary_y": True}],],
                        vertical_spacing=0.03
                        )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hr"], mode='lines', name='Heart Rate', line=dict(color='red', width=3)),
                  row=1,col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed"], mode='lines', name='GAP', line=dict(color='blue', dash="solid")),
                  row=1, col=1, secondary_y=True)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["speed_garmin"], mode='lines', name='Speed (Garmin)', line=dict(color='gray')),row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power"]/30, mode='lines', name='Power', line=dict(color='lightpink')),
                  row=1, col=1, secondary_y=True)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade"], mode='lines', name='grade', line=dict(color='grey')),                 row=1, col=1, secondary_y=False)
    # fig.add_trace(go.Scatter(x=df["timestamp"], y=df["elevation_change_raw"]*20, mode='markers', name='elevation_change_raw', line=dict(color='lightgrey')),                 row=1, col=1, secondary_y=False)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["distance_raw"]*3.6, mode='markers', name='distance', line=dict(color='lightgrey')),                 row=1, col=1, secondary_y=True)
    # fig.add_trace(go.Scatter(x=df["timestamp"], y=df["distance_smoothed"]*5, mode='lines', name='distance_smoothed', line=dict(color='orange')),                 row=1, col=1, secondary_y=True)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["speed"], mode='lines', name='speed', line=dict(color='cyan')),                 row=1, col=1, secondary_y=True)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["speed_garmin"], mode='lines', name='speed_garmin', line=dict(color='blue')),                 row=1, col=1, secondary_y=True)

    # fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power"]/30, mode='lines', name='power', line=dict(color='red')),                 row=1, col=1, secondary_y=True)

    # fig.add_trace(go.Scatter(x=df["timestamp"], y=df["steps_per_min"]/10, mode='lines', name='steps_per_min', line=dict(color='lightsteelblue')),                 row=1, col=1, secondary_y=True)

    list_pace_ticks = np.array([4,5,6,7,8,10])  # paces for which to display ticks
    list_hr_ticks = np.array([120, 140, 150, 160, 170, 180])  # paces for which to display ticks

    # Update Axes
    fig.update_yaxes(
        gridcolor='deeppink',
        griddash=["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"][1],
        gridwidth=2,
        tickvals=list_hr_ticks,  
        secondary_y=False,
    )
    fig.update_yaxes(
        gridcolor='green',
        griddash = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"][3],
        gridwidth=2,
        tickvals=60/list_pace_ticks,  # convert pace to km/h
        ticktext=list_pace_ticks,
        secondary_y=True,
    )
    fig.update_yaxes(range=[100, 200], title_text="HR", row=1, col=1, secondary_y=False) 
    fig.update_yaxes(range=[0, 17], title_text="Pace", row=1, col=1, secondary_y=True) 
    fig.update_yaxes(fixedrange=True)


    # Get figure layout
    fig_layout = fig.full_figure_for_development(warn=False).layout
    fig_xlim = fig_layout.xaxis.range
    fig_ylim = fig_layout.yaxis.range
    if False:
        # Add rectangular background boxes marking times manually labeled
        fig.add_shape(
            type="rect",
            x0=pd.to_datetime('2024-02-20 17:24:47+00:00'),
            y0=fig_layout.yaxis.range[0],
            x1=pd.to_datetime('2024-02-20 17:27:47+00:00'),
            y1=fig_layout.yaxis.range[1],
            fillcolor="blue",
            opacity=0.1,
            layer="below",
        )


    # Set interactive behaviour and layout
    fig = helper_streamlit.update_screen_height_of_fig_v2(fig, height_factor=0.9, debug=False) # Does not work well with ptloyl_events. glitches
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
        hovermode='x unified',  # Enable unified hover mode across all traces
    )
    # Show plot
    fig_plotting_model = ["plotly_event_handler", "plotly_slider_v1"][1]
    if fig_plotting_model == "plotly_event_handler":
        # Recommendation: not use it
        # Advantages: more or less built in
        # + can retrieve value of ll lines at specified x
        # Disadvantages:
        # - does not work well with update_screen_height_of_fig_v2(fig, height_factor=0.7, debug=False). glitches. also hovermode='x unified' might need to be disabled?
        # - there is a lag ("running") for 5s each time something is selected
        # - unable to add config scrollzoom
        # - additionl library
        # how to use it. pass (plotly_elevents(fig) instead of st.plotly_chart(fig))
        selected_plotly_point = plotly_events(fig, click_event=True, select_event=False, hover_event=False)
        if len(selected_plotly_point) > 0:
            selected_datetime = selected_plotly_point[0]["x"]
            st.write(selected_datetime)
        st.write(selected_plotly_point)
    elif fig_plotting_model == "plotly_slider_v1":

        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        fig.add_shape(
                type="rect",
                x0=pd.to_datetime('2024-03-09 11:55+00:00'),
                y0=fig_layout.yaxis.range[0],
                x1=pd.to_datetime('2024-03-09 11:56+00:00'),
                y1=fig_layout.yaxis.range[1],
                fillcolor="yellow",
                opacity=0.1,
                layer="below",
            )
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


        def datetime_range_slider_on_change():
            start_datetime = st.session_state.datetime_range_slider[0]
            end_datetime = st.session_state.datetime_range_slider[1]
            fig.add_shape(
                type="rect",
                x0=start_datetime,
                y0=fig_layout.yaxis.range[0],
                x1=end_datetime,
                y1=fig_layout.yaxis.range[1],
                fillcolor="yellow",
                opacity=0.1,
                layer="below",
            )

            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


        st.select_slider(
            'Select a range of color wavelength',
            options=df.timestamp,
            value=(df.timestamp.iloc[0], df.timestamp.iloc[-1]),
            key="datetime_range_slider",
            on_change=datetime_range_slider_on_change
        )

        st.write(st.session_state)


def section_show_plotly_timeseries_plot_v3(df: pd.DataFrame):
    # Purpose: exploring

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
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["uphill_grade_ew_10s"], mode='lines', name='uphill_grade_ew_10s', line=dict(color='blueviolet')), row=1, col=1, secondary_y=True)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["uphill_grade_ew_120s"], mode='lines', name='uphill_grade_ew_120s', line=dict(color='black')), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade_last_20m"], mode='lines', name='grade_last_20m', line=dict(color='blueviolet')), row=1, col=1, secondary_y=True)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade_ew_120s"], mode='lines', name='grade_ew_120s', line=dict(color='black')), row=1, col=1, secondary_y=True)
    
    
    ## Second subplot
    # Add Elevation trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["elevation"], mode='lines', name='elevation', line=dict(color='green')),                 row=2, col=1, secondary_y=False)
    # Add Power trace
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power100"], mode='lines', name='power100'), 
                row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power100_ew_10s"], mode='lines', name='power100_ew_10s'), 
                row=2, col=1, secondary_y=True)
    

    ## Third subplot
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["elevation_change"], mode='lines', name='elevation_change', line=dict(color='green')), row=3, col=1, secondary_y=False)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade_ew_10s"], mode='lines', name='grade_ew_10s'), row=3, col=1, secondary_y=True)
    #fig.add_trace(go.Scatter(x=df["timestamp"], y=df["grade_ew_120s"], mode='lines', name='grade_ew_120s', line=dict(color='black')), row=3, col=1, secondary_y=True)

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



