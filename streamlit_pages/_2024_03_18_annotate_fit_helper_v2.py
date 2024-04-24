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


    if True:  # Alternative to use dropdown menu
        list_activity_ids = df_activities["activity_id"].unique().tolist()
        list_activity_ids_of_long_running_activities = df_activities.query("sport == 'running' and sub_sport=='generic' and distance >= 5").activity_id.unique().tolist()
        st.selectbox("Select activity id", list_activity_ids_of_long_running_activities, key="activity_id_for_labeling", index=7)
    if False:  # Alternative to enter activity id manually
            # Select specific activity
        if not "activity_id" in st.session_state:
            st.session_state.activity_id_for_labeling = 14441010384
        activity_id = st.text_input("Enter activity id", key="activity_id") # hill reps=14057922527
        if st.session_state.activity_id_for_labeling not in df_activities["activity_id"].values:
            st.error(f"Activity id '{activity_id}' not found. Please copy and paste a valid activity id from the table above.", icon="🚨")

    if True:
        # ser_activity_overview = df_activities.query('activity_id == @activity_id').squeeze()
        ser_activity_overview = df_activities.query('activity_id == @st.session_state.activity_id_for_labeling').squeeze()

        # Display general information about the activity
        if display_df_overview_option == "single_line_df":
            st.dataframe(df_activities.query('activity_id == @st.session_state.activity_id_for_labeling'))
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
        path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{st.session_state.activity_id_for_labeling}_ACTIVITY.fit')
        df = helper_load_fit_file_v1.load_fit_file(path_fit_file)
    elif df_loading_method == "activity_init":
        config = dict()
        config["df__elevation_change_interm__threshold_for_setting_sudden_change_to_zero"] = 2  # [m]. if elevation changes by 2m within 1s, then we assume it's noise and set it to 0
        config["df__elevation_change__gaussian_kernel_sigma"] = 4  # [s], Applies Gaussian kernel to elevation_change_interm column with this standard deviation
        config["df__distance__gaussian_kernel_sigma"] = 5  # [s], Applies Gaussian kernel to distance_raw column with this standard deviation. Affects speed
        config["df__grade__delta_distance_for_grade_computation"] = 10  # [m], Computes grade := (delta elevation_change / delta distance), over the last e.g. >=5m distance. Can't make it too large (100m) because then the grade informtion will lag behind ~100m/2. Need it to be >0.5 because otherwise small delta distance can create a lot of noisy/huge grades.
        config["df__keep_raw_elevation_distance_columns_for_debugging"] = True

        df = Activity(id=st.session_state.activity_id_for_labeling, project_path=project_path, config=config).df 

    return df



@st.cache_data
def section_get_plotly_timeseries_fig_v4(_df: pd.DataFrame, activity_id: str, plot_fig: bool = False):
    df = _df
    fig = make_subplots(rows=1, cols=1, 
                        shared_xaxes=True, shared_yaxes=False, 
                        specs=[[{"secondary_y": True}],],
                        vertical_spacing=0.03
                        )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hr"], mode='lines', name='Heart Rate (measured)', line=dict(color='red', width=3)),
                  row=1,col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed"], mode='lines', name='Gradient Adjusted Pace [min/km]', line=dict(color='cornflowerblue', dash="dash")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed4_ew_100s"], mode='lines', name='Gradient Adjusted Pace (exp. smoothed with span=100s) [min/km]', line=dict(color='blue', dash="solid")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power"]/25, mode='lines', name='Power [W]', line=dict(color='lightpink', dash="dash")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power100_ew_100s"]/25, mode='lines', name='Power (exp. smoothed with span=100s) [W]', line=dict(color='coral')),
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
    fig.update_yaxes(range=[100, 200], row=1, col=1, secondary_y=False) 
    fig.update_yaxes(range=[0, 17], row=1, col=1, secondary_y=True) 
    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(range=[df["timestamp"].iloc[0], df["timestamp"].iloc[-1]])

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
        uirevision='foo',  # Keep zoom level when updating layout
        # legend=dict(orientation="h")
        legend=dict(x=0.70, y=0.02),  # (0,0) denotes bottom left
        margin=dict(l=0, r=0, t=0, b=0),
    )

    if plot_fig:
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

    return fig

def section_upload_existing_df_hr_accuracy_labels():
    # Adds UI to upload existing df_hr_accuracy_labels csv file
    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Continue from existing labels")
            uploaded_file_hr_accuracy_labels = st.file_uploader("Upload existing df_hr_accuracy_labels (optional)", type=["csv", "xlsx"])
            if uploaded_file_hr_accuracy_labels is not None:
                df_uploaded = pd.read_csv(uploaded_file_hr_accuracy_labels)
                if set(df_uploaded.columns) != set(st.session_state.columns_df_hr_accuracy_labels):
                    st.error(f'File has wrong columns. It should have columns {st.session_state.columns_df_hr_accuracy_labels}, however it has {list(df_uploaded.columns)}.', icon="🚨")
                else:
                    st.session_state.df_hr_accuracy_labels = df_uploaded
                    st.success('File uploaded successfully', icon="🎉")

        with col2:
            st.subheader("Reset existing labels")
            # Add button for resetting df_hr_accuracy_labels for this activity id
            def button_drop_all_row_for_this_activity_from_df_hr_accuracy_labels_on_click():
                st.session_state.df_hr_accuracy_labels = st.session_state.df_hr_accuracy_labels.query("activity_id != @st.session_state.activity_id_for_labeling")
            st.button(":black[Reset all labels for this activity]", key="button_drop_all_row_for_this_activity_from_df_hr_accuracy_labels", on_click=button_drop_all_row_for_this_activity_from_df_hr_accuracy_labels_on_click)

            # Add button for resetting df_hr_accuracy_labels completly
            def button_reset_from_df_hr_accuracy_labels_on_click():
                st.session_state.df_hr_accuracy_labels = pd.DataFrame(columns=st.session_state.columns_df_hr_accuracy_labels)
            st.button(":red[Reset labels for all activities]", key="button_reset_df_hr_accuracy_labels", on_click=button_reset_from_df_hr_accuracy_labels_on_click)




def section_slider_datetime_selection_and_add_colored_background_to_fig(fig: plotly.graph_objs.Figure, df: pd.DataFrame):
    # Adds a datetime slider to selected datetime range to be labeled
    # Adds colored background to fig to visualize labeled time ranges (green, orange, red for hr accurary. blue to selected range)

    fig_layout = fig.full_figure_for_development(warn=False).layout

    # Define datetime range select slider
    # Selects the start and end datetime (closed interval). In this range we categorize trustworthiness of HR data
    def datetime_range_slider_on_change():
        pass
    def datetime_range_slider_format_func(timestamp: pd.Timestamp) -> str:
        datetime = pd.to_datetime(timestamp)
        string_repr = f"{datetime.hour}   {datetime.minute:02d}   {datetime.second:02d}"
        return string_repr
    st.select_slider(
        'Select time segment of the activity to label HR accuracy',
        options=df.timestamp,
        value=(df.timestamp.iloc[0], df.timestamp.iloc[-1]),
        key="datetime_range_slider",
        on_change=datetime_range_slider_on_change,
        format_func=datetime_range_slider_format_func
    )

    # Mark selected time range with blue background
    # Remark: this has to be added here and not in on_change function bc. otherwise no change
    fig.add_shape(
        type="rect",
        x0=st.session_state.datetime_range_slider[0], # selected start time
        y0=fig_layout.yaxis.range[0],
        x1=st.session_state.datetime_range_slider[1],  # selected stop time (inclusive)
        y1=(fig_layout.yaxis.range[1]- fig_layout.yaxis.range[0]) * 0.95 + fig_layout.yaxis.range[0],
        fillcolor="blue",
        opacity=0.1,
        layer="below",
    )

    # Mark labeled time range with red/orange/green background
    if "df_hr_accuracy_labels" in st.session_state:
        df_labels_this_activity = st.session_state.df_hr_accuracy_labels.query("activity_id == @st.session_state.activity_id_for_labeling")

        for index, row in df_labels_this_activity.iterrows():
            if row.hr_accuracy_in_segment == "perfect":
                color = "green"
            elif row.hr_accuracy_in_segment == "medium":
                color = "orange"
            elif row.hr_accuracy_in_segment == "wrong":
                color = "red"
            
            fig.add_shape(
                type="rect",
                x0=row.timestamp_start, # selected start time
                y0=fig_layout.yaxis.range[0],
                x1=row.timestamp_end,  # selected stop time (inclusive)
                y1=fig_layout.yaxis.range[1],
                fillcolor=color,
                opacity=0.1,
                layer="below",
            )
    return fig

def save_df_hr_accuracy_labels_as_csv(project_path: str = os.getcwd()):
    nrows = len(st.session_state.df_hr_accuracy_labels)

    str_datetime_now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    path_save_df_hr_accuracy_labels = os.path.join(project_path, "data","HR_Labels", f"hr_labels_{str_datetime_now}__{nrows}_rows.csv")
    st.session_state.df_hr_accuracy_labels.to_csv(path_save_df_hr_accuracy_labels, index=False)

def section_ui_to_add_manual_labels(df: pd.DataFrame, project_path: str = os.getcwd()):
    # Radio buttons to select hr_accuracy_in_segment ("perfect", "medium",  "wrong") and hr_accuracy_entire_activity("use_all", "use_none", "mixed")
    # Buttons to enter this select and append as row to st.session_state.df_hr_accuracy_labels
    # Uses st.session_state variables: activity_id, df_hr_accuracy_labels, datetime_range_slider

    col1, col2 = st.columns(2)
    # Add labels about entire activity
    with col1:
        with st.container(border=True):
            col1a, col1b = st.columns(2)
            with col1a:
                # Radio button for categorizing HR accuracy of entire activity
                st.radio("HR accuracy of the **entire activity**", ["use_all", "use_none"], key="radio_hr_accuracy_entire_activity", index=1,  format_func=lambda option: ":green[Use All] 😊" if option == "use_all" else ":red[Use None] 😔")
            with col1b:
                # Button for adding row to df about HR accruacy label in entire activity
                def button_add_row_to_df_labels_about_entire_activity_on_click():
                    if st.session_state.radio_hr_accuracy_entire_activity == "use_all":
                        hr_accuracy_in_segment = "perfect"
                    elif st.session_state.radio_hr_accuracy_entire_activity == "use_none":
                        hr_accuracy_in_segment = "wrong"
                    else:
                        raise ValueError(f"Unexpected value for st.session_state.radio_hr_accuracy_entire_activity: {st.session_state.radio_hr_accuracy_entire_activity}")

                    dict_row_hr_accuracy_labels = {
                        "activity_id": st.session_state.activity_id_for_labeling,
                        "hr_accuracy_in_segment": hr_accuracy_in_segment, # Automatically added since we did not save it for the entire activity. 
                        "hr_accuracy_entire_activity": st.session_state.radio_hr_accuracy_entire_activity,  # mixed <=> specify it for each timestamp individully. 
                        "timestamp_start": df.timestamp.iloc[0],
                        "timestamp_end": df.timestamp.iloc[-1],
                    }
                    # Append as new row
                    if len(st.session_state.df_hr_accuracy_labels) == 0:
                        st.session_state.df_hr_accuracy_labels = pd.DataFrame([dict_row_hr_accuracy_labels])
                    else:
                        st.session_state.df_hr_accuracy_labels = pd.concat([st.session_state.df_hr_accuracy_labels, pd.DataFrame([dict_row_hr_accuracy_labels])], ignore_index=True, axis=0)

                    # Save df_hr_accuracy_labels as csv file in repo
                    save_df_hr_accuracy_labels_as_csv(project_path=project_path)
                st.button("Save labels for the **entire activity** ", 
                        key="button_add_row_to_df_labels_about_entire_activity", 
                        on_click=button_add_row_to_df_labels_about_entire_activity_on_click
                )
            
    # Add labels about an activity segment only
    with col2:
        with st.container(border=True):
            col2a, col2b = st.columns(2)
            with col2a:
                # Radio button for categorizing HR accuracy in segment
                #st.radio("HR Accuracy in Segment", ["perfect", "medium", "wrong"], key="radio_hr_accuracy_in_segment", index=2)
                def radio_hr_accuracy_in_segment_format_func(option):
                    if option == "perfect":
                        return ":green[Perfect] 😊"
                    elif option == "medium":
                        return ":orange[Medium] 😐"
                    elif option == "wrong":
                        return ":red[Wrong] 😔"
                    else:
                        raise ValueError(f"Unexpected value for option: {option}")
                st.radio(
                    "HR Accuracy in Segment",
                    ["perfect", "medium", "wrong"],
                    index=2,
                    key="radio_hr_accuracy_in_segment",
                    format_func=radio_hr_accuracy_in_segment_format_func
                )
            with col2b:
                # Button for adding row to df about HR accruacy label in specified segment
                def button_add_row_to_df_labels_about_activity_segment_on_click():
                    dict_row_hr_accuracy_labels = {
                        "activity_id": st.session_state.activity_id_for_labeling,
                        "hr_accuracy_in_segment": st.session_state.radio_hr_accuracy_in_segment,
                        "hr_accuracy_entire_activity": "mixed",  # Automatically added since we did not save it for the entire activity. mixed <=> specify it for each timestamp individully. 
                        "timestamp_start": st.session_state.datetime_range_slider[0],
                        "timestamp_end": st.session_state.datetime_range_slider[1],
                    }
                    # Append as new row
                    if len(st.session_state.df_hr_accuracy_labels) == 0:
                        st.session_state.df_hr_accuracy_labels = pd.DataFrame([dict_row_hr_accuracy_labels])
                    else:
                        st.session_state.df_hr_accuracy_labels = pd.concat([st.session_state.df_hr_accuracy_labels, pd.DataFrame([dict_row_hr_accuracy_labels])], ignore_index=True, axis=0)

                    # Save df_hr_accuracy_labels as csv file in repo
                    save_df_hr_accuracy_labels_as_csv(project_path=project_path)
                st.button("Save labels for specified **segment** ", 
                        key="button_add_row_to_df_labels_about_activity_segment", 
                        on_click=button_add_row_to_df_labels_about_activity_segment_on_click
                )