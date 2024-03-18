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
sys.path.append(os.path.join(os.getcwd(), 'streamlit_pages'))  
# from streamlit_pages._2024_03_18_annotate_fit_helper_v2 import test1
from streamlit_pages import _2024_03_18_annotate_fit_helper_v2
from utils import helper_load_fit_file_v1, helper_load_specific_df, helper_pandas, helper_streamlit
# from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report

project_path = os.getcwd()


def do_stuff_on_page_load():
    st.set_page_config(layout="wide")
    st.session_state.screen_height, st.session_state.screen_width = helper_streamlit.get_screen_height_and_width()  # retrieve screen heigh, width in pixels
    if not "columns_df_hr_accuracy_labels" in st.session_state:
        st.session_state.columns_df_hr_accuracy_labels = ["activity_id", "hr_accuracy_in_segment", "hr_accuracy_entire_activity", "timestamp_start", "timestamp_end"]
do_stuff_on_page_load()



# Section: show and filter running activities
df_activities, df_running_activities, df_running_activities_filtered = _2024_03_18_annotate_fit_helper_v2.section_running_activities_show_and_filter(project_path=project_path)


# Section: Select specific running activity and show tabular information
df = _2024_03_18_annotate_fit_helper_v2.section_select_activity_and_retrieve_df(
    df_activities,
    df_loading_method = ["load_fit_file_v1", "activity_init"][1])

# Show df information
st.write("df columns: " + str(list(df.columns)))

# Create df_hr_accuracy_labels to store manual labels 
# Each row ^= 1 time range for which we specify if HR is trustworthy
# Colums
# - activity_id
# - timestamp_start: start of labeled segment. (pd.Timestamp)
# - timestamp_end: end of labeled segment. (pd.Timestamp)
# - hr_accuracy_in_segment: "perfect", "medium", "wrong"
# - hr_accuracy_entire_activity: "use_all", "use_none", "mixed"
# -- "mixed" ^= specify it for each timestamp individully.
# -- hr_accuracy_in_segment contains more information than hr_accuracy_entire_activity
# -- However, we still keep entire_activty information. Reason: can easily filter for all activities with "use_all" or "use_none". Maybe don't even load activities with use_none 
# Small limitation: does not check for overlapping labels. Will need to be handled later. e.g. be conservtive
# Option 1: Upload already existing labels and continue from there
_2024_03_18_annotate_fit_helper_v2.section_upload_existing_df_hr_accuracy_labels()
# Option 2: If no file was uploaded, create empty df
if "df_hr_accuracy_labels" not in st.session_state:
    st.session_state.df_hr_accuracy_labels = pd.DataFrame(columns=st.session_state.columns_df_hr_accuracy_labels)


# Display df_hr_accuracy_labels
st.data_editor(st.session_state.df_hr_accuracy_labels.sort_index(ascending=False))
if "df_hr_accuracy_labels" not in st.session_state:
    st.session_state.csv_hr_accuracy_labels 

# Retrieve fig to be displayed
fig = _2024_03_18_annotate_fit_helper_v2.section_get_plotly_timeseries_fig_v4(df, activity_id=st.session_state.activity_id, plot_fig=True)

# Add datetime slider widget and color fig background (of selected/labeled time ranges)
fig = _2024_03_18_annotate_fit_helper_v2.section_slider_datetime_selection_and_add_colored_background_to_fig(fig=fig, df=df)


# Show ui to add and save labels
_2024_03_18_annotate_fit_helper_v2.section_ui_to_add_manual_labels(df=df, project_path=project_path)


# Plot figure with labels
st.checkbox("Show Plot with labels", value=False, key="checkbox_show_plotly_with_labels")
if st.session_state.checkbox_show_plotly_with_labels:
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

st.write(st.session_state)