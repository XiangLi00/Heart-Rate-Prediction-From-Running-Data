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
# from streamlit_pages._2024_03_10_annotate_fit_helper import test1
from streamlit_pages import _2024_03_10_annotate_fit_helper
from utils import helper_load_fit_file, helper_load_specific_df, helper_pandas, helper_streamlit
# from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report

if False:
    def dataframe_with_selections(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "Select", False)

        # Get dataframe row-selections from user with st.data_editor
        edited_df = st.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns,
        )

        st.write(edited_df.Select)
        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df.Select]
        return selected_rows.drop('Select', axis=1)

    st.write(dataframe_with_selections(df))

# Section: show and filter running activities
df_activities, df_running_activities, df_running_activities_filtered = _2024_03_10_annotate_fit_helper.section_running_activities_show_and_filter()

# Section: Select specific running activity and show tabular information
df = _2024_03_10_annotate_fit_helper.section_select_activity_and_retrieve_df(df_activities)

# View df
st.dataframe(df)
st.write("df columns: " + str(list(df.columns)))


# st.write(f"annotate_fit – current working directory: {os.getcwd()}")

_2024_03_10_annotate_fit_helper.section_show_plotly_timeseries_plot_v3(df)

# seaborn scatterplot of df columns distance vs elevation_change. color by imputed. figsize=(10, 5)
# fig_seaborn, ax = plt.subplots(figsize=(10, 5))
fig = plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='distance', y='elevation_change')  # , hue='imputed')
st.pyplot(fig)

