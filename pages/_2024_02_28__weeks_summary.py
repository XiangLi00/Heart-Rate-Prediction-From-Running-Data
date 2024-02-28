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

from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report

# Append project path to system path
# print(f"Project root folder: {os.getcwd()}")
sys.path.append(os.getcwd())  


def page():

    df_weeks_summary = load_df_v2(
        table_name='weeks_summary',
        root_path_db=os.path.join(os.getcwd(), 'data'),
        sql_selected_columns="*",
        sql_condition=""
        )

    column_config_df_weeks_summary = {
        "inactive_hr_avg": st.column_config.NumberColumn(
            format="%.1f",
            ),
        "floors": st.column_config.NumberColumn(
            format="%d",
            ),
    }

    st.dataframe(
        df_weeks_summary, 
        column_config=column_config_df_weeks_summary)
    st.write(f"Shape: {df_weeks_summary.shape}")

    st.data_editor(df_weeks_summary)

    st.write("in the page() summmmmary file()")

def page3():
    st.write("inside summary.page3()")

st.write("helileeeeo moon!")
