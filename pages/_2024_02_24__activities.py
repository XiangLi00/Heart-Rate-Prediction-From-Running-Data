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
import plotly.express as px
import seaborn as sns
import streamlit as st
# from ydata_profiling import ProfileReport

from utils.helper_load_df import load_df, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report

# Append project path to system path
print(f"Project root folder: {os.getcwd()}")
sys.path.append(os.getcwd())  


def page():
    """
        table: monitoring_hr

        First version, trying out plotly, altair, vega-lite, and bokeh.
        
    """

    st.header("_2024_02_24__activities")

    df_activities = load_df('garmin_monitoring.db', 'monitoring_hr', root_path_db=os.path.join(os.getcwd(), 'data'))

    st.dataframe(df_activities)

