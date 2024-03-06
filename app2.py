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
# print(f"Project root folder: {os.getcwd()}")
sys.path.append(os.getcwd())  

if False:
    # Choose page to display
    module_name = [
        '_2024_02_28__weeks_summary',
        '_2024_02_24__11_28_monitoring_hr_different_plotting_libs', '_2024_02_24__activities'][0]
    st.header(module_name)
    module = importlib.import_module("pages." + module_name)
    module.page()

from pages._2024_02_28__weeks_summary import page3
#from _2024_02_28__debug_streamlit_community import page3
page3()


st.write("app2.py: v1")