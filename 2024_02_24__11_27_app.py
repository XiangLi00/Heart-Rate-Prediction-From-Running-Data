from datetime import datetime, timedelta
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

# Get project root folder
project_root_folder = os.getcwd() # r'd:\OneDrive\7Temporary\Coding\2024_02_20_Garmin'
print(f"project_root_folder: {project_root_folder}")
sys.path.append(project_root_folder)    # project root folder
#sys.path.append(os.path.join(project_root_folder, 'streamlit'))    

st.title("Empty")

# from streamlit._2024_02_24__11_28_monitoring_hr_different_plotting_libs import page
import bcd
from bcd import page

from utils.helper_load_df import load_df, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report

print(os.listdir(os.path.join(project_root_folder)))



import abc
