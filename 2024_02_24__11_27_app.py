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
sys.path.append(project_root_folder)    # project root folder
sys.path.append(os.path.join(project_root_folder, 'streamlit'))    

# from streamlit._2024_02_24__11_28_monitoring_hr_different_plotting_libs import page
import abc

from utils.helper_load_df import load_df, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report
