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


# Append project path to system path
# print(f"Project root folder: {os.getcwd()}")
sys.path.append(os.getcwd())  

action = 1
if action == 1:
    # Choose page to display
    module_name = [
        '_2024_02_28__weeks_summary',
        '_2024_02_24__11_28_monitoring_hr_different_plotting_libs', 
        '_2024_02_24__activities'
        ][-1]
    st.header(module_name)
    module = importlib.import_module("streamlit_pages." + module_name)
    module.page2()
if action == 2:
    from streamlit_pages._2024_02_28__weeks_summary import page3
    #from _2024_02_28__debug_streamlit_community import page3
    page3()


#st.write("app2.py: v4")