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

action = 3

if action == 1:
    """
    – Prints a specific function in the.py page 
    """
    from streamlit_pages._2024_02_28__weeks_summary import page3
    #from _2024_02_28__debug_streamlit_community import page3
    page3()

if action == 2:
    """
    – Prints all of the .py page 
    """
    # Choose page to display
    module_name = [
        '_2024_02_28__weeks_summary',
        '_2024_02_24__11_28_monitoring_hr_different_plotting_libs', 
        '_2024_02_24__activities'
        ][-1]
    

    st.header(module_name)

    # Shows the content of this streamlit page (just by importing it)
    module = importlib.import_module("streamlit_pages." + module_name) # fancy way of writing "import streamlit_pages._2024_02_28__weeks_summary"

if action == 3:
    """
        Dropdown menu to select the page to display
    """
    folder_path = "streamlit_pages"
    # Retrieve all .py files in folder
    file_names_of_streamlit_pages_with_py_extension = os.listdir(folder_path)
    # Remove extionsion. E.g. now get ['_2024_02_24__11_28_monitoring_hr_different_plotting_libs', '_2024_02_24__activities', '_2024_02_28__weeks_summary']
    file_names_of_streamlit_pages = [os.path.splitext(filename)[0] for filename in file_names_of_streamlit_pages_with_py_extension if filename.endswith(".py")]

    streamlit_page_displayed = st.selectbox("Select page to display", file_names_of_streamlit_pages)

    importlib.import_module("streamlit_pages." + streamlit_page_displayed) 


#st.write("app2.py: v4")