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
    # Make sure that page elements use the full page width
    st.set_page_config(layout="wide")
    
    # Retrieve screen heigh, width in pixels
    st.session_state.screen_height, st.session_state.screen_width = helper_streamlit.get_screen_height_and_width()  
do_stuff_on_page_load()





st.write(st.session_state)