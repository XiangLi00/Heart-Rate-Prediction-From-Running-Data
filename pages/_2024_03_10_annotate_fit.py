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

from utils import helper_load_fit_file, helper_pandas
from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report


project_path = os.getcwd()

list_activity_ids = [13738732552, 14057922527]
activity_id = st.selectbox("Select activity id", list_activity_ids)
st.write(activity_id)

path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{activity_id}_ACTIVITY.fit')
#df = helper_load_fit_file.load_fit_file(path_fit_file)

#st.dataframe(df)