from datetime import datetime, timedelta
import os
import sqlite3
import sys
import time


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
# from ydata_profiling import ProfileReport

project_root_folder = os.getcwd() # r'd:\OneDrive\7Temporary\Coding\2024_02_20_Garmin'
sys.path.append(project_root_folder)    # project root folder
from utils.helper_load_df import load_df, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report


st.title("Title")
st.header("Header")
st.subheader("Subheader")
st.caption("Caption")
st.text("Text")
st.write("Hello World!")

root_path_db = os.path.join(project_root_folder, 'data')
print(root_path_db)
df_monitoring = load_df('garmin_monitoring.db', 'monitoring_hr', root_path_db=root_path_db).tail(10000)

if False:
    st.dataframe(df_monitoring.head(100))

    st.write(df_monitoring.head(1000))

    st.write("Streamlit line chart")
    st.line_chart(df_monitoring["heart_rate"].head(1000))

st.write("Plotly Plot")
fig1 = px.line(df_monitoring, x="timestamp", y="heart_rate",
                 title="Plotly title")
#fig1.update_layout(editable=False)
# Add x-axis range slider
fig1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
fig1.update_yaxes(fixedrange=True)  # Lock the y-axis
fig1.update_layout(
    dragmode='zoom',  # zoom, pan, select, lasso
)
st.plotly_chart(fig1, use_container_width=True, update_mode='transform')