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

"""Plotly Plot"""
st.write("Plotly Plot")
st.empty()
fig1 = px.line(df_monitoring.tail(1000), x="timestamp", y="heart_rate",
                 title="Plotly title")
#fig1.update_layout(editable=False)
# Add x-axis range slider
fig1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
fig1.update_yaxes(fixedrange=True)  # Lock the y-axis
fig1.update_layout(
    dragmode='zoom',  # zoom, pan, select, lasso
    uirevision="foo",
)
st.plotly_chart(fig1, use_container_width=True)

"""Altair Plot"""
st.scatter_chart(data=df_monitoring.tail(10000), x="timestamp", y="heart_rate")


"""Vega Lite Plot"""
# Convert DataFrame to JSON
# data_json = df_monitoring.to_dict(orient='records')

# Vega-Lite spec
spec = {
    "description": "A scatterplot showing heart rate over time.",
    "mark": "point",
    "encoding": {
        "x": {"field": "timestamp", "type": "temporal", "title": "Timestamp"},
        "y": {"field": "heart_rate", "type": "quantitative", "title": "Heart Rate"}
    },
    "selection": {
        "x_zoom_pan": {
            "type": "interval",
            "bind": "scales",
            "encodings": ["x"]
        }
    }
}
st.vega_lite_chart(df_monitoring, spec=spec, use_container_width=True)


"""Bokeh Plot"""
from bokeh import plotting
# Disable the default browser zoom behavior by adding the following CSS to the Streamlit app:
st.markdown("""
<style>
body {
    zoom: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# Create a new plot with a title and axis labels
p = bokeh.plotting.figure(title="A line plot showing heart rate over time.",
           x_axis_label='Timestamp', 
           y_axis_label='Heart Rate',
           x_axis_type='datetime', # Assuming 'timestamp' is in datetime format
           tools="") # Start with no tools, add them as needed

# Add a line renderer
p.line(x='timestamp', y='heart_rate', source=df_monitoring)

# Add Pan and Wheel Zoom tools
p.add_tools(bokeh.models.PanTool(dimensions="width"))  # Restrict to horizontal pan
p.add_tools(bokeh.models.WheelZoomTool(dimensions="width"))  # Restrict to horizontal zoom
p.add_tools(bokeh.models.BoxZoomTool(dimensions="width"))

# Show the results in Streamlit
st.bokeh_chart(p, use_container_width=True)
