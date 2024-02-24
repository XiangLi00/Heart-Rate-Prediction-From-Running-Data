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
df_monitoring = load_df('garmin_monitoring.db', 'monitoring_hr', root_path_db=root_path_db).tail(1000000)

if False:
    st.dataframe(df_monitoring.head(100))

    st.write(df_monitoring.head(1000))

    st.write("Streamlit line chart")
    st.line_chart(df_monitoring["heart_rate"].head(1000))

"""Plotly Plot"""
st.write("Plotly Plot")

num_points_displayed = st.number_input("Number of points to display", min_value=1, value=1000, step=1)
fig1 = px.line(df_monitoring.tail(num_points_displayed), x="timestamp", y="heart_rate",
                 title="Plotly title")
#fig1.update_layout(editable=False)
# Add x-axis range slider
fig1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
fig1.update_yaxes(fixedrange=True)  # Lock the y-axis
fig1.update_layout(
    dragmode='pan',  # zoom, pan, select, lasso
)
config = {'scrollZoom': True,
          'displaylogo': False}
st.plotly_chart(fig1, use_container_width=True, config=config)

"""Altair Plot"""
st.scatter_chart(data=df_monitoring.tail(1000), x="timestamp", y="heart_rate")


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
st.vega_lite_chart(df_monitoring.tail(1000), spec=spec, use_container_width=True)


"""Bokeh Plot"""
from bokeh import plotting

# Create a new plot with a title and axis labels
p = bokeh.plotting.figure(title="A line plot showing heart rate over time.",
           x_axis_label='Timestamp', 
           y_axis_label='Heart Rate',
           x_axis_type='datetime', # Assuming 'timestamp' is in datetime format
           tools="") # Start with no tools, add them as needed

# Add a line renderer
p.line(x='timestamp', y='heart_rate', source=df_monitoring.tail(10000))

# Add Pan and Wheel Zoom tools
p.add_tools(bokeh.models.PanTool(dimensions="width"))  # Restrict to horizontal pan
p.add_tools(bokeh.models.WheelZoomTool(dimensions="width"))  # Restrict to horizontal zoom
p.add_tools(bokeh.models.BoxZoomTool(dimensions="width"))

# Show the results in Streamlit
st.bokeh_chart(p, use_container_width=True)


"""Plotly Filtered"""
# Function to filter data based on visible range
def filter_data_by_range(df, x_range):
    start, end = x_range
    filtered_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    return filtered_df

# Streamlit app
st.title("Plotly Filtering Example")

# Number of points input
num_points_displayed = st.number_input("Number of points to display", min_value=1, value=1000, step=1, key=2)

# Create Plotly figure
fig1 = px.line(df_monitoring.tail(num_points_displayed), x="timestamp", y="heart_rate", title="Plotly title")

# Add x-axis range slider
fig1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

# Update y-axis and layout
fig1.update_yaxes(fixedrange=True)  # Lock the y-axis
fig1.update_layout(dragmode='pan')  # Enable pan

# Display Plotly chart
plotly_chart = st.plotly_chart(fig1, use_container_width=True)

# Get the x-axis range from the Plotly chart
x_range = st.slider("Select X-axis range", min_value=df_monitoring['timestamp'].min(),
                    max_value=df_monitoring['timestamp'].max(), value=(df_monitoring['timestamp'].min(), df_monitoring['timestamp'].max()))

# Filter data based on the visible range
filtered_data = filter_data_by_range(df_monitoring, x_range)

# Update Plotly chart with filtered data
fig1 = px.line(filtered_data.tail(num_points_displayed), x="timestamp", y="heart_rate", title="Plotly title")
fig1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
fig1.update_yaxes(fixedrange=True)  # Lock the y-axis
fig1.update_layout(dragmode='pan')  # Enable pan

# Display updated Plotly chart
plotly_chart.plotly_chart(fig1, use_container_width=True)