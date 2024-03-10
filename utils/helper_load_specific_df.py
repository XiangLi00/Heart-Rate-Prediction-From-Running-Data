from datetime import datetime, timedelta
import importlib
import os
import sqlite3
import sys
import time

import fitdecode  # for parsing fit into csv
import numpy as np
import pandas as pd
import streamlit as st

import helper_pandas
import helper_load_df

@st.cache_data
def load_df_activities(root_path_db: str) -> pd.DataFrame:
    """
    Load the activities table from the database.

    Args:
        root_path_db (str): The root path of the database.

    Returns:
        pd.DataFrame: The activities table.
    """
    df = helper_load_df.load_df_v2(
        table_name='activities',
        root_path_db=root_path_db
        )

    # Reorder columns
    columns_desired_order = ["activity_id", 'start_time', 'sport', 'sub_sport',  'moving_time', 'distance', 'avg_hr', 'avg_speed', 'training_load', 'training_effect', 'anaerobic_training_effect',  'ascent', 'descent']
    df = helper_pandas.reorder_columns(df=df, columns_desired_order=columns_desired_order)

    # Sort by start_time
    df = df.sort_values(by='start_time', ascending=False)

    # drop the columns 'max_temperature', 'min_temperature', 'avg_temperature' if they exist. Otherwise, an informative error will be raised.
    columns_to_drop = ['avg_rr', 'max_rr', 'description',
       'type', 'course_id', 'laps', 'device_serial_number', 'self_eval_feel',
       'self_eval_effort', 'max_temperature', 'min_temperature', 'avg_temperature', 'hr_zones_method',
       'hrz_1_hr', 'hrz_2_hr', 'hrz_3_hr', 'hrz_4_hr', 'hrz_5_hr']
    df = df.drop(columns=columns_to_drop, errors='raise')

    return df

    # All original columns ['activity_id', 'start_time', 'sport', 'sub_sport', 'moving_time','distance', 'avg_hr', 'avg_speed', 'training_load', 'training_effect',    'anaerobic_training_effect', 'ascent', 'descent', 'name', 'description',    'type', 'course_id', 'laps', 'device_serial_number', 'self_eval_feel',   'self_eval_effort', 'stop_time', 'elapsed_time', 'cycles', 'max_hr',    'avg_rr', 'max_rr', 'calories', 'avg_cadence', 'max_cadence',    'max_speed', 'max_temperature', 'min_temperature', 'avg_temperature',    'start_lat', 'start_long', 'stop_lat', 'stop_long', 'hr_zones_method',    'hrz_1_hr', 'hrz_2_hr', 'hrz_3_hr', 'hrz_4_hr', 'hrz_5_hr',    'hrz_1_time', 'hrz_2_time', 'hrz_3_time', 'hrz_4_time', 'hrz_5_time'],