from datetime import datetime, timedelta
import importlib
import os
import sqlite3
import sys
import time

import fitdecode  # for parsing fit into csv
import numpy as np
import pandas as pd

import helper_pandas

def _process_df_from_fit(df_activity):
    ## rename columns 
    df_activity = df_activity.rename(columns={'enhanced_speed': 'speed', 
                                              'enhanced_altitude': 'altitude', 
                                              'heart_rate': 'hr', 
                                              'accumulated_power': 'cum_power',
                                              'distance': 'cum_distance'})

    ## Unit conversion
    df_activity['speed'] = df_activity['speed']*3.6  # m/s to km/h

    ## Add new columns
    df_activity['steps_per_min'] = (df_activity['cadence'] + df_activity['fractional_cadence']) * 2  # steps per minute
    df_activity['power100'] = np.maximum(0, df_activity['power'] - 100)  # Motivation 100W running and 0W are similarly exhausting

    # Add column pace
    if False:
        df_activity.insert(5, 'pace', 60 / (df_activity['speed'] )) 
        # convert inf to nan to avoid "FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
        df_activity['pace'] = df_activity['pace'].replace([np.inf, -np.inf], np.nan)

    ## Drop columns
    df_activity = df_activity.drop(columns=['stance_time_percent', 'stance_time_balance', 'fractional_cadence', 'cadence'], errors = 'ignore')

    ## Reorder columns
    df_activity = helper_pandas.move_columns_to_end(df_activity, ['position_lat', 'position_long'])

    return df_activity


def _impute_df_from_fit(df_activity):

    # Add column. imputed=True ^= entirue row was imputed
    df_activity["imputed"] = False

    df_activity = df_activity.set_index('timestamp').resample('s').asfreq().reset_index()

    # Set boolean columns to boolean type (Reason: fillna gives wornang otherwise)
    df_activity['imputed'] = df_activity['imputed'].astype(bool)


    # Apply imputation method 1: Impute with fixed values
    dict_column_and_fixed_imputed_value = {
        "imputed": True,
        "power": 0,
        "speed": 0,
        "step_length": 0,
        "activity_type": "pause",
        "steps_per_min": 0,
        "power100": 0,
    }
    # Assert that the this columns exist in df_activity
    assert set(dict_column_and_fixed_imputed_value.keys()).issubset(set(df_activity.columns)), f"The columns {str(set(dict_column_and_fixed_imputed_value.keys())- set(df_activity.columns))} to be imputed with fixed values are not present in the dataframe. Existing columns are {list(df_activity.columns)}."
    # Impute
    df_activity = df_activity.fillna(dict_column_and_fixed_imputed_value)

    # Apply imputation method 2: linear interpolation
    columns_to_impute_linearly = [
        "cum_distance", "altitude", "hr", "cum_power", "position_lat", "position_long"
        ]
    # Assert that the this columns exist in df_activity
    assert set(columns_to_impute_linearly).issubset(set(df_activity.columns)), f"The columns {set(columns_to_impute_linearly)- set(df_activity.columns)} do not exist in the dataframe. Existing columns are {list(df_activity.columns)}."
    # Impute linearly
    for column in columns_to_impute_linearly:
        df_activity[column] = df_activity[column].interpolate(method='linear')

    # Apply imputation method 3: backward fill
    columns_to_impute_via_backwards_fill = [
        "cum_power", 
        ]
    # Assert that the this columns exist in df_activity
    assert set(columns_to_impute_via_backwards_fill).issubset(set(df_activity.columns)), f"The columns {set(columns_to_impute_via_backwards_fill)- set(df_activity.columns)} do not exist in the dataframe. Existing columns are {list(df_activity.columns)}."
    # Impute linearly
    for column in columns_to_impute_via_backwards_fill:
        df_activity[column] = df_activity[column].bfill()

    if False:
        # Special case imputation: column "cum_power" – first row - impute NaN wit 0
        if pd.isna(df_activity.loc[df_activity.index[0], 'cum_power']):
            df_activity.loc[df_activity.index[0], 'cum_power'] = 0

    
    # Assert that most columns are fully imputed
    columns_not_requiring_imputation = {"vertical_oscillation", "stance_time", "vertical_ratio"}
    columns_requiring_imputation = list(set(df_activity.columns) - columns_not_requiring_imputation)
    ser_nans_per_column = df_activity[columns_requiring_imputation].isna().sum()
    ser_nans_per_column_positive = ser_nans_per_column[ser_nans_per_column > 0]
    assert len(ser_nans_per_column_positive) == 0, f"Imputation failed. The following columns still contain NaNs: {ser_nans_per_column_positive}"


    return df_activity


def _load_raw_df_from_fit(path_fit_file: str, frame_name: str = 'record', lat_long_update: bool = True, debug: bool = False) -> pd.DataFrame:
    """
    Decodes a .FIT file and returns the data as a pandas DataFrame.

    Parameters:
    - path_fit_file (str): The path to the .FIT file.
    - frame_name (str): The name of the frame to extract data from. Defaults to 'record'. E.g. device_info, gps_metadata, record, file_id, event, unknown_288,
    - lat_long_update (bool): Whether to update the latitude and longitude columns to standard degrees. Defaults to True.
    - debug (bool): Whether to print the list of frames in the file for debugging purposes. Defaults to False.

    Returns:
    - df_activity (pd.DataFrame): The decoded data as a pandas DataFrame.
    """
    # Initialize some useful variables for the loops
    check_list = good_list = []
    list_check = {}
    df_activity = pd.DataFrame()

    # list of all rows. Will be merged to a dataframe at the end
    list_rows = []

    # Open the file with fitdecode
    with fitdecode.FitReader(path_fit_file) as file:
        
        # Iterate through the .FIT frames
        for frame in file:

            # Procede if the frame object is the correct data type
            if isinstance(frame, fitdecode.records.FitDataMessage): # Remark: the type FitDefinitionMessage also exists
                
                # Add the frames and their corresponding counts to a dictionary for debugging
                if frame.name not in check_list:
                    check_list.append(frame.name)
                    list_check[frame.name] = 1
                else:
                    list_check.update({frame.name: list_check.get(frame.name) + 1})
                
                # If the current frame is a record, we'll reset the row_dict variable
                # and add the field values for all fields in the good_list variable
                if frame.name == frame_name:
                    row_dict = {}
                    for field in frame.fields: 
                        if field.name.find('unknown') < 0:
                            if field.name not in good_list and field.name.find('unknown') < 0:
                                good_list.append(field.name)
                            row_dict[field.name] = frame.get_value(field.name)
                    
                    list_rows.append(row_dict)

        df_activity = pd.DataFrame(list_rows)
    
        # Update the Long/Lat columns to standard degrees
        if lat_long_update:
            for column in ['position_lat', 'position_long']:
                df_activity[column] = df_activity[column].apply(lambda x: x / ((2**32)/360))
        
        # If you want to check to see which frames are in the file, print the list_check variable
        if debug:
            print(list_check)

    return df_activity


def load_fit_file(fit_file_path: str) -> pd.DataFrame:
    df_activity = _load_raw_df_from_fit(fit_file_path)
    df_activity = _process_df_from_fit(df_activity)
    df_activity = _impute_df_from_fit(df_activity)
    
    return df_activity

if False:
    activity_id = 14057922527
    project_path = r'D:\OneDrive\7Temporary\Coding\2024_02_20_Garmin'

    path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{activity_id}_ACTIVITY.fit')
    df_activity = load_fit_file(path_fit_file)