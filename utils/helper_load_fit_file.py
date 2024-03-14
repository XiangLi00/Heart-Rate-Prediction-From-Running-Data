from datetime import datetime, timedelta
import importlib
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import fitdecode  # for parsing fit into csv
import numpy as np
import pandas as pd
import streamlit as st

from utils import helper_compute_features
import helper_pandas

# –––––––– Start adding grade column ––––––––––
def _get_first_last_row_indices_for_grade_calc(current_row_index: int, current_cum_distance: float, df: pd.DataFrame, delta_distance_for_grade_computation: float = 10) -> Tuple[Any, Any]:
    """
    Calculates the first and last row indices for grade computation. ("Sekantensteigung")

    Suppose delta_distance_for_grade_computation = 10 [m]
    Idea: Compute grade based on now and >= 10 meters ago (i.e. the first entry >= 10 meters ago). 
    Edge cases:
    - If we just started, then we use meters [0, 10]
    - If we are at the end of the activity, then we use meters [last cum_distance - 10, last cum_distance]
    
    Parameters:
    current_row_index (int): The index of the current row.
    current_cum_distance (float): The cumulative distance at the current row.
    df (pd.DataFrame): The dataframe containing the data.
    delta_distance_for_grade_computation (float): The minimal delta distance [m] used for grade computation. Default is 10. Motivation: Make sure that the grade is not too noisy for short distances.

    Returns:
    Tuple[int, int]: A tuple containing the first and last row indices for grade computation.
    """
    
    ## Compute first_row_index ^= usually the row >= 10 meters ago
    first_row_index = helper_pandas.get_previous_row_index(
        df["cum_distance"].searchsorted(
            current_cum_distance-delta_distance_for_grade_computation
        ),
        df=df
    )
    if False:
        # This code is not robust. It assumes that index is a range of consecutive integers
        first_row_index = np.maximum(
            df["cum_distance"].searchsorted(current_cum_distance-delta_distance_for_grade_computation) - 1,
            df.index[0]  # ^= first row
        )

    ## Computes the last_row_index ^= usually the current row
    first_cum_distance = df.loc[df.index[0], "cum_distance"]
    last_cum_distance = df.loc[df.index[-1], "cum_distance"]
    # Case: we are at the last 10 meters of the activity. Then we want to use meters [last cum_distance - 10, last cum_distance]
    if current_cum_distance > last_cum_distance - delta_distance_for_grade_computation:
        last_row_index = df.index[-1] # ^= last row of activity
    # Case: We are at the first 10 meters of the activity. Then we want to use meters [0, 10]
    elif current_cum_distance <= first_cum_distance + delta_distance_for_grade_computation:
        last_row_index = df["cum_distance"].searchsorted(
                current_cum_distance+delta_distance_for_grade_computation
            )
    # Normal case
    else:
        last_row_index = current_row_index
    return (first_row_index, last_row_index)


def _add_grade_column(df: pd.DataFrame, delta_distance_for_grade_computation: float = 10, add_debug_columns: bool = False) -> pd.DataFrame:
    """
    Adds a column to the dataframe containing the grade (of the last n meters)

    Details
    - Ignores elevation changes during breaks (i.e. when imputed == True)

    Flaws:
    - The start of an uphill interval still uses the (often negative) grade of the last interval. So it reacts a bit slowly.

    Call functions via: 
    - df, grade_col_name = _add_grade_column(df, delta_distance_for_grade_computation=10, add_debug_columns=False)



    Args:
        df (pd.DataFrame): The dataframe.
        delta_distance_for_grade_computation (float, optional): The minimal delta distance [m] used for grade computation. Default is 10. Motivation: Make sure that the grade is not too noisy for short distances.
        add_debug_columns (bool, optional): If True, then additional intermediate columns used for grade computation are added to the dataframe. Default is False.

    Returns:
        pd.DataFrame: The dataframe with the added grade_last_10m column.
    """

    assert df['cum_distance'].is_monotonic_increasing, "df.cum_distance (of fit file) is not monotonically increasing"
    assert delta_distance_for_grade_computation > 0, f"delta_distance_for_grade_computation {delta_distance_for_grade_computation} has to be positive"

    # Add columns containing information about which 2 rows to use for grade computation
    # Why need lambda function: Because some parameters should not be vectorized
    df["first_row_index_for_grade_calc"], df["last_row_index_for_grade_calc"] = \
        np.vectorize(
            lambda current_row_index, current_cum_distance: 
                _get_first_last_row_indices_for_grade_calc(
                    current_row_index=current_row_index,
                    current_cum_distance=current_cum_distance,
                    delta_distance_for_grade_computation=delta_distance_for_grade_computation,
                    df=df
                )
        )(
            current_row_index=df.index,
            current_cum_distance=df["cum_distance"]        
        )

    # Retrieve the elevation and cum_distance for the 2 given rows/times
    # Remark: need to convert to np.array or list in order to avoid the new irrelevant index causing issues
    df["first_elevation_for_grade_calc"] = np.array(df["elevation"].loc[df["first_row_index_for_grade_calc"]])
    df["first_cum_distance_for_grade_calc"] = np.array(df["cum_distance"].loc[df["first_row_index_for_grade_calc"]])
    df["last_elevation_for_grade_calc"] = np.array(df["elevation"].loc[df["last_row_index_for_grade_calc"]])
    df["last_cum_distance_for_grade_calc"] = np.array(df["cum_distance"].loc[df["last_row_index_for_grade_calc"]])
    df["cum_distance_difference_for_grade_calc"] = df["last_cum_distance_for_grade_calc"] - df["first_cum_distance_for_grade_calc"]

    # Assert that Sekantensteigung is taken for distances >= delta_distance_for_grade_computation [m] (as intended)
    df_cum_distance_difference_too_low = df.query("cum_distance_difference_for_grade_calc < @delta_distance_for_grade_computation")
    if len(df_cum_distance_difference_too_low) > 0:
        debug_columns_of_interest = ["timestamp", "cum_distance","cum_distance_difference_for_grade_calc", "first_cum_distance_for_grade_calc", "last_cum_distance_for_grade_calc",  "first_row_index_for_grade_calc", "last_row_index_for_grade_calc"]
        raise AssertionError(f"cum_distance_difference_for_grade_calc is too low (i.e. < {delta_distance_for_grade_computation} [m]) for the following rows: {df_cum_distance_difference_too_low.index} {df_cum_distance_difference_too_low[debug_columns_of_interest]}")

    # Grade in %
    grade_col_name = f'grade_last_{int(round(delta_distance_for_grade_computation))}m'  # e.g. "grade_last_10m"
    df[grade_col_name] = (df["last_elevation_for_grade_calc"] - df["first_elevation_for_grade_calc"]) / df["cum_distance_difference_for_grade_calc"]  * 100

    if not add_debug_columns:
        df = df.drop(columns = ['first_row_index_for_grade_calc', 'last_row_index_for_grade_calc', 'first_elevation_for_grade_calc', 'first_cum_distance_for_grade_calc', 'last_elevation_for_grade_calc', 'last_cum_distance_for_grade_calc', 'cum_distance_difference_for_grade_calc'])

    return df, grade_col_name

# –––––––– Stop adding grade column ––––––––––


def _rename_columns_and_convert_units(df):
    # rename columns 
    df = df.rename(columns={'enhanced_speed': 'speed', 
                                              'enhanced_altitude': 'elevation', 
                                              'heart_rate': 'hr', 
                                              'accumulated_power': 'cum_power',
                                              'distance': 'cum_distance'})

    # Unit conversion
    df['speed'] = df['speed']*3.6  # m/s to km/h

    return df

def _add_columns(df):
    """_summary_
    – Remark: Whenever I ad a neew colun, I'll need to specify in _impute_df_from_fit if it needs to be imputed. If yes, how
    """
    df['steps_per_min'] = (df['cadence'] + df['fractional_cadence']) * 2  # steps per minute
    df['power100'] = np.maximum(0, df['power'] - 100)  # Motivation 100W running and 0W are similarly exhausting

    # Remark: These should not have any nans
    df['distance'] = df['cum_distance'].diff().fillna(0)
    df['elevation_change_raw'] = df['elevation'].diff().fillna(0)
    # Setting elevation_change=0 for imputed rows (^= paused times). Reason: Otherwise grade computation just after break is very off
    df["elevation_change"] = (~df["imputed"]).astype(int) * df["elevation_change_raw"]
    # cum_elevation_change is similar to elevation if there were no breaks
    df['cum_elevation_change'] = df['elevation_change'].cumsum()
    df['ascent'] = np.maximum(0, df['elevation_change'])
    df['descent'] = np.abs(np.maximum(0, -df['elevation_change']))  # np.abs to have 0.0 instead of -0.0

    if False:
        # Compute grade and uphill_grade
        # Remark: They should not have any nans
        df["grade"] = np.where(df["distance"] == 0, 0, df["elevation_change_raw"] / df["distance"] * 100)  # Set it to 0 if distance is 0. Not good because sensitive to small distance changes between different rows/seconds
        df["uphill_grade"] = np.where(df["distance"] == 0, 0, df["ascent"] / df["distance"] * 100)
    df, grade_col_name = _add_grade_column(df, delta_distance_for_grade_computation=20, add_debug_columns=False) # grade_last_10m

    # Add exponential weighted moving average columns
    variable_to_smoothen_exponentially = ["speed", "power100", grade_col_name]
    list_ew_spans_in_s = [10, 120]
    for variable in variable_to_smoothen_exponentially:
        for ew_span in list_ew_spans_in_s:
            df[f"{variable}_ew_{ew_span}s"] = df[variable].ewm(span=ew_span).mean()
    # Add gaspeed and gaspeed4 columns (exponentially weighted)
    for ew_span in list_ew_spans_in_s:
        col_name_grade= grade_col_name # f"grade_ew_{ew_span}s"
        col_name_speed = f"speed_ew_{ew_span}s"
        col_name_gaspeed = f"gaspeed_ew_{ew_span}s"
        col_name_gaspeed4 = f"gaspeed4_ew_{ew_span}s"

        df[col_name_gaspeed] = df[col_name_speed] * helper_compute_features.get_spline_for_gap_computation()(df[col_name_grade])  # calling get_spline_for_gap_computation() each time might be inefficient
        df[col_name_gaspeed4] = np.maximum(0, df[col_name_gaspeed] - 4)  # Motivation 4km/h running/walking and standing still are similarly exhasting
    if False:
        # For grade/uphill_grade: add exponential weighted moving average column
        df[f"grade_ew_120s"] = df["grade"].ewm(span=120).mean()  # Intuition: If I spent most of the time(!) at 10%, then this is ~10%
        df[f"uphill_grade_ew_120s"] = df["uphill_grade"].ewm(span=120).mean()  # Intuition: If I spent most of the time(!) at 10%, then this is ~10%

    # Add column pace
    if False:
        df.insert(5, 'pace', 60 / (df['speed'] )) 
        # convert inf to nan to avoid "FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
        df['pace'] = df['pace'].replace([np.inf, -np.inf], np.nan)
    return df

def _drop_and_reorder_columns(df):
    # Drop columns
    df = df.drop(columns=['stance_time_percent', 'stance_time_balance', 'fractional_cadence', 'cadence'], errors = 'ignore')

    # Reorder columns
    df = helper_pandas.move_columns_to_end(df, ['position_lat', 'position_long'])

    return df


def _impute_df_from_fit(df):
    """
    Imputes most columns (with fixed value, linearly or backward fill)

    – Some columns don't require imputation (columns_not_requiring_imputation)
    """

    # Add column. imputed=True ^= entirue row was imputed
    df["imputed"] = False

    df = df.set_index('timestamp').resample('s').asfreq().reset_index()

    # Set boolean columns to boolean type (Reason: fillna gives wornang otherwise)
    df['imputed'] = df['imputed'].astype(bool)


    # Apply imputation method 1: Impute with fixed values
    dict_column_and_fixed_imputed_value = {
        "imputed": True,
        "power": 0,
        "speed": 0,
        "cadence": 0,
        "fractional_cadence": 0,
        "step_length": 0,
        "activity_type": "pause",
        #"steps_per_min": 0,
        #"power100": 0,
    }
    # Assert that the this columns exist in df
    assert set(dict_column_and_fixed_imputed_value.keys()).issubset(set(df.columns)), f"The columns {str(set(dict_column_and_fixed_imputed_value.keys())- set(df.columns))} to be imputed with fixed values are not present in the dataframe. Existing columns are {list(df.columns)}."
    # Impute
    df = df.fillna(dict_column_and_fixed_imputed_value)

    # Apply imputation method 2: linear interpolation
    columns_to_impute_linearly = [
        "cum_distance", "hr", "elevation", "cum_power", "position_lat", "position_long"
        ]
    # Assert that the this columns exist in df
    assert set(columns_to_impute_linearly).issubset(set(df.columns)), f"The columns {set(columns_to_impute_linearly)- set(df.columns)} do not exist in the dataframe. Existing columns are {list(df.columns)}."
    # Impute linearly
    for column in columns_to_impute_linearly:
        df[column] = df[column].interpolate(method='linear')

    # Apply imputation method 3: backward fill
    columns_to_impute_via_backwards_fill = [
        "cum_power",
        ]
    # Assert that the this columns exist in df
    assert set(columns_to_impute_via_backwards_fill).issubset(set(df.columns)), f"The columns {set(columns_to_impute_via_backwards_fill)- set(df.columns)} do not exist in the dataframe. Existing columns are {list(df.columns)}."
    # Impute linearly
    for column in columns_to_impute_via_backwards_fill:
        df[column] = df[column].bfill()

    if False:
        # Special case imputation: column "cum_power" – first row - impute NaN wit 0
        if pd.isna(df.loc[df.index[0], 'cum_power']):
            df.loc[df.index[0], 'cum_power'] = 0
    return df

def _assert_df_is_mostly_imputed(df):
    
    # Assert that most columns are fully imputed
    columns_not_requiring_imputation = {"vertical_oscillation", "stance_time", "vertical_ratio", 'position_lat', 'position_long'}
    columns_requiring_imputation = list(set(df.columns) - columns_not_requiring_imputation)
    ser_nans_per_column = df[columns_requiring_imputation].isna().sum()
    ser_nans_per_column_positive = ser_nans_per_column[ser_nans_per_column > 0]
    assert len(ser_nans_per_column_positive) == 0, f"Imputation failed. The following columns still contain NaNs: {ser_nans_per_column_positive}"



def _load_raw_df_from_fit(path_fit_file: str, frame_name: str = 'record', lat_long_update: bool = True, debug: bool = False) -> pd.DataFrame:
    """
    Decodes a .FIT file and returns the data as a pandas DataFrame.

    Parameters:
    - path_fit_file (str): The path to the .FIT file.
    - frame_name (str): The name of the frame to extract data from. Defaults to 'record'. E.g. device_info, gps_metadata, record, file_id, event, unknown_288,
    - lat_long_update (bool): Whether to update the latitude and longitude columns to standard degrees. Defaults to True.
    - debug (bool): Whether to print the list of frames in the file for debugging purposes. Defaults to False.

    Returns:
    - df (pd.DataFrame): The decoded data as a pandas DataFrame.
    """
    # Initialize some useful variables for the loops
    check_list = good_list = []
    list_check = {}
    df = pd.DataFrame()

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

        df = pd.DataFrame(list_rows)
    
        # Update the Long/Lat columns to standard degrees
        if lat_long_update:
            for column in ['position_lat', 'position_long']:
                df[column] = df[column].apply(lambda x: x / ((2**32)/360))
        
        # If you want to check to see which frames are in the file, print the list_check variable
        if debug:
            print(list_check)

    return df

@st.cache_data
def load_fit_file(fit_file_path: str) -> pd.DataFrame:
    """
    Loads fit fit as DataFrame

    Args:
        fit_file_path (str): path to the fit file

    Returns:
        pd.DataFrame: df of activity. 1 row per second
    """
    df = _load_raw_df_from_fit(fit_file_path)
    df = _rename_columns_and_convert_units(df)
    df = _impute_df_from_fit(df)
    df = _add_columns(df)
    df = _drop_and_reorder_columns(df)
    _assert_df_is_mostly_imputed(df)
    
    return df

if False:
    activity_id = 14057922527
    project_path = r'D:\OneDrive\7Temporary\Coding\2024_02_20_Garmin'

    path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{activity_id}_ACTIVITY.fit')
    df = load_fit_file(path_fit_file)