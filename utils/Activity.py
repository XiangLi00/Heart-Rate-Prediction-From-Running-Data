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
import scipy 
import streamlit as st

from utils import helper_compute_features
import helper_pandas

class Activity():

    def __init__(self, id: int, config: Dict[str, Any], project_path: str, add_experimental_columns = True) -> None:
        self.id = id
        self.config = config
        self.df_special_col_names = pd.DataFrame()

        path_fit_file = os.path.join(project_path, 'data', 'FitFiles', 'Activities', f'{id}_ACTIVITY.fit')

        self.df = load_raw_df_from_fit_v2(path_fit_file, frame_name='record', lat_long_update=True)

        # Rename columns 
        self.df = self.df.rename(columns={
            'enhanced_speed': 'speed_garmin', 
            'enhanced_altitude': 'elevation_raw', 
            'heart_rate': 'hr', 
            'accumulated_power': 'cum_power',
            'distance': 'cum_distance_raw'
        })

        # Unit conversion
        self.df['speed_garmin'] = self.df['speed_garmin']*3.6  # m/s to km/h

        self.df = resample_each_second_and_add_imputed_column(self.df)
        
        # For each column (which might have NaNs): define imputation strategy
        # Key: column name. 
        # Value: imputation strategy ("bfill","linear", "no_imputation") or <fixed value>
        self.dict_column_imputation_strategy = {
            "cum_distance_raw": "bfill",
            "hr": "linear",
            "elevation_raw": "linear",
            "cum_power": "linear",            
            "position_lat": ["linear", "drop"][1],
            "position_long": ["linear", "drop"][1],
            "cum_power": "bfill",
            "power": 0,
            "speed_garmin": [0, "drop"][0],
            "cadence": 0,
            "fractional_cadence": 0,
            "step_length": [0, "drop"][1],
            "activity_type": "pause",
            "vertical_oscillation": ["no_imputation", "drop"][1],
            "stance_time": ["no_imputation", "drop"][1],
            "vertical_ratio": ["no_imputation", "drop"][1],
            "stance_time_balance": "drop",
            "stance_time_percent": "drop",
        }

        # Impute, drop, and assert that most columns are imputed
        self.df = helper_pandas.impute_drop_or_assert_columns(
            df=self.df, 
            dict_column_imputation_strategy=self.dict_column_imputation_strategy,
            impute=True,
            drop=True,
            assert_most_columns_imputed=True
        )


        # Add column steps_per_min and drop cadence and fractional_cadence
        self.df['steps_per_min'] = (self.df['cadence'] + self.df['fractional_cadence']) * 2 
        self.df = self.df.drop(columns=['fractional_cadence', 'cadence'])

        # Add column cum_elevation_change (by cleaning and smoothing elevation_change_raw)
        self.df['elevation_change_raw'] = self.df['elevation_raw'].diff().fillna(0)
        # Clean/process elevation_change_raw data: Assume that sudden change (abovf threshold) and change during break(^=imputed row) is noise. In this case set to 0
        self.df["elevation_change_interm"] = (~self.df["imputed"]).astype(int) * self.df["elevation_change_raw"] * \
            (self.df["elevation_change_raw"].abs() <= self.config["df__elevation_change_interm__threshold_for_setting_sudden_change_to_zero"] )  # assume: sudden elevation change ^= noise
        # Applies Gaussian kernel to smoothen it
        self.df["elevation_change"]  = scipy.ndimage.gaussian_filter1d(self.df["elevation_change_interm"], sigma=self.config["df__elevation_change__gaussian_kernel_sigma"])
        self.df["cum_elevation_change"] = self.df["elevation_change"].cumsum()
        # Drop raw columns unless specified otherwise
        if (not "df__keep_raw_elevation_distance_columns_for_debugging" in config.keys()) or self.config["df__keep_raw_elevation_distance_columns_for_debugging"] == False:
            self.df = self.df.drop(columns=['elevation_change_raw', 'elevation_change_interm', 'elevation_change'])

        # Add column speed and cum_distance (by smoothing distance_raw)
        self.df['distance_raw'] = self.df['cum_distance_raw'].diff().fillna(0)
        self.df["distance"] = scipy.ndimage.gaussian_filter1d(self.df["distance_raw"], sigma=self.config["df__distance__gaussian_kernel_sigma"]) # Remark: Get better values by smoothing on non-cumulative values
        self.df["speed"] = self.df["distance"] * 3.6  # km/h since distance is given in m (for the past 1 second)
        self.df["cum_distance"] = self.df["distance"].cumsum()
        # Drop raw columns unless specified otherwise
        if (not "df__keep_raw_elevation_distance_columns_for_debugging" in config.keys()) or self.config["df__keep_raw_elevation_distance_columns_for_debugging"] == False:
            self.df = self.df.drop(columns=['distance_raw', 'distance'])


        # Adds column grade
        # Formula: As Sekantensteigung between now and >= e.g., 5 meters ago
        # Param (df__grade__delta_distance_for_grade_computation): If it is 100m, then grade information is "lagging behind" about 100m/2. Around 5 is good. Need it to be >0.5 because otherwise small delta distance can create a lot of noisy/huge grades.
        self.df, grade_col_name = add_grade_column(
            df=self.df,
            delta_distance_for_grade_computation=self.config["df__grade__delta_distance_for_grade_computation"],
            custom_grade_col_name="grade",
            col_name_cum_distance="cum_distance",
            col_name_elevation="cum_elevation_change",
            add_debug_columns=False
        )

        # Print a warning in case absolute grade is unrealistically high
        threshold_abs_grade = 50
        high_grade_rows = self.df[abs(self.df["grade"]) > threshold_abs_grade][["timestamp", "grade"]]
        if not high_grade_rows.empty:
            print(f"Warning: The following rows have an absolute grade greater than {threshold_abs_grade}")
            print(high_grade_rows)
        spline_object = helper_compute_features.get_spline_for_gap_computation()
        self.df["gaspeed"] = self.df["speed"] * spline_object(self.df["grade"])
 

        if add_experimental_columns:
            self.df["gaspeed4"] = np.maximum(4, self.df["gaspeed"]) 
            # Add exponential weighted moving average columns
            variable_to_smoothen_exponentially = ["gaspeed4"]
            list_ew_spans_in_s = [30, 100]
            for variable in variable_to_smoothen_exponentially:
                for ew_span in list_ew_spans_in_s:
                    self.df[f"{variable}_ew_{ew_span}s"] = self.df[variable].ewm(span=ew_span).mean()

        # Assert that most columns are imputed
        self.df = helper_pandas.impute_drop_or_assert_columns(
            df=self.df, 
            dict_column_imputation_strategy=self.dict_column_imputation_strategy,
            impute=False,
            drop=False,
            assert_most_columns_imputed=True
        )

    def add_rolling_avg_columns(
        self,
        base_vars: Union[List[str], None] = None,
        window_lengths: Union[List[int], None] = None
    ) -> None:
        """
        Adds rolling average columns to the dataframe based on the specified configuration.

        Parameters:
        - rolling_avg_base_vars: Union[List[str], None]. Specify for which variables to compute rolling average. 
        - rolling_avg_window_lengths: Union[List[int], None]. Specify for which previous seconds to compute rolling average.
        E.g. [10, 30] means: first use seconds 0-9 earlier, then use seconds 10-39 earlier.

        The rolling average columns are added to the self.df dataframe, and the column name information is stored in self.df_special_col_names.
        """

        # Set default values from self.config if parameters are not provided
        try:
            window_lengths = window_lengths or self.config["df__rolling_avg_window_lengths"]
            if not base_vars:
                base_vars = self.config["df__rolling_avg_base_vars"]  # Equivalent syntax
            
        except KeyError as e:
            raise ValueError("Both rolling_avg_base_vars and rolling_avg_window_lengths must be provided either as arguments or in self.config.") from e

        # Prepare to add column name information to self.df_special_col_names
        if not hasattr(self, "df_special_col_names"):
            self.df_special_col_names = pd.DataFrame()
        list_for_creating_df_special_col_names = []

        # Decode configuration. Retrieve for which past seconds to compute rolling average
        rolling_avg_first_previous_times = np.cumsum(window_lengths).tolist()  # e.g. [10, 40], when df__rolling_avg_base_vars=[10, 30]
        rolling_avg_last_previous_times = [0] + rolling_avg_first_previous_times[:-1]  # e.g. [0, 10]

        # Add columns such as "gaspeed_avg_last_29-10s"
        # Do this for all base_vars and window_lengths
        for base_var in base_vars:
            for i in range(len(window_lengths)):

                first_previous_time = rolling_avg_first_previous_times[i]-1  # i.e. how many seconds ago to start the rolling average
                last_previous_time = rolling_avg_last_previous_times[i]
                window_size = first_previous_time+1 - last_previous_time

                # print(f'Compute rolling averge of the rows ({first_previous_time}, {last_previous_time}] above current row')

                col_name = f"gaspeed_avg_last_{first_previous_time}-{last_previous_time}s"

                if first_previous_time+1 > self.df.shape[0]:
                    self.df[col_name] = 0.0
                    # print(f'The dataframe only has length {len(df2)}. Therefore, the entire column is set to 0.')
                    continue

                # Compute rolling average
                self.df[col_name] = self.df["gaspeed"].shift(last_previous_time).fillna(0).rolling(window=window_size, min_periods=1).apply(helper_pandas.mean_with_zero_imputation, raw=True, kwargs={"expected_length": window_size})

                list_for_creating_df_special_col_names.append({
                    "base_var": base_var,
                    "method": "shifted_rolling_avg",
                    "first_previous_time": first_previous_time,
                    "last_previous_time": last_previous_time,
                    "col_name": col_name
                })
        
        # add column name information
        self.df_special_col_names = pd.concat([self.df_special_col_names, pd.DataFrame(list_for_creating_df_special_col_names)], axis=0, ignore_index=True)


def resample_each_second_and_add_imputed_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples the dataframe to have a row for each second, filling in missing rows with NaNs.
    """

    # Add column. imputed=True ^= entire row was imputed
    df["imputed"] = False

    df = df.set_index('timestamp').resample('s').asfreq().reset_index()

    # Set boolean columns to boolean type (Reason: fillna gives warning otherwise)
    df['imputed'] = df['imputed'].astype(bool)

    return df

def load_raw_df_from_fit_v2(path_fit_file: str, frame_name: str = 'record', lat_long_update: bool = True) -> pd.DataFrame:
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

    # list of all rows. Will be merged to a dataframe at the end
    list_row_dicts = []

    # Open the file with fitdecode
    with fitdecode.FitReader(path_fit_file) as file:    
        for frame in file:
            if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == frame_name:
                
                field_names = [field.name for field in frame.fields if not field.name.startswith('unknown')]

                row_dict = {field_name: frame.get_value(field_name) for field_name in field_names}
                list_row_dicts.append(row_dict)

    df = pd.DataFrame(list_row_dicts)
    
    # Update the Long/Lat columns to standard degrees
    if lat_long_update:
        for column in ['position_lat', 'position_long']:
            df[column] = df[column].apply(lambda x: x / ((2**32)/360))
    
    return df


def add_grade_column__get_first_last_row_indices_for_grade_calc(
    current_row_index: int,
    current_cum_distance: float,
    df: pd.DataFrame,
    delta_distance_for_grade_computation: float = 5,
    col_name_cum_distance: str = "cum_distance"
) -> Tuple[Any, Any]:
    """
    Calculates the first and last row indices for grade computation. ("Sekantensteigung")

    Suppose delta_distance_for_grade_computation = 10 [m]
    Idea: Compute grade based on now and >= 10 meters ago (i.e. the first entry >= 10 meters ago). 
    Edge cases:
    - If we just started, then we use meters [0, 10]
    - If we are at the end of the activity, then we use meters [last cum_distance - 10, last cum_distance]
    
    Args:
        current_row_index (int): The index of the current row.
        current_cum_distance (float): The cumulative distance at the current row.
        df (pd.DataFrame): The DataFrame containing the data.
        delta_distance_for_grade_computation (float, optional): The delta distance used to determine the range of rows for grade calculation. Defaults to 5 [m].
        col_name_cum_distance (str, optional): The name of the column containing the cumulative distance. Defaults to "cum_distance".

    Returns:
        Tuple[Any, Any]: A tuple containing the indices of the first and last rows to use for grade calculation.
    """
    
    ## Compute first_row_index ^= usually the row >= 10 meters ago
    first_row_index = helper_pandas.get_previous_row_index(
        df[col_name_cum_distance].searchsorted(
            current_cum_distance-delta_distance_for_grade_computation
        ),
        df=df
    )

    ## Computes the last_row_index ^= usually the current row
    first_cum_distance = df.loc[df.index[0], col_name_cum_distance]
    last_cum_distance = df.loc[df.index[-1], col_name_cum_distance]
    # Case: we are at the last 10 meters of the activity. Then we want to use meters [last cum_distance - 10, last cum_distance]
    if current_cum_distance > last_cum_distance - delta_distance_for_grade_computation:
        last_row_index = df.index[-1] # ^= last row of activity
    # Case: We are at the first 10 meters of the activity. Then we want to use meters [0, 10]
    elif current_cum_distance <= first_cum_distance + delta_distance_for_grade_computation:
        last_row_index = df[col_name_cum_distance].searchsorted(
                current_cum_distance+delta_distance_for_grade_computation
            )
    # Normal case
    else:
        last_row_index = current_row_index
    return (first_row_index, last_row_index)


def add_grade_column(
    df: pd.DataFrame,
    delta_distance_for_grade_computation: float = 5,
    custom_grade_col_name: str = None,
    col_name_cum_distance: str = "cum_distance",
    col_name_elevation: str = "elevation",
    add_debug_columns: bool = False
) -> pd.DataFrame:
    """
    Adds a column to the dataframe containing the grade (of the last n meters).

    Details:
    - Ignores elevation changes during breaks (i.e. when imputed == True).

    Flaws:
    - The start of an uphill interval still uses the (often negative) grade of the last interval. So it reacts a bit slowly.

    Call functions via: 
    - df, grade_col_name = _add_grade_column(df, delta_distance_for_grade_computation=10, add_debug_columns=False).

rgs:
        df (pd.DataFrame): The dataframe.
        delta_distance_for_grade_computation (float, optional): The minimal delta distance [m] used for grade computation. Default is 10. Motivation: Make sure that the grade is not too noisy for short distances.
        custom_grade_col_name (str, optional): The name of the column to store the computed grades. If None, a default name "grade_last_10m" will be used. 
        col_name_cum_distance (str, optional): The name of the column containing cumulative distance values. Default is "cum_distance".
        col_name_elevation (str, optional): The name of the column containing elevation values. Default is "elevation".
        add_debug_columns (bool, optional): If True, additional intermediate columns used for grade computation are added to the dataframe. Default is False.
    Returns:
        pd.DataFrame: The dataframe with the added grade_last_10m column.
    """

    assert df[col_name_cum_distance].is_monotonic_increasing, "df.cum_distance (of fit file) is not monotonically increasing"
    assert delta_distance_for_grade_computation > 0, f"delta_distance_for_grade_computation {delta_distance_for_grade_computation} has to be positive"

    # Add columns containing information about which 2 rows to use for grade computation
    # Why need lambda function: Because some parameters should not be vectorized
    df["first_row_index_for_grade_calc"], df["last_row_index_for_grade_calc"] = \
        np.vectorize(
            lambda current_row_index, current_cum_distance: 
                add_grade_column__get_first_last_row_indices_for_grade_calc(
                    current_row_index=current_row_index,
                    current_cum_distance=current_cum_distance,
                    delta_distance_for_grade_computation=delta_distance_for_grade_computation,
                    df=df,
                    col_name_cum_distance=col_name_cum_distance,
                )
        )(
            current_row_index=df.index,
            current_cum_distance=df[col_name_cum_distance]        
        )

    # Retrieve the elevation and cum_distance for the 2 given rows/times
    # Remark: need to convert to np.array or list in order to avoid the new irrelevant index causing issues
    df["first_elevation_for_grade_calc"] = np.array(df[col_name_elevation].loc[df["first_row_index_for_grade_calc"]])
    df["first_cum_distance_for_grade_calc"] = np.array(df[col_name_cum_distance].loc[df["first_row_index_for_grade_calc"]])
    df["last_elevation_for_grade_calc"] = np.array(df[col_name_elevation].loc[df["last_row_index_for_grade_calc"]])
    df["last_cum_distance_for_grade_calc"] = np.array(df[col_name_cum_distance].loc[df["last_row_index_for_grade_calc"]])
    df["cum_distance_difference_for_grade_calc"] = df["last_cum_distance_for_grade_calc"] - df["first_cum_distance_for_grade_calc"]

    # Assert that Sekantensteigung is taken for distances >= delta_distance_for_grade_computation [m] (as intended)
    df_cum_distance_difference_too_low = df.query("cum_distance_difference_for_grade_calc < @delta_distance_for_grade_computation")
    if len(df_cum_distance_difference_too_low) > 0:
        debug_columns_of_interest = ["timestamp", col_name_cum_distance,"cum_distance_difference_for_grade_calc", "first_cum_distance_for_grade_calc", "last_cum_distance_for_grade_calc",  "first_row_index_for_grade_calc", "last_row_index_for_grade_calc"]
        raise AssertionError(f"cum_distance_difference_for_grade_calc is too low (i.e. < {delta_distance_for_grade_computation} [m]) for the following rows: {df_cum_distance_difference_too_low.index} {df_cum_distance_difference_too_low[debug_columns_of_interest]}")

    # Define column name
    if custom_grade_col_name is None:
        grade_col_name = f'grade_last_{int(round(delta_distance_for_grade_computation))}m'
    else:
        grade_col_name = custom_grade_col_name

    # Grade in %
    df[grade_col_name] = (df["last_elevation_for_grade_calc"] - df["first_elevation_for_grade_calc"]) / df["cum_distance_difference_for_grade_calc"]  * 100

    if not add_debug_columns:
        df = df.drop(columns = ['first_row_index_for_grade_calc', 'last_row_index_for_grade_calc', 'first_elevation_for_grade_calc', 'first_cum_distance_for_grade_calc', 'last_elevation_for_grade_calc', 'last_cum_distance_for_grade_calc', 'cum_distance_difference_for_grade_calc'])

    return df, grade_col_name

# –––––––– Stop adding grade column ––––––––––
