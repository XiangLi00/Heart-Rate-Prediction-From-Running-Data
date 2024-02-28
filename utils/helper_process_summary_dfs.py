from datetime import datetime
import os
import sqlite3
import streamlit as st

import pandas as pd

def process_df_weeks_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Process a DataFrame after loading it from a database.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """

    set_cols_to_be_dropped = {'calories_goal', 'calories_consumed_avg', 'spo2_avg', 'spo2_min', 'hydration_goal', 'hydration_avg', 'hydration_intake', 'weight_avg', 'weight_min', 'weight_max'}
    df = df_raw.drop(columns=set_cols_to_be_dropped)

    return df