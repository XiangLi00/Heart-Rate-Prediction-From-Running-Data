from datetime import datetime, timedelta
import importlib
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Tuple, Union


import numpy as np
import pandas as pd
import scipy

def move_columns_to_end(df, columns):
    for col in columns:
        if col in df.columns: 
            df[col] = df.pop(col)
    return df

def move_column_left_of_another_column(df, col_to_move, ref_col):
    """
    This function moves a column in a DataFrame just before another column.

    Parameters:
    df (pd.DataFrame): The DataFrame in which to move the columns.
    col_to_move (str): The name of the column to move.
    ref_col (str): The name of the column before which col_to_move should be placed.

    Returns:
    pd.DataFrame: The DataFrame with the columns reordered.
    """
    # Ensure the columns exist in the DataFrame
    if col_to_move not in df.columns or ref_col not in df.columns:
        raise ValueError(f"Columns {col_to_move} or {ref_col} not in DataFrame")

    # Pop the column to move and store it in a variable
    col_data = df.pop(col_to_move)

    # Insert the column at the correct location
    df.insert(df.columns.get_loc(ref_col), col_to_move, col_data)

    return df

def reorder_columns(df: pd.DataFrame, columns_desired_order: list[str]) -> pd.DataFrame:
    """
    Reorders the columns of a DataFrame according to a desired order.

    Example: df = reorder_columns(df, columns_desired_order)


    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_desired_order (list[str]): The desired order of the columns. Columns not present in this list will be appended at the end of the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the columns reordered according to the desired order.
    """
    # contains exactly the same columns as df_resampled, but in the desired order
    cols_new_order = [col for col in columns_desired_order if col in df.columns] + [col for col in df.columns if col not in columns_desired_order]
    df = df[cols_new_order]
    return df

def _adjust_column_width_for_single_excel_sheet_v2(worksheet, max_width):
    for col in worksheet.columns:
        
        # Retrieve existing max cell width in the column
        max_cell_width_in_col = max([len(str(cell.value)) for cell in col if hasattr(cell.value, '__str__')])
        # Adjust the width if necessary
        adjusted_width = min(max_cell_width_in_col + 2, max_width)  # +2 for a little extra margin

        column_letter = col[0].column  # Get the column letter

        worksheet.column_dimensions[openpyxl.utils.get_column_letter(column_letter)].width = adjusted_width

def adjust_excel_column_width(excel_filename, max_width=60):
    """
    Adjusts the column widths in an Excel file to a specified maximum width.

    Parameters:
    excel_filename (str): The path of the Excel file to adjust.
    max_width (int): The maximum width in characters (default is 60).

    Returns:
    None
    """
    # Load the workbook
    workbook = openpyxl.load_workbook(excel_filename)

    # Adjust column widths in each sheet
    for sheetname in workbook.sheetnames:
        sheet = workbook[sheetname]
        _adjust_column_width_for_single_excel_sheet_v2(sheet, max_width)

    # Save the workbook
    workbook.save(excel_filename)


def get_previous_row_index(index_value, df: pd.DataFrame):
    """
    Given an index value, returns the previous index value (if existing).

    Used in grade computation

    Args:
        index_value (int): The index value.
        df (pd.DataFrame): The dataframe.

    Returns:
        int: The previous row index.
    """
    current_index_number = df.index.get_loc(index_value)
    if current_index_number == 0:
        previous_index_number_if_existing = 0
    else:
        previous_index_number_if_existing = current_index_number - 1
    return df.index[previous_index_number_if_existing]


def assert_df_is_mostly_imputed(df: pd.DataFrame, columns_not_requiring_imputation: set[str]):
    """
    Asserts that most columns are fully imputed

    Parameters:
    - df (pd.DataFrame): The DataFrame to be checked for imputation.
    - columns_not_requiring_imputation (set[str]): A set of column names that are not required to be fully imputed. It may contain columns that do not exist in df

    Raises:
    - AssertionError: If any column requiring imputation still contains NaN values.
    """
    columns_requiring_imputation = list(set(df.columns) - set(columns_not_requiring_imputation))
    # Count NaNs per column
    ser_nans_per_column = df[columns_requiring_imputation].isna().sum()

    ser_nans_per_column_positive = ser_nans_per_column[ser_nans_per_column > 0]
    assert len(ser_nans_per_column_positive) == 0, f"Imputation failed. The following columns still contain NaNs: {ser_nans_per_column_positive}"


def impute_drop_or_assert_columns(
    df: pd.DataFrame,
    dict_column_imputation_strategy: dict[str, Any],
    impute: bool = True,
    drop: bool = True,
    assert_most_columns_imputed: bool = True
) -> None:
    """
    Perform imputation, dropping, or asserting on columns of a DataFrame based on provided strategies.


    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        dict_column_imputation_strategy (dict[str, Any]): A dictionary specifying the imputation strategy for each column.
            Keys are column names, values are the imputation strategy ("linear", "bfill") or a fixed imputed value or "drop" (for dropping the column).

            If the columns should be imputed, then the key/column_name has to exist.
        impute (bool, optional): Whether to perform imputation. Defaults to False.
        drop (bool, optional): Whether to drop columns. Defaults to False.
        assert_most_columns_imputed (bool, optional): Whether to assert that most columns are imputed. Defaults to False.
    """

    # Split the columns into different imputation strategies
    dict_column_and_fixed_imputed_value = {k: imputed_value for k, imputed_value in dict_column_imputation_strategy.items() if imputed_value not in {"linear", "bfill", "no_imputation", "drop"}}
    columns_to_drop = [k for k, v in dict_column_imputation_strategy.items() if v == "drop" and k in df.columns]
    columns_with_bfill_imputation = {k for k, v in dict_column_imputation_strategy.items() if v == "bfill"}
    columns_with_linear_imputation = {k for k, v in dict_column_imputation_strategy.items() if v == "linear"}
    columns_not_requiring_imputation = {k for k, v in dict_column_imputation_strategy.items() if v == "no_imputation"}



    # Drop columns
    if drop:
        df = df.drop(columns=columns_to_drop)

    # Impute columns
    if impute:
        # Assert that all columns to be imputed exist in the dataframe
        columns_required_to_exist = {k for k, v in dict_column_imputation_strategy.items() if v not in {"no_imputation", "drop"}}  # All thse columns names have to exist
        assert set(columns_required_to_exist).issubset(set(df.columns)), f"The columns {set(columns_required_to_exist)- set(df.columns)} do not exist in the dataframe. Existing columns are {list(df.columns)}."

        # Imput
        df = df.fillna(dict_column_and_fixed_imputed_value)
        for column in columns_with_linear_imputation:
            df[column] = df[column].interpolate(method='linear')
        for column in columns_with_bfill_imputation:
            df[column] = df[column].bfill()

    # Assert that most columns are imputed
    if assert_most_columns_imputed:
        assert_df_is_mostly_imputed(df=df, columns_not_requiring_imputation=columns_not_requiring_imputation)

    return df

def mean_with_zero_imputation(x: np.ndarray, expected_length: int):
    """
    Calculate the mean of an array with zero imputation if the array length is less than the expected length.

    Used for compute rolling average imputed with 0

    Parameters:
    x (np.ndarray): The input array.
    expected_length (int): The expected length of the array.

    Returns:
    float: The mean of the array with zero imputation if necessary.

    Raises:
    ValueError: If the length of the array is greater than the expected length.
    """
    window_size_received = len(x)
    if window_size_received == expected_length:
        return x.mean()
    elif window_size_received < expected_length:
        # implictly impute with 0 to make sure we compute the mean of the full expected window_size
        return window_size_received/expected_length * x.mean()
    else:
        raise ValueError(f'window_size_received = {window_size_received} > expected_length = {expected_length}')