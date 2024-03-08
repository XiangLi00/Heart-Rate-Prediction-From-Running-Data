from datetime import datetime, timedelta
import importlib
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import scipy

def move_columns_to_end(df, columns):
    for col in columns:
        if col in df.columns: 
            df[col] = df.pop(col)
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