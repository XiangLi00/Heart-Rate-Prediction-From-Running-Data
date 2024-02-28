from datetime import datetime
import os
import sqlite3
import streamlit as st

import pandas as pd

import helper_process_summary_dfs

def print_column_info_of_all_tables(
    db_name: str,
    print_columns: bool = True,
    specific_table: None | str = None,
    root_path_db: str = r"C:\Users\Xiang\HealthData\DBs"
):
    """
    Prints all table names (and column names+types) of a given database.

    Args:
        db_name (str): The name of the database.
        print_columns (bool, optional): Whether to print the column names and data types for each table. Defaults to True.
        root_path_db (str, optional): The root path of the database. Default
    """
    print(f"------------------- Database {db_name}:")

    path_db = os.path.join(root_path_db, db_name)

    with sqlite3.connect(path_db) as con:
        cursor = con.cursor()
        # Get a list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Display information about each table
        for table in tables:
            table_name = table[0]
            
            # If a specific table is specified, only print the columns of that table   
            if specific_table and specific_table != table_name:
                continue

            print(f"\nTable: {table_name}")

            cursor.execute(f"SELECT * FROM {table_name};")  
            rows = cursor.fetchall()
            if len(rows) > 0:
                print(f"  Shape: {len(rows)} rows, {len(rows[0])} columns")

            if print_columns:
                # Get the column names and data types for each table
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()  # List with elements like (0, 'first_day', 'DATE', 1, None, 1)

                print("Columns:")
                # Create a dictionary with column names as keys and data types as values
                dict_this_table = {column[1]: column[2] for column in columns}  
                print(dict_this_table)

    print('--------------------')


def get_column_info_of_specific_table(
    db_name: str,
    table_name: str,
    root_path_db: str = r"C:\Users\Xiang\HealthData\DBs"
):
    """
    Returns a dictionary with column names as keys and data types as values for a specific table in a database.

    Args:
        db_name (str): The name of the database.
        table_name (str): The name of the table.
        root_path_db (str, optional): The root path of the database. Defaults to 

    Returns:
        dict: A dictionary with column names as keys and data types as values.
    """
    path_db = os.path.join(root_path_db, db_name)
    # print("Loading data from: ", root_path_db)
    try:
        with sqlite3.connect(path_db) as con:
            cursor = con.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()  # List with elements like (0, 'first_day', 'DATE', 1, None, 1)

            # Create a dictionary with column names as keys and data types as values
            dict_column_name_and_type = {column[1]: column[2] for column in columns}  
    except sqlite3.OperationalError as e:
        error_message = f"Error in get_column_info_of_specific_table: {e}. root_path_db: {root_path_db}, db_name: {db_name}, table_name: {table_name}"
        raise sqlite3.OperationalError(error_message) from e
        #raise sqlite3.OperationalError(f"root_path_db: {root_path_db} db_name: {db_name}, table_name: {table_name}")
        

    return dict_column_name_and_type

# get_column_info_of_specific_table('summary.db', 'weeks_summary')
# print_column_info_of_all_tables('garmin_monitoring.db', print_columns=True)


def remove_empty_nan_or_zero_rows(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Remove rows where all values are nan, zero or empty.

        Currently only applieds to summary.db/weeks_summary

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    if 'first_day' in df.columns: 
        df_indexed = df.set_index('first_day')

        # make df boolean. True if value is nan, zero or empty
        df_indexed_bool_is_nan_zero_or_empty = df_indexed.map(lambda x: pd.isna(x) or not bool(x))

        # remove rows where all values are nan, zero or empty
        df_indexed_empty_rows_removed = df_indexed[~df_indexed_bool_is_nan_zero_or_empty.all(axis=1)]

        df_shape_before = df.shape

        # reset index
        df = df_indexed_empty_rows_removed.reset_index()
        if verbose:
            if df.shape != df_shape_before:
                print(f"Removed {df_shape_before[0] - df.shape[0]} empty/nan/zero rows. Shape before: {df_shape_before}, shape after: {df.shape}. Printed in 'helper_load_df.py/load_df()/remove_empty_nan_or_zero_rows()'")

    return df

def process_df(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Process a DataFrame after loading it from a database.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        table_name (str): The name of the table.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """

    if table_name == 'weeks_summary':
        df = helper_process_summary_dfs.process_df_weeks_summary(df)

    return df

dict_table_name_to_db_name = {
    'activities': 'garmin_activities.db',
    'activity_laps': 'garmin_activities.db',
    'activity_records': 'garmin_activities.db',
    'steps_activities': 'garmin_activities.db',
    'days_summary': 'summary.db',
    'years_summary': 'summary.db',
    'weeks_summary': 'summary.db',
    'months_summary': 'summary.db',
    'intensity_hr': 'garmin_summary.db',
    'monitoring': 'garmin_monitoring.db',
    'monitoring_climb': 'garmin_monitoring.db',
    'monitoring_hr': 'garmin_monitoring.db',
    'monitoring_rr': 'garmin_monitoring.db',
    'monitoring_intensity': 'garmin_monitoring.db',
    'monitoring_climb': 'garmin_monitoring.db',
    'files': 'garmin.db',
    'resting_hr': 'garmin.db',
    'sleep': 'garmin.db',
    'sleep_events': 'garmin.db',
    'stress': 'garmin.db',
    'weight': 'garmin.db',
}


@st.cache_data
def load_df_v2(
        table_name: str,
        root_path_db: str = r"C:\Users\Xiang\HealthData\DBs",
        sql_selected_columns: str = "*",
        sql_condition: str = "",
) -> pd.DataFrame:
    """
    Load a DataFrame from a SQLite database table.

    Parse column data types:
    - DATETIME: Parse as datetime64
    - DATE: Parse as datetime64
    - TIME: Parse as timedelta64
    - INTEGER: Parse as float

    Parameters:
        table_name (str): The name of the table to load.
        root_path_db (str, optional): The root path of the database. 

    Returns:
        pd.DataFrame: The loaded DataFrame.

    """
    db_name = dict_table_name_to_db_name[table_name]

    # Get the column names and data types for the specified table
    dict_table_column_info = get_column_info_of_specific_table(db_name=db_name, table_name=table_name, root_path_db=root_path_db)

    # pinf('dict_table_column_info')
    datetime_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'DATETIME']
    date_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'DATE']
    time_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'TIME']
    integer_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'INTEGER']

    path_db = os.path.join(root_path_db, db_name)
    with sqlite3.connect(path_db) as con:

        # Read the SQL query with the updated variables

        # If we have no condition, then it is empty, else need to add WHERE
        sql_where_statement = f"WHERE {sql_condition}" if sql_condition else ""

        df = pd.read_sql(
            f"SELECT {sql_selected_columns} FROM {table_name} {sql_where_statement}",  # WHERE first_day > '2023-11-25'",
            con,
            parse_dates=datetime_column_names + date_column_names  # Parse datetime and date as datetime64
        )
        # df.name = table_name

    # Convert time columns to timedelta
    df[time_column_names] = df[time_column_names].map(pd.to_timedelta)
    # Convert integer columns to float
    df[integer_column_names] = df[integer_column_names].astype(float)

    df = remove_empty_nan_or_zero_rows(df, verbose=False)
    df = process_df(df=df, table_name=table_name)

    print(f"Loaded {table_name} from {db_name}. Shape: {df.shape}. ")

    return df

@st.cache_data
def load_df(
        db_name: str,
        table_name: str,
        root_path_db: str = r"C:\Users\Xiang\HealthData\DBs",
        sql_selected_columns: str = "*",
        sql_condition: str = "",
) -> pd.DataFrame:
    """
    Load a DataFrame from a SQLite database table.

    Parse column data types:
    - DATETIME: Parse as datetime64
    - DATE: Parse as datetime64
    - TIME: Parse as timedelta64
    - INTEGER: Parse as float

    Parameters:
        db_name (str): The name of the SQLite database.
        table_name (str): The name of the table to load.
        root_path_db (str, optional): The root path of the database. 

    Returns:
        pd.DataFrame: The loaded DataFrame.

    """

    # Get the column names and data types for the specified table
    dict_table_column_info = get_column_info_of_specific_table(db_name=db_name, table_name=table_name, root_path_db=root_path_db)

    # pinf('dict_table_column_info')
    datetime_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'DATETIME']
    date_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'DATE']
    time_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'TIME']
    integer_column_names = [column_name for (column_name, data_type) in dict_table_column_info.items() if data_type == 'INTEGER']

    path_db = os.path.join(root_path_db, db_name)
    with sqlite3.connect(path_db) as con:

        # Read the SQL query with the updated variables

        # If we have no condition, then it is empty, else need to add WHERE
        sql_where_statement = f"WHERE {sql_condition}" if sql_condition else ""

        df = pd.read_sql(
            f"SELECT {sql_selected_columns} FROM {table_name} {sql_where_statement}",  # WHERE first_day > '2023-11-25'",
            con,
            parse_dates=datetime_column_names + date_column_names  # Parse datetime and date as datetime64
        )
        # df.name = table_name

    # Convert time columns to timedelta
    df[time_column_names] = df[time_column_names].map(pd.to_timedelta)
    # Convert integer columns to float
    df[integer_column_names] = df[integer_column_names].astype(float)

    df = remove_empty_nan_or_zero_rows(df, verbose=False)
    df = process_df(df=df, table_name=table_name)

    print(f"Loaded {table_name} from {db_name}. Shape: {df.shape}. ")

    return df


def generate_report(df: pd.DataFrame):
    """Generates a profiling report for a DataFrame and saves it as an HTML file.

    Args:
        df (pd.DataFrame): The DataFrame to generate a report for.
    """
    print(f'Generating profiling report for {df.name}...')

    # Retrieve current datetime, e.g., \"2024_02_15__11_27_38\
    current_datetime_as_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")   # 
    path_df_profiling_report = os.path.join("temp", f"{current_datetime_as_string}_profiling_report_{df.name}.html")

    ProfileReport(df, title=f"Profiling Report {df.name}").to_file(path_df_profiling_report)

    print(f"Generated profiling report for {df.name}. Saved as {path_df_profiling_report}")


