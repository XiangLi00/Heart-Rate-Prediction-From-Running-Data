import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly
import streamlit as st
from streamlit_js_eval import streamlit_js_eval



def _is_datetime_utc_column(df, col):
    return pd.api.types.is_datetime64tz_dtype(df[col])

def get_screen_height_and_width():
    screen_height = streamlit_js_eval(js_expressions='screen.height', key='get_screen_height_javascript')
    screen_width = streamlit_js_eval(js_expressions='screen.width', key='get_screen_width_javascript')
    return screen_height, screen_width

def update_screen_height_of_fig_v2(fig: plotly.graph_objs.Figure, height_factor = 0.9, debug=False) -> plotly.graph_objs.Figure:
    screen_height, screen_width = get_screen_height_and_width()
    if debug:
        st.write(f"screen_height: {screen_height}, screen_width: {screen_width}, type_screen_height: {type(screen_height)}")

    if screen_height is not None:
        try:
            fig.update_layout(height=screen_height*height_factor)
        except TypeError:
            st.write(f"TypeError in helper_streamlit.update_screen_height_of_fig_v2(). Could not update screen height. It still has the default value screeen_height = {screen_height}.")
    return fig



def add_df_activities_filtering_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Designed for df_activities since it specifies some default filters (="running general")


    Source: https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if _is_datetime_utc_column(df, col):
        # if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                pass
                # print(f"Parsing error in column {col}")
                # print(f'Error in column {col}: {e} ')

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        # Select which columns to filter on (with certain default columns)
        if {"sport", "sub_sport", "distance"}.issubset(df.columns):
            default_multiselected_filtering_columns = ["sport", "sub_sport", "distance"]
        else:
            default_multiselected_filtering_columns = None
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, default=default_multiselected_filtering_columns)

        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < n unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 20:

                # Define default values
                if column == "sport":
                    default_user_cat_input = ["running"]
                elif column == "sub_sport":
                    default_user_cat_input = ["generic"]
                else:
                    default_user_cat_input = list(df[column].unique())

                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=default_user_cat_input,
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100

                if column == "distance":
                    default_min_max = (5.0, _max)
                else:
                    default_min_max = (_min, _max)

                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    default_min_max,
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def add_df_filtering_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a generic UI on top of a dataframe to let viewers filter columns
    Source: https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, default=None)

        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < n unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 20:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df