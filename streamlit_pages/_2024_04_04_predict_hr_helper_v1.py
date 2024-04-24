from datetime import datetime, timedelta
import importlib
import math
import os
import sqlite3
import sys
import time

import bokeh
import lightgbm as lgb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import sklearn
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from streamlit_plotly_events import plotly_events


from utils import helper_load_fit_file_v1, helper_load_specific_df, helper_pandas, helper_streamlit, helper_sklearn
from utils.Activity import Activity

project_path = os.getcwd()


@st.cache_data 
def load_df_all_activities_labeled_and_processed():
    # 1 row per second, for all labeled activity ids.
    # Useful feature columns (in particular power and gaspeed averages) were already added 
    directory_activities_data = os.path.join(project_path, "data", "Processed", "2024_03_16__only_above_5km__75_activities")
    df_all_activities_labeled_and_processed = pd.read_feather(os.path.join(directory_activities_data, "df_all_activities_labeled_and_processed__2024_03_16__only_above_5km__75_activities.feather"))
    return df_all_activities_labeled_and_processed



    # Read data
    df = _2024_04_04_predict_hr_helper_v1.load_df_all_activities_labeled_and_processed()

def train_model_and_predict_hr_v2():
    # Splits the activities into training and validation activities
    # Trains a regression model 
    # Predicts the heart rate for the training and validation activities
    # Creates df_train and df_val that include the hr prediction columns and all rows (even those for which the hr measurement was apriori labeled as unreliable)

    # Extract data/information from df
    #df_only_correct_hr_rows = st.session_state.df.query("hr_acc_label == 'perfect'")
    

    # Get X_train, X_val, y_train, y_val, df_train, df_val from df_all_activities
    
    #feature_columns = ['cum_power', 'power100_ew_100s', 'power100_ew_30s', 'gaspeed4_ew_100s', 'gaspeed4_ew_30s']
    feature_columns = ["cum_power"] + [col for col in st.session_state.df.columns if "_avg_last_" in col] 
    target_column = "hr"

    # Divide into training and validation activities
    train_activities_ratio = 0.7
    seed = None
    list_all_activity_ids = st.session_state.df['activity_id'].unique().tolist()
    st.session_state.list_train_activity_ids, st.session_state.list_val_activity_ids = sklearn.model_selection.train_test_split(list_all_activity_ids, train_size=train_activities_ratio, random_state=seed)

    X_train = st.session_state.df.query("hr_acc_label == 'perfect' and activity_id in @st.session_state.list_train_activity_ids")\
        [['activity_id', 'timestamp']+feature_columns].set_index(['activity_id', 'timestamp'])
    y_train = st.session_state.df.query("hr_acc_label == 'perfect' and activity_id in @st.session_state.list_train_activity_ids")\
        .set_index(['activity_id', 'timestamp'])[target_column]
    X = st.session_state.df[['activity_id', 'timestamp']+feature_columns].set_index(['activity_id', 'timestamp'])


    # Define model
    model = sklearn.pipeline.Pipeline([
        #('scaler', sklearn.preprocessing.StandardScaler()),
        ("model", sklearn.linear_model.Ridge(alpha=0.1))
    ])
    model = sklearn.pipeline.Pipeline([
        #('scaler', sklearn.preprocessing.StandardScaler()),
        ("model", lgb.LGBMRegressor(objective = "mse", verbose=-1, n_jobs=4,
            num_leaves=10,
            min_data_in_leaf=10,
            learning_rate=0.1,
            max_depth=10,
            n_estimators=50,
            lambda_l2=10,
        ))
    ])
    

    # Train model
    model.fit(X_train, y_train)

    # Predict hr for all activities
    st.session_state.df["hr_pred"] = model.predict(X)
    st.session_state.df["hr_pred_error"] = st.session_state.df["hr_pred"] - st.session_state.df["hr"]
    st.session_state.df["hr_mae"] = np.abs(st.session_state.df["hr_pred_error"])

    

    # Add column signifying whether row was used as training data, validation data, or it was unused because the hr was apriori labeled as unreliable
    def get_column_train_val_or_unused_due_to_unreliable_hr(row):
        if row["hr_acc_label"] != "perfect":
            return "unused_due_to_unreliable_hr"
        else:
            if row["activity_id"] in st.session_state.list_train_activity_ids:
                return "train"
            elif row["activity_id"] in st.session_state.list_val_activity_ids:
                return "val"
            else:
                raise ValueError(f'activity_id {row["activity_id"]} not in list_train_activity_ids {str(st.session_state.list_train_activity_ids)} or list_val_activity_ids {str(st.session_state.list_val_activity_ids)}.')

    st.session_state.df["train_val_or_unused_due_to_unreliable_hr"] = st.session_state.df.apply(get_column_train_val_or_unused_due_to_unreliable_hr, axis=1)

    display_ridge_parameters = False
    if display_ridge_parameters:
        st.session_state.ridge_model = model.named_steps['model']
        st.session_state.X = X

        # The following needs to be put outside of this function. Otherwise, it will disappear immediately
        st.write(st.session_state.ridge_model.coef_)
        st.write(st.session_state.ridge_model.intercept_)
        st.write(st.session_state.X.iloc[100:105,:])




    if False:
        # Add column with hr predictions
        df_only_correct_hr_rows_train["hr_pred"]= model.predict(X_train)
        df_only_correct_hr_rows_val["hr_pred"]= model.predict(X_val)

        # Add columns with hr error and mae
        df_only_correct_hr_rows_train["hr_pred_error"] = df_only_correct_hr_rows_train["hr_pred"] - df_only_correct_hr_rows_train["hr"]
        df_only_correct_hr_rows_val["hr_pred_error"] = df_only_correct_hr_rows_val["hr_pred"] - df_only_correct_hr_rows_val["hr"]
        df_only_correct_hr_rows_train["hr_mae"] = np.abs(df_only_correct_hr_rows_train["hr_pred_error"])
        df_only_correct_hr_rows_val["hr_mae"] = np.abs(df_only_correct_hr_rows_val["hr_pred_error"])

        # Create dataframes that include the hr prediction columns and all rows (even those for which the hr measurement was apriori labeled as unreliable)
        st.session_state.df_train = pd.merge(
            df_only_correct_hr_rows_train[["activity_id", "timestamp", "hr_pred", "hr_pred_error", "hr_mae"]],  
            st.session_state.df.query('activity_id in @st.session_state.list_train_activity_ids'),
            on=["activity_id", "timestamp"], 
            how="right"
            )
        st.session_state.df_val = pd.merge(
            df_only_correct_hr_rows_val[["activity_id", "timestamp", "hr_pred", "hr_pred_error", "hr_mae"]],  
            st.session_state.df.query('activity_id in @st.session_state.list_val_activity_ids'),
            on=["activity_id", "timestamp"], 
            how="right"
            )

        # Move column hr just before column hr_pred
        for df_temp in [st.session_state.df_train, st.session_state.df_val]:
            hr_col = df_temp.pop('hr')
            df_temp.insert(df_temp.columns.get_loc('hr_pred'), 'hr', hr_col)

        # Concatenate df_train and df_val
        st.session_state.df = pd.concat([
            st.session_state.df_train.assign(train_or_val="train"),
            st.session_state.df_val.assign(train_or_val="val")
        ], axis=0, ignore_index=True)

def train_model_and_predict_hr():
    # Splits the activities into training and validation activities
    # Trains a regression model 
    # Predicts the heart rate for the training and validation activities
    # Creates df_train and df_val that include the hr prediction columns and all rows (even those for which the hr measurement was apriori labeled as unreliable)

    # Extract data/information from df
    df_only_correct_hr_rows = st.session_state.df_without_hr_pred.query("hr_acc_label == 'perfect'")
    list_all_activity_ids = st.session_state.df_without_hr_pred['activity_id'].unique().tolist()

    # Get X_train, X_val, y_train, y_val, df_train, df_val from df_all_activities
    train_activities_ratio = 0.7
    feature_columns = ['cum_power', 'power100_ew_100s', 'power100_ew_30s', 'gaspeed4_ew_100s', 'gaspeed4_ew_30s']
    X_train, X_val, y_train, y_val, df_only_correct_hr_rows_train, \
        df_only_correct_hr_rows_val, st.session_state.list_train_activity_ids, st.session_state.list_val_activity_ids = \
        helper_sklearn.train_test_split_by_activity2(
            df_only_correct_hr_rows=df_only_correct_hr_rows,
            feature_columns=feature_columns,
            target_column='hr',
            list_all_activity_ids=list_all_activity_ids,
            train_activities_ratio=train_activities_ratio
        )


    # Define model
    model = sklearn.pipeline.Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()),
        ("model", sklearn.linear_model.Ridge(alpha=0.1))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Add column with hr predictions
    df_only_correct_hr_rows_train["hr_pred"]= model.predict(X_train)
    df_only_correct_hr_rows_val["hr_pred"]= model.predict(X_val)

    # Add columns with hr error and mae
    df_only_correct_hr_rows_train["hr_pred_error"] = df_only_correct_hr_rows_train["hr_pred"] - df_only_correct_hr_rows_train["hr"]
    df_only_correct_hr_rows_val["hr_pred_error"] = df_only_correct_hr_rows_val["hr_pred"] - df_only_correct_hr_rows_val["hr"]
    df_only_correct_hr_rows_train["hr_mae"] = np.abs(df_only_correct_hr_rows_train["hr_pred_error"])
    df_only_correct_hr_rows_val["hr_mae"] = np.abs(df_only_correct_hr_rows_val["hr_pred_error"])

    # Create dataframes that include the hr prediction columns and all rows (even those for which the hr measurement was apriori labeled as unreliable)
    st.session_state.df_without_hr_pred_train = pd.merge(
        df_only_correct_hr_rows_train[["activity_id", "timestamp", "hr_pred", "hr_pred_error", "hr_mae"]],  
        st.session_state.df_without_hr_pred.query('activity_id in @st.session_state.list_train_activity_ids'),
        on=["activity_id", "timestamp"], 
        how="right"
        )
    st.session_state.df_without_hr_pred_val = pd.merge(
        df_only_correct_hr_rows_val[["activity_id", "timestamp", "hr_pred", "hr_pred_error", "hr_mae"]],  
        st.session_state.df_without_hr_pred.query('activity_id in @st.session_state.list_val_activity_ids'),
        on=["activity_id", "timestamp"], 
        how="right"
        )

    # Move column hr just before column hr_pred
    for df_temp in [st.session_state.df_without_hr_pred_train, st.session_state.df_without_hr_pred_val]:
        hr_col = df_temp.pop('hr')
        df_temp.insert(df_temp.columns.get_loc('hr_pred'), 'hr', hr_col)

    # Concatenate df_train and df_val
    st.session_state.df = pd.concat([
        st.session_state.df_without_hr_pred_train.assign(train_or_val="train"),
        st.session_state.df_without_hr_pred_val.assign(train_or_val="val")
    ], axis=0, ignore_index=True)


def input_ui_choose_activity_id():
    # Adds st.session_state.activity_id to the session state

    with st.container(border=True):
        col1, col2, col3= st.columns([0.7, 3, 0.5], gap="medium")

        with col1:
            st.radio("View training or validation activity?", ["Train activity", "Validation activity"], key="radio_view_train_or_val_activity", label_visibility="collapsed") # View training or validation activity?

        with col2: 
            st.select_slider(
                'Select activity id to view',
                options=st.session_state.list_train_activity_ids if st.session_state.radio_view_train_or_val_activity == "Train" else st.session_state.list_val_activity_ids,
                key="activity_id"
            )
        
        with col3:
                
            def button_previous_activity_id_on_click():
                # Moves to the previous activity id
                if st.session_state.radio_view_train_or_val_activity == "Train":
                    list_train_or_val_activity_ids = st.session_state.list_train_activity_ids
                else:
                    list_train_or_val_activity_ids = st.session_state.list_val_activity_ids
                current_activity_id = st.session_state.activity_id
                current_activity_id_index = list_train_or_val_activity_ids.index(current_activity_id)
                previous_activity_id_index = (current_activity_id_index - 1) % len(list_train_or_val_activity_ids)
                st.session_state.activity_id = list_train_or_val_activity_ids[previous_activity_id_index]
            st.button("Previous", key="button_previous_activity_id", on_click=button_previous_activity_id_on_click)
            
            # Do the same with next instead of previous
            def button_next_activity_id_on_click():
                # Moves to the next activity id
                if st.session_state.radio_view_train_or_val_activity == "Train":
                    list_train_or_val_activity_ids = st.session_state.list_train_activity_ids
                else:
                    list_train_or_val_activity_ids = st.session_state.list_val_activity_ids
                current_activity_id = st.session_state.activity_id
                current_activity_id_index = list_train_or_val_activity_ids.index(current_activity_id)
                next_activity_id_index = (current_activity_id_index + 1) % len(list_train_or_val_activity_ids)
                st.session_state.activity_id = list_train_or_val_activity_ids[next_activity_id_index]
            st.button("Next", key="button_next_activity_id", on_click=button_next_activity_id_on_click)


def ui_display_activity_overview():
    # Displays the activity overview (distance, moving time, avg speed, avg hr) for the selected activity

    ser_summary_specific_activity = st.session_state.df_activities.loc[st.session_state.activity_id]
    cols_activity_overview = st.columns(4)
    cols_activity_overview[0].metric("Distance", f'{ser_summary_specific_activity.distance: .2f} km')
    cols_activity_overview[1].metric("Moving Time", 
                                        f'{ser_summary_specific_activity.moving_time.total_seconds()//3600:.0f}h {(ser_summary_specific_activity.moving_time.total_seconds()% 3600) // 60:.0f}min ')
    cols_activity_overview[2].metric("Avg Speed", f'{ser_summary_specific_activity.avg_speed: .2f} km/h')
    cols_activity_overview[3].metric("Avg HR", f'{ser_summary_specific_activity.avg_hr: .0f} bpm')

    # st.dataframe(ser_summary_specific_activity)


# @st.cache_data
def ui_get_plotly_timeseries_fig_v5(_df: pd.DataFrame, plot_fig: bool = False):
    df = _df
    fig = make_subplots(rows=1, cols=1, 
                        shared_xaxes=True, shared_yaxes=False, 
                        specs=[[{"secondary_y": True}],],
                        vertical_spacing=0.03
                        )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hr"], mode='lines', name='Heart Rate (measured)', line=dict(color='red', width=2)),
                  row=1,col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hr_pred"], mode='lines', name='Heart Rate (predicted)', line=dict(color='lime', width=4, dash="solid")),
                row=1,col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed"], mode='lines', name='Gradient Adjusted Pace [min/km]', line=dict(color='cornflowerblue', dash="dash")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gaspeed4_ew_100s"], mode='lines', name='Gradient Adjusted Pace (exp. smoothed with span=100s) [min/km]', line=dict(color='blue', dash="solid")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power"]/25, mode='lines', name='Power [W]', line=dict(color='lightpink', dash="dash")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power100_ew_100s"]/25, mode='lines', name='Power (exp. smoothed with span=100s) [W]', line=dict(color='coral')),
                row=1, col=1, secondary_y=True)

    # Ticks and lines for each tick
    array_pace_ticks = np.array([4,5,6,7,8,10])  # paces for which to display ticks
    array_hr_ticks = np.array([120, 140, 150, 160, 170, 180])  # hr values for which to display ticks
    fig.update_yaxes(
        gridcolor='deeppink',
        griddash=["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"][1],
        gridwidth=2,
        tickvals=array_hr_ticks,  # display ticks at these values
        secondary_y=False,
    )
    fig.update_yaxes(
        gridcolor='green',
        griddash = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"][3],
        gridwidth=2,
        tickvals=60/array_pace_ticks,  # convert pace to km/h
        ticktext=array_pace_ticks,
        secondary_y=True,
    )

    # Set y-axis titles/range
    fig.update_yaxes(range=[100, 200], row=1, col=1, secondary_y=False) 
    fig.update_yaxes(range=[0, 17], row=1, col=1, secondary_y=True) 
    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(range=[df["timestamp"].iloc[0], df["timestamp"].iloc[-1]])

    # Set interactive behaviour and layout
    # fig = helper_streamlit.update_screen_height_of_fig_v2(fig, height_factor=0.8, debug=False) # Does not work well with ptloyl_events. glitches
    fig = helper_streamlit.update_screen_height_of_fig_v3(
        fig=fig,
        screen_height=st.session_state.screen_height,
        screen_width=st.session_state.screen_width,
        height_factor=0.8,
        debug=False
    )
    fig.update_layout(
        dragmode='pan',  # zoom, pan, select, lasso
        hovermode='x unified',  # Enable unified hover mode across all traces
        uirevision='foo',  # Keep zoom level when updating layout
        # legend=dict(orientation="h")
        legend=dict(x=0.70, y=0.02),  # (0,0) denotes bottom left
        margin=dict(l=0, r=0, t=0, b=0),
    )

    if plot_fig:
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

    return fig





