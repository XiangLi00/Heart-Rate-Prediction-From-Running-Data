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
import sklearn
from sklearn import linear_model, pipeline, preprocessing, ensemble 


def train_test_split_by_activity(
    X: pd.DataFrame,
    y: pd.Series,
    list_of_all_activity_ids: List[int],
    train_activities_ratio: float = 0.7,
    seed: None|int = None,
    return_list_train_val_activities: bool = False
):
    # Divide into training and validation activities
    list_train_activity_ids, list_val_activity_ids = sklearn.model_selection.train_test_split(list_of_all_activity_ids, train_size=train_activities_ratio, random_state=seed)
    
    # Split X (resp. y) depending on activity_id (the first column of the multiindex column index)
    X_train = X.loc[X.index.get_level_values('activity_id').isin(list_train_activity_ids)]
    X_val = X.loc[X.index.get_level_values('activity_id').isin(list_val_activity_ids)]
    y_train = y.loc[y.index.get_level_values('activity_id').isin(list_train_activity_ids)]
    y_val = y.loc[y.index.get_level_values('activity_id').isin(list_val_activity_ids)]

    if return_list_train_val_activities:
        return X_train, X_val, y_train, y_val, list_train_activity_ids, list_val_activity_ids
    else:
        return X_train, X_val, y_train, y_val 
    

def train_test_split_by_activity2(
    df_only_correct_hr_rows: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    list_all_activity_ids: List[int],
    train_activities_ratio: float = 0.7,
    seed: None|int = None
):
    # Divide into training and validation activities
    list_train_activity_ids, list_val_activity_ids = sklearn.model_selection.train_test_split(list_all_activity_ids, train_size=train_activities_ratio, random_state=seed)

    df_only_correct_hr_rows_train = df_only_correct_hr_rows.query('activity_id in @list_train_activity_ids')
    df_only_correct_hr_rows_val = df_only_correct_hr_rows.query('activity_id in @list_val_activity_ids')
    
    X_train = df_only_correct_hr_rows_train[['activity_id', 'timestamp']+feature_columns].set_index(['activity_id', 'timestamp'])
    X_val = df_only_correct_hr_rows_val[['activity_id', 'timestamp']+feature_columns].set_index(['activity_id', 'timestamp'])
    y_train = df_only_correct_hr_rows_train.set_index(['activity_id', 'timestamp'])[target_column]
    y_val = df_only_correct_hr_rows_val.set_index(['activity_id', 'timestamp'])[target_column]

    return X_train, X_val, y_train, y_val, df_only_correct_hr_rows_train, df_only_correct_hr_rows_val, list_train_activity_ids, list_val_activity_ids