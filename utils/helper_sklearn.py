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
    list_train_activities, list_val_activities = sklearn.model_selection.train_test_split(list_of_all_activity_ids, train_size=train_activities_ratio, random_state=seed)
    
    # Split X (resp. y) depending on activity_id (the first column of the multiindex column index)
    X_train = X.loc[X.index.get_level_values('activity_id').isin(list_train_activities)]
    X_val = X.loc[X.index.get_level_values('activity_id').isin(list_val_activities)]
    y_train = y.loc[y.index.get_level_values('activity_id').isin(list_train_activities)]
    y_val = y.loc[y.index.get_level_values('activity_id').isin(list_val_activities)]

    if return_list_train_val_activities:
        return X_train, X_val, y_train, y_val, list_train_activities, list_val_activities
    else:
        return X_train, X_val, y_train, y_val 