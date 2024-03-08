from datetime import datetime, timedelta
import importlib
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import scipy

def get_spline_for_gap_computation():
    """
    Used to calculate GAP based of Grade and Pace (i.e. computes the speed correction factor)

    Usage: 
    spline_for_gap_computation = get_spline_for_gap_computation()
    df_resampled['gaspeed'] = df_resampled['speed'] * spline_for_gap_computation(df_resampled['grade'])

    Example: spline_for_gap_computation(df_resampled['grade']) ^=  1.6 slower slower when grade=12%

    """

    points_for_deriving_gap_formula = [(-100,3),(-35, 1.7), (-30, 1.49), (-26, 1.32), (-22, 1.17), (-20,1.085), (-17.4, 1),(-14, 0.965),(-12, 0.885),(-10, 0.87),(-9, 0.875), (-6, 0.89),(-4, 0.915),(-2, 0.96), (-1,0.983), (0,1), (2, 1.06), (3.2,1.1),(4, 1.135),(5.3, 1.2), (6, 1.225), (8, 1.33), (10,1.47), (12, 1.6),(16.8,2), (20,2.3),(26, 2.8), (28,2.99), (35, 3.55), (70,7.1), (105, 10.55)]#, (210, 21.1)] (-200, 5), (-120, 4), (-100, 4), (-80,3.3),

    df_for_deriving_gap_formula = pd.DataFrame(points_for_deriving_gap_formula, columns=['grade', 'speed_factor'])

    x = df_for_deriving_gap_formula['grade']
    y = df_for_deriving_gap_formula['speed_factor']

    spline_for_gap_computation = scipy.interpolate.CubicSpline(x, y, extrapolate='True')

    return spline_for_gap_computation
