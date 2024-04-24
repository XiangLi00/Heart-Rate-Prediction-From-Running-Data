from datetime import datetime, timedelta
import importlib
import math
import os
import sqlite3
import sys
import time

import bokeh
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
sys.path.append(os.path.join(os.getcwd(), 'streamlit_pages'))  
# from streamlit_pages._2024_03_10_annotate_fit_helper import test1
from streamlit_pages import _2024_03_10_annotate_fit_helper
from utils import helper_load_fit_file_v1, helper_load_specific_df, helper_pandas, helper_streamlit
# from utils.helper_load_df import load_df_v2, print_column_info_of_all_tables, get_column_info_of_specific_table, generate_report



def section_try_input_1_buttons(df: pd.DataFrame = None):

    with st.container(border=True):
        st.header("section_try_input_1_buttons")
        if "button11" not in st.session_state:
            st.session_state.button11 = False
        
        def on_button1_clicked():
            st.session_state.button11 = True

        def on_button33_reset():
            st.session_state.button11 = False

        button11 = st.button("Click me", help="This is a help text", disabled=False, key="button1", on_click=on_button1_clicked)
        button22 = st.button("Click me. Button 2", help="This is a help text", disabled=False, key="button2")
        button33_reset = st.button("Reset button11" , help="This is a help text", disabled=False, key="button3", on_click=on_button33_reset)


        st.write(f'button1: {st.session_state.button11} button 2: {button22} button 3: {button33_reset}')

def section_try_input_2():

    # Conclusion: checkboxes to do need session state
    # Reason: checkbox4 and st.session_state.checkbox4 are alwyas the same


    if not "checkbox2b" in st.session_state:
        st.session_state.checkbox2b = 0   # number of times checkbox was clicked to True

    if not "checkbox1" in st.session_state:
        st.session_state.checkbox1 = 1   # number of times checkbox was clicked to True

    if not "checkbox1" in st.session_state:
        st.session_state.checkbox4 = True  # does notihng. usefess
    
    if not "checkbox5" in st.session_state:
        st.session_state.checkbox5 = True

    # Expl: This written text will disppear when stremlit reruns it (e.g. when code changed)
    def checkbox1_on_change():
        st.session_state.checkbox1 += 1
    
    def checkbox2_on_change():
        st.write(checkbox2)
        if checkbox2 == True: # checkbox2 stores the old value
            st.session_state.checkbox2b += 1
        # st.write(f"checkbox2 was changed {st.session_state.checkbox2b} times")
    def checkbox4_on_change():
        # st.session_state.checkbox4 = checkbox4
        st.write(f'{st.session_state.checkbox4=}')

    button1  = st.button("Button 1 v6")
    checkbox1 = st.checkbox("Checkbox 1", on_change=checkbox1_on_change, key="checkbox1")
    checkbox2 = st.checkbox("Checkbox 2", label_visibility="visible", key="checkbox2", on_change=checkbox2_on_change)
    checkbox3 = st.checkbox("Checkbox 3", key="checkbox3", value=True)  # value ^= default first value
    checkbox5 = st.checkbox("Checkbox 5", key="checkbox5")  # value ^= default first value
    checkbox4 = st.checkbox("Checkbox 4", key="checkbox4", on_change=checkbox4_on_change)
    radio1 = st.radio("Radio1", [":rainbow[Apple]", "Banana", "Cherry"], key="radio1",
                      captions= ["Applehelp", "Bananahelp", "Cherryhelp"],
                      index=1)  # index is the default value


    st.write(f"{checkbox1=} {st.session_state.checkbox1=} \n {checkbox2=} {st.session_state.checkbox2b=} {button1=}")
    st.write(f'{checkbox3=}')
    st.text(f'{checkbox4=}        {st.session_state.checkbox4 = }')
    st.write(f'{checkbox5=}')
    st.write(f'{radio1=}')

    st.write(st.session_state)

def section_try_input_3():

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Disable radio widget", key="disabled")
        st.checkbox("Orient radio options horizontally", key="horizontal")

    with col2:
        st.radio(
            "Set label visibility 👇",
            ["visible", "hidden", "collapsed"],
            key="visibility",
            # label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            horizontal=st.session_state.horizontal,
        )

def section_try_input_4():
    start_color, end_color = st.select_slider(
        'Select a range of color wavelength',
        options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
        value=('orange', 'blue'))
    st.write('You selected wavelengths between', start_color, 'and', end_color)
section_try_input_4()
