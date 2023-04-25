import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from sodapy import Socrata


#############################################

st.markdown("# Rat Complaint Dataset")

#############################################

st.markdown("### Below you can find the rat complaints in NYC over the past month")

#############################################

def load_rat_data_drive():
    url = 'https://drive.google.com/file/d/11aUJLdJqLDfVq_LAbyw3DDKxZYFbEZH0/view?usp=share_link'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url)

    return df

# Helper function
@st.cache
def convert_df(df):
    '''
    Cache the conversion to prevent computation on every rerun

    Input: 
        - df: pandas dataframe
    Output: 
        - Save file to local file system
    '''
    return df.to_csv().encode('utf-8')

###################### FETCH DATASET #######################



df = load_data('cvf2-zn8s')

print(df)

if df is not None:
    # Front End UI
    st.write(df.head())
    