import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import tarfile
import urllib.request

# Example Data Addition
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


#############################################

st.markdown("# Lorem ipsum")

#############################################

st.markdown("### Lorem ipsum dolor sit amet")

#############################################

st.markdown('# Lorem ipsum dolor sit amet')

#############################################

st.markdown('### Lorem ipsum dolor sit amet')

# Insert Functions Here
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): # Example fetching housing data
    """
    This function fetches a dataset from a URL, saves it in .tgz format, and extracts it to a specified directory path.

    Inputs:
    - housing_url (str): The URL of the dataset to be fetched.
    - housing_path (str): The path to the directory where the extracted dataset should be saved.

    Outputs: None
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Helper function
@st.cache
def convert_df(df):
    """
    Cache the conversion to prevent computation on every rerun

    Input: 
        - df: pandas dataframe
    Output: 
        - Save file to local file system
    """
    return df.to_csv().encode('utf-8')

###################### FETCH DATASET #######################
df = None

if df is not None:
    # Front End UI
    df = None
    