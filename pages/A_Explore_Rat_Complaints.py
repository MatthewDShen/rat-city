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

# Insert Functions Here
def load_data(endpoint):
    '''
    Loads in data from NYC Open Data

    Input:
        - Endpoint (example: "a0aa-aaaa")
        - Query in SQL format

    Output:
        - Dataframe with information from client
    '''
    # Set up endpoint
    endpoint = endpoint

    # Load api key
    load_dotenv()

    # Get client information
    client = Socrata('data.cityofnewyork.us',
                    os.getenv('ny_app_token'),
                    username = os.getenv('ny_api_key_id'),
                    password = os.getenv('ny_api_key_secret'))
    
    # Get total number of records in api
    # query_count = "SELECT COUNT(*)"
    # NUM_RECORDS = int(client.get(endpoint, query = query_count)[0]['COUNT'])
    NUM_RECORDS = 2000
    
    query = f"""
        SELECT *
        LIMIT {NUM_RECORDS}
    """
    
    # Get results from client
    results = client.get(endpoint, query=query)

    # Change results into dataframe
    df = pd.DataFrame.from_records(results)

    return df

print(load_data('cvf2-zn8s'))

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

if df is not None:
    # Front End UI
    st.write(df)
    