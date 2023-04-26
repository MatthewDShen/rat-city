import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


#############################################

st.markdown("# Rat Complaint Dataset")

#############################################

st.markdown("### Below you can find the rat complaints in NYC over the past month")

#############################################

def load_drive_data(url):
    '''
    Loads in csv from google drive and converts to pandas dataframe

    Input:
        - url: string with google drive link to csv
    Output:
        - df: dataframe of loaded csv with column names set to lower case
    '''
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url)

    return df

def str_to_time(df,time_feature):
    '''
    Converts string to datetime feature in dataframe

    Input:
        - df: dataframe
        - time_feature: string with the name of the time feature
    Output:
        - df: dataframe with converted datetime feature
    '''
    df[time_feature] = df[time_feature].apply(lambda x: x[:-12])
    
    df[time_feature] = pd.to_datetime(df[time_feature], errors='ignore', format='%m/%d/%y')
    
    df.dropna(subset = [time_feature], inplace = True)

    return df

def time_frame(df, time_feature, start_date, end_date):
    '''
    Appends dataframe so that it only includes times before and after the given window
    '''
    return df


def clean_map(df, lon, lat):
    '''
    Cleans up data from dataframe so it can be applied to a map

    Input:
        - df: dirty dataframe with longitude and latitude information
        - lon: string longitude feature name
        - lon: string latitude feature name
    Output:
        - datafram with that can be used to create a map
    '''
    
    df['lat'] = pd.to_numeric(df[lat])
    df['lon'] = pd.to_numeric(df[lon])

    df.dropna(subset = ['lat','lon'], inplace = True)

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

if ['df_rat'] not in st.session_state:
    df_rat = load_drive_data('https://drive.google.com/file/d/11aUJLdJqLDfVq_LAbyw3DDKxZYFbEZH0/view?usp=share_link')

if ['df_restaurants'] not in st.session_state:
    df_restaurant = load_drive_data('https://drive.google.com/file/d/1PywCUxGogJMA256H6v3BaFqgU-NCrIIb/view?usp=share_link')

if df_rat is not None and df_restaurant is not None:
    # Front End UI
    st.write(df_rat.head())
    
    df_rat = str_to_time(df_rat,'Created Date')

    start_date = st.date_input('Start point search') 
    end_date = st.date_input('End point search')

    if start_date <= end_date:
        df_rat = time_frame(df_rat, 'Created Date', start_date, end_date)
    else:
        st.write('Your start date must be after your end date')

        
    st.write('### Map of all rat citings within the past month in nyc')
    df_rat = clean_map(df_rat, 'Longitude', 'Latitude')
    st.map(df_rat)

    st.session_state['df_rat'] = df_rat
    st.session_state['df_restaurants'] = df_restaurant

    