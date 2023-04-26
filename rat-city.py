import os
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from sodapy import Socrata

st.markdown("# Title")

#############################################

st.markdown("### Title")

#############################################

st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse ac ultricies lacus. Proin gravida magna magna, quis consequat est ultricies sed. Sed quis lorem libero. Morbi vel sapien nibh.")

st.markdown("""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse ac ultricies lacus. Proin gravida magna magna, quis consequat est ultricies sed. Sed quis lorem libero. Morbi vel sapien nibh. Cras aliquet tempor dui ornare volutpat. Maecenas quam est, eleifend sit amet varius eu, semper nec urna. Aenean eleifend, magna sit amet vestibulum hendrerit, lectus leo tempus dui, in dictum leo mi in neque. Maecenas cursus nec purus eu pellentesque. Sed a augue aliquam, dictum urna nec, aliquam sapien.

Pellentesque porta elit ante, eget sagittis sem sagittis vel. Cras venenatis et sem nec suscipit. Nulla diam justo, dignissim non risus ut, aliquam venenatis est. Vivamus facilisis ac dolor et mollis. Nullam sodales, nisl nec interdum hendrerit, turpis orci dignissim erat, eget faucibus lorem nisl sit amet diam. Suspendisse lacinia id justo eget pellentesque. Suspendisse ut facilisis ligula. Duis sapien magna, lobortis nec volutpat in, commodo sed augue. Etiam augue lectus, lobortis eget dapibus ac, malesuada at felis. In lacinia dui sit amet odio volutpat euismod. Aliquam ornare nulla eu ligula eleifend, vitae euismod ante faucibus. Sed quis leo in sem pulvinar condimentum eget non eros.
""")

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

if ['df_rat'] not in st.session_state:
    df_rat = load_drive_data('https://drive.google.com/file/d/11aUJLdJqLDfVq_LAbyw3DDKxZYFbEZH0/view?usp=share_link')

if ['df_restaurants'] not in st.session_state:
    df_restaurant = load_drive_data('https://drive.google.com/file/d/1PywCUxGogJMA256H6v3BaFqgU-NCrIIb/view?usp=share_link')