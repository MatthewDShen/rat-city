import pandas as pd
import pickle
import streamlit as st



st.markdown('# Rat City')

#############################################

st.markdown('### This website lets people see how restaurants effect the number of rat sightings in an area')
st.markdown('This program is currently only available for New York City')

#############################################

st.markdown('### On this page you can explore the data that we used to predict the number of rat sightings')

data_df = pd.DataFrame({'zip code number': [10001,12001], # test df
                        'population estimate': [1000,5000],
                        'Number of Resaurants': [50,100],
                       })

st.selectbox()