import geopandas as gpd
import pandas as pd
import pickle
import streamlit as st



st.markdown('# Rat City')

#############################################

st.markdown('On this page you can build your own zipcode and see how many rats are predicited to show up')

#############################################


# TODO: Import ML model with pickle

# TODO: Import model data
df = pd.read_csv('data/processed_data/feature_data.csv')


st.write(df)

if df is not None:
    st.markdown('#### Select how you would like to build your neighborhood')

    ##### Pick starting zipcode #####
    col1, col2 = st.columns(2)
    scratch_bool = True
    ### Starting from a blank neighborhood
    with(col1):
        if st.button('Create Your Own Neighborhood from Scratch'):
            scratch_bool = True
            st.write('You have selected to create your own neighborhood')
            
    ### Base your neighborhood from an existing zipcode        
    with(col2):
        zip_int = st.selectbox('Select Existing Zipcode', df['zipcode'])
        zip_index_int = df[df['zipcode'] == zip_int].index[0]
        
        if (zip_int and st.button('Confirm Zipcode')):
            scratch_bool = False
            st.write('You have selected to adjust {}'.format(zip_int))

    # Show current state of zipcode
    if scratch_bool == False:
        st.write('Zipcode {} Currently Information'.format(zip_int))
        st.write('Population: {}'.format(df.iloc[zip_index_int]['population estimate']))

    # Population
    if scratch_bool:
        st.number_input('Select Population', min_value = 0)
    else:
        st.number_input('Select Population', min_value = 0, value = df.iloc[zip_index_int]['population estimate'])

    # col1, col2 = st.columns(2)

    # with(col1):
    #     # Number of Restaurants
    #     if scratch_bool:
    #         num_rest_int = st.number_input('Select Number of Restaurants', min_value = 0)
    #     else:
    #         num_rest_int = st.number_input('Select Number of Restaurants', min_value = 0, value = df.iloc[zip_index_int]['Number of Resaurants'])
            
    # with(col2):
    #     # Average restaurant rating
    #     if scratch_bool:
    #         st.number_input('Select the Average Restaurant Score')
    

    