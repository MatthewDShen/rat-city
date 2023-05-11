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
    ### Starting from a blank neighborhood
    with(col1):
        if st.button('Create Your Own Neighborhood from Scratch'):
            st.session_state['scratch_bool'] = True
            st.write('You have selected to create your own neighborhood')
            
    ### Base your neighborhood from an existing zipcode        
    with(col2):
        zip_int = st.selectbox('Select Existing Zipcode', df['zipcode'])
        zip_index_int = df[df['zipcode'] == zip_int].index[0]
        
        if (zip_int and st.button('Confirm Zipcode')):
            st.session_state['scratch_bool'] = False
            

    if 'scratch_bool' in st.session_state:
        # Confirm Zipcode
        if st.session_state['scratch_bool'] == False:
            st.write('You have selected to adjust {}'.format(zip_int))

        # Population
        if st.session_state['scratch_bool']:
            st.number_input('Select Population', min_value = 0)
        else:
            st.number_input('Select Population', min_value = 0, value = int(df.iloc[zip_index_int]['population']))

        # Area
        if st.session_state['scratch_bool']:
            st.number_input('Select Area (sqft)', min_value = 0.0)
        else:
            st.number_input('Select Area (sqft)', min_value = 0.0, value = df.iloc[zip_index_int]['area'])

        # Critical Flag
        if st.session_state['scratch_bool']:
            st.number_input('Critical Flags', min_value = 0.0)
        else:
            st.number_input('Critical Flags', min_value = 0.0, value = df.iloc[zip_index_int]['critical flag'])
        
        # Action
        if st.session_state['scratch_bool']:
            st.number_input('Number of Actions', min_value = 0.0)
        else:
            st.number_input('Number of Actions', min_value = 0.0, value = df.iloc[zip_index_int]['action'])

        # Average Score
        st.markdown('Choose a health rating from 0 (best) to 40 (worst):')
        if st.session_state['scratch_bool']:
            st.slider('A(0-13) B(14-27) C(28-40)', min_value=0, max_value=40, value=20)
        else:
            st.slider('A(0-13) B(14-27) C(28-40)', min_value=0, max_value=40, value=int(df.iloc[zip_index_int]['avg score']))
        
        # Sidewalk Dimensions (SQFT)
        if st.session_state['scratch_bool']:
            st.number_input('Sidewalk Dimensions (sqft)', min_value = 0.0)
        else:
            st.number_input('Sidewalk Dimensions (sqft)', min_value = 0.0, value = df.iloc[zip_index_int]['sidewalk dimensions (area)'])
        
        # Roadway Dimensions (SQFT)
        if st.session_state['scratch_bool']:
            st.number_input('Roadway Dimensions (sqft)', min_value = 0.0)
        else:
            st.number_input('Roadway Dimensions (sqft)', min_value = 0.0, value = df.iloc[zip_index_int]['roadway dimensions (area)'])

        col1, col2 = st.columns(2)
        with(col1):
            # Approved for Sidewalk Seating
            if st.session_state['scratch_bool']:
                st.number_input('Number of Restaurants Approved for Sidewalk Seating', min_value = 0)
            else:
                st.number_input('Number of Restaurants Approved for Sidewalk Seating', min_value = 0, value = df.iloc[zip_index_int]['approved for sidewalk seating'])
        with(col2):
            # Approved for Sidewalk Seating
            if st.session_state['scratch_bool']:
                st.number_input('Number of Restaurants Approved for Roadway Seating', min_value = 0)
            else:
                st.number_input('Number of Restaurants Approved for Roadway Seating', min_value = 0, value = df.iloc[zip_index_int]['approved for roadway seating'])
        
        # Qualify Alcohol
        if st.session_state['scratch_bool']:
            st.number_input('Number of Restaurants that Serve Alcohol', min_value = 0)
        else:
            st.number_input('Number of Restaurants that Serve Alcohol', min_value = 0, value = df.iloc[zip_index_int]['qualify alcohol'])
        
        # Total Number Restaurants
        if st.session_state['scratch_bool']:
            st.number_input('Number of Restaurants', min_value = 0)
        else:
            st.number_input('Number of Restaurants', min_value = 0, value = df.iloc[zip_index_int]['total_number_restaurants'])

    