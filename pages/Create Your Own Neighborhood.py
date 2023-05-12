import geopandas as gpd
import pandas as pd
import pickle
import streamlit as st



st.markdown('# Rat City')

#############################################

st.markdown('On this page you can build your own zipcode and see how many rats are predicited to show up')

#############################################


# TODO: Import ML model with pickle

def deploy_model(dict):
    rat_count=None
    model=st.session_state['deploy_model']

    # Add code here
    rat_count = model.predict(dict)

    return rat_count

# TODO: Import model data
df = pd.read_csv('data/processed_data/feature_data.csv')

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

    user_inputs = {}
            
    if 'scratch_bool' in st.session_state:
        # Confirm Zipcode
        if st.session_state['scratch_bool'] == False:
            st.write('You have selected to adjust {}'.format(zip_int))

        # Population
        if st.session_state['scratch_bool']:
            pop_user_select = st.number_input('Select Population', min_value = 0)
            user_inputs['population'] = pop_user_select
        else:
            pop_user_select = st.number_input('Select Population', min_value = 0, value = int(df.iloc[zip_index_int]['population']))
            user_inputs['population'] = pop_user_select

        # Area
        if st.session_state['scratch_bool']:
            area_user_select = st.number_input('Select Area (sqft)', min_value = 0.0)
            user_inputs['area'] = area_user_select
        else:
            area_user_select = st.number_input('Select Area (sqft)', min_value = 0.0, value = df.iloc[zip_index_int]['area'])
            user_inputs['area'] = area_user_select

        # Critical Flag
        if st.session_state['scratch_bool']:
            flags_user_select = st.number_input('Critical Flags', min_value = 0.0)
            user_inputs['critical flag'] = flags_user_select
        else:
            flags_user_select = st.number_input('Critical Flags', min_value = 0.0, value = df.iloc[zip_index_int]['critical flag'])
            user_inputs['critical flag'] = flags_user_select
        
        # Action
        if st.session_state['scratch_bool']:
            actions_user_select = st.number_input('Number of Actions', min_value = 0.0)
            user_inputs['action'] = actions_user_select
        else:
            actions_user_select = st.number_input('Number of Actions', min_value = 0.0, value = df.iloc[zip_index_int]['action'])
            user_inputs['action'] = actions_user_select

        # Average Score
        st.markdown('Choose a health rating from 0 (best) to 40 (worst):')
        if st.session_state['scratch_bool']:
            health_rating_user_select = st.slider('A(0-13) B(14-27) C(28-40)', min_value=0, max_value=40, value=20)
            user_inputs['avg score'] = health_rating_user_select
        else:
            health_rating_user_select = st.slider('A(0-13) B(14-27) C(28-40)', min_value=0, max_value=40, value=int(df.iloc[zip_index_int]['avg score']))
            user_inputs['avg score'] = health_rating_user_select
        
        # Sidewalk Dimensions (SQFT)
        if st.session_state['scratch_bool']:
            sidwalk_user_select = st.number_input('Sidewalk Dimensions (sqft)', min_value = 0.0)
            user_inputs['sidewalk dimensions (area)'] = sidwalk_user_select
        else:
            sidwalk_user_select = st.number_input('Sidewalk Dimensions (sqft)', min_value = 0.0, value = df.iloc[zip_index_int]['sidewalk dimensions (area)'])
            user_inputs['sidewalk dimensions (area)'] = sidwalk_user_select
        
        # Roadway Dimensions (SQFT)
        if st.session_state['scratch_bool']:
            roadway_user_select = st.number_input('Roadway Dimensions (sqft)', min_value = 0.0)
            user_inputs['roadway dimensions (area)'] = roadway_user_select
        else:
            roadway_user_select = st.number_input('Roadway Dimensions (sqft)', min_value = 0.0, value = df.iloc[zip_index_int]['roadway dimensions (area)'])
            user_inputs['roadway dimensions (area)'] = roadway_user_select

        col1, col2 = st.columns(2)
        with(col1):
            # Approved for Sidewalk Seating
            if st.session_state['scratch_bool']:
                sidewalk_seating_user_select = st.number_input('Number of Restaurants Approved for Sidewalk Seating', min_value = 0)
                user_inputs['approved for sidewalk seating'] = sidewalk_seating_user_select
            else:
                sidewalk_seating_user_select = st.number_input('Number of Restaurants Approved for Sidewalk Seating', min_value = 0, value = df.iloc[zip_index_int]['approved for sidewalk seating'])
                user_inputs['approved for sidewalk seating'] = sidewalk_seating_user_select
        with(col2):
            # Approved for Sidewalk Seating
            if st.session_state['scratch_bool']:
                roadway_seating_user_select = st.number_input('Number of Restaurants Approved for Roadway Seating', min_value = 0)
                user_inputs['approved for roadway seating'] = roadway_seating_user_select
            else:
                roadway_seating_user_select = st.number_input('Number of Restaurants Approved for Roadway Seating', min_value = 0, value = df.iloc[zip_index_int]['approved for roadway seating'])
                user_inputs['approved for roadway seating'] = roadway_seating_user_select
        
        # Qualify Alcohol
        if st.session_state['scratch_bool']:
            alc_user_select = st.number_input('Number of Restaurants that Serve Alcohol', min_value = 0)
            user_inputs['qualify alcohol'] = alc_user_select
        else:
            alc_user_select = st.number_input('Number of Restaurants that Serve Alcohol', min_value = 0, value = df.iloc[zip_index_int]['qualify alcohol'])
            user_inputs['qualify alcohol'] = alc_user_select
        
        # Total Number Restaurants
        if st.session_state['scratch_bool']:
            num_restaurants_user_select = st.number_input('Number of Restaurants', min_value = 0)
            user_inputs['total_number_restaurants'] = num_restaurants_user_select
        else:
            num_restaurants_user_select = st.number_input('Number of Restaurants', min_value = 0, value = df.iloc[zip_index_int]['total_number_restaurants'])
            user_inputs['total_number_restaurants'] = num_restaurants_user_select
        
        st.write(user_inputs)

    