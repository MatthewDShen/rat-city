import pandas as pd
import pickle
import streamlit as st



st.markdown('# Rat City')

#############################################

st.markdown('### The objective of this project is to let people see how changes in restaurants effect the number of rat sightings in an area')
st.markdown('### This program is currently only available for New York City')

#############################################


# TODO: Import ML model with pickle

# TODO: Import model data
data_df = pd.DataFrame({'zip code number': [10001,12001], # test df
                        'population estimate': [1000,5000],
                        'Number of Resaurants': [50,100],
                       })


st.markdown('#### Select how you would like to build your neighborhood')
## Pick starting zipcode ##
col1, col2 = st.columns(2)
scratch_bool = True
### Starting from a blank neighborhood
with(col1):
    if st.button('Create Your Own Neighborhood from Scratch'):
        scratch_bool = True
        st.write('You have selected to create your own neighborhood')
        
### Base your neighborhood from an existing zipcode        
with(col2):
    zip_int = st.number_input('Adjust Existing Zipcode', min_value = min(data_df['zip code number']), max_value = max(data_df['zip code number']))
    zip_index_int = data_df[data_df['zip code number'] == zip_int].index[0]
    
    if (zip_int and st.button('Confirm Zipcode')):
        scratch_bool = False
        st.write('You have selected to adjust {}'.format(zip_int))

# Show current state of zipcode
if scratch_bool == False:
    st.write('Zipcode {} Currently Information'.format(zip_int))
    st.write('Population: {}'.format(data_df.iloc[zip_index_int]['population estimate']))

# Population
if scratch_bool:
    st.number_input('Select Population', min_value = 0)
else:
    st.number_input('Select Population', min_value = 0, value = data_df.iloc[zip_index_int]['population estimate'])

col1, col2 = st.columns(2)

with(col1):
    # Number of Restaurants
    if scratch_bool:
        num_rest_int = st.number_input('Select Number of Restaurants', min_value = 0)
    else:
        num_rest_int = st.number_input('Select Number of Restaurants', min_value = 0, value = data_df.iloc[zip_index_int]['Number of Resaurants'])
        
with(col2):
    # Average restaurant rating
    if scratch_bool:
        st.number_input('Select the Average Restaurant Score')
    

    
#Total inputs
# Restaurant grade
# Cuisine
# Sidewalk Seating
# Roadway Seating
# Qualify Alcohol
# Sidewalk Area
# Roadway Area