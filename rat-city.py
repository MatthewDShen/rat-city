import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.express as px
import streamlit as st

st.markdown('# Rat City')

#############################################

st.markdown('### This website lets people see how restaurants effect the number of rat sightings in an area')
st.markdown('This program is currently only available for New York City')

#############################################

st.markdown('### On this page you can explore the data that we used to predict the number of rat sightings')

df = pd.read_csv('processed_data/feature_data.csv')

st.markdown('#### Explore Relationship with Rat Counts')
# Histogram
explore_feature_str = st.selectbox('What feature would you like to explore?', df.columns)
st.write(px.histogram(data_frame=df, x=explore_feature_str))

# Map
df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
fig, ax = plt.subplots()
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:2263')

gdf.plot(column=explore_feature_str, ax=ax, legend=True)
plt.title('{} of New York City'.format(explore_feature_str))
ax.axis('off')

st.pyplot(fig)
