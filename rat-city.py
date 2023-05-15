import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

st.markdown('# Rat City')

#############################################

st.markdown('### A Machine Learning tool to help the Rat Czar clean the streets of NYC by better predicting where rats will be')
st.markdown('"Fighting crime, fighting inequiality, and fighting rats." - Mayor Eric Adams')
st.markdown('[Rats](https://www.nytimes.com/2023/04/12/nyregion/rat-czar-kathleen-corradi.html) [are](https://www.nytimes.com/2021/11/08/nyregion/an-urban-problem-rats-on-the-rise.html) [a](https://www.frontiersin.org/articles/10.3389/fevo.2019.00013/full) [major](https://www.google.com/url?q=https://royalsocietypublishing.org/doi/10.1098/rspb.2018.0245&sa=D&source=editors&ust=1684113286168462&usg=AOvVaw2ssovpBBcY7KyO9YnhmDKT) [problem](https://www.wsj.com/articles/new-york-city-has-two-million-rats-and-one-new-rat-czar-35780bd5) [for](https://www.theatlantic.com/science/archive/2017/11/rats-of-new-york/546959/) [New York City!](https://a816-health.nyc.gov/ABCEatsRestaurants/#!/faq)')

st.markdown('One of the first applications of ML to understand rat locations')

#############################################

st.markdown('### On this page you can explore the data that we used to predict the number of rat sightings')

df = pd.read_csv('data/processed_data/feature_data.csv')

if df is not None:
    st.markdown('### Original Datasets')

    st.markdown('- [Restaurant Inspection by DOHMH](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)')
    st.markdown('- [Open Restaurant Applications](https://data.cityofnewyork.us/Transportation/Open-Restaurant-Applications/pitm-atqc)')   
    st.markdown('- [Rat Complaints through 311](https://data.cityofnewyork.us/City-Government/311-Call-Center-Inquiry/wewp-mm3p)')
    st.markdown('- [Zipcode shapefile (for plotting)](https://data.cityofnewyork.us/Business/Zip-Code-Boundaries/i8iw-xf4u/data)')
    

    st.markdown('### Preprocessed Data')
    st.write(df)

    st.markdown('#### Explore Features')

    # Select Feature
    explore_feature_lst = ['zipcode', 'rat count', 'population', 'avg score', 'critical flag', 'sidewalk dimensions (area)', 'roadway dimensions (area)', 'approved for sidewalk seating', 'approved for roadway seating', 'qualify alcohol', 'total_number_restaurants']
    explore_feature_str = st.selectbox('What feature would you like to explore?', explore_feature_lst)

    # Histogram
    st.write(px.histogram(data_frame=df, x=explore_feature_str))

    # Map
    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    fig, ax = plt.subplots()
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:2263')
    gdf.plot(column=explore_feature_str, ax=ax, legend=True)
    plt.title('New York City {}'.format(explore_feature_str))
    ax.axis('off')
    st.pyplot(fig)

    st.markdown('##### Correlations with rat count')
    corr_float = round(df['rat count'].corr(df[explore_feature_str]),3)
    st.write('The correlation between rat count and {0} is {1}'.format(explore_feature_str, corr_float))

    st.write('#### Data Cleaning Process')
    st.write('- Removed irrelevant, redundant columns ')
    st.write('- Filtered out rows with missing or messy data')
    st.write('- Filtered for 2022 inspections only')
    st.write('- Joined open restaurant with 2022 restaurants')
    st.write('- Groupby zip code calculate sums and averages for features')

    # Store Dataframe
    st.session_state['df'] = df
