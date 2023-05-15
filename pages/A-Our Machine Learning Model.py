import pandas as pd
import streamlit as st

st.markdown('# Rat City')

#############################################

st.markdown('### On this page we explore how we built our model')

#############################################

df = pd.read_csv('data/processed_data/feature_data.csv')

st.markdown('#### Model Data')

st.write(df)

st.markdown('#### Model Variables')
st.markdown('##### Output Variable')
st.markdown('- rat count')
st.markdown('##### Input Variable')
st.markdown('- population')
st.markdown('- critical flags')
st.markdown('- sidewalk dimensions (sqft)')
st.markdown('- roadway dimensions (sqft)')
st.markdown('- number of restaurants approved with sidewalk seating')
st.markdown('- number of restaurants approved for roadway seating')
st.markdown('- number of restaurants that serve alcohol')
st.markdown('- number of restaurants')

st.markdown('#### Train/Test Split')
st.markdown('- 75% of data used for training')
st.markdown('- 25% used for validation')

st.markdown('#### Models Tested')
st.markdown('- Multiple Linear Regression')
st.markdown('- Polynomial Regression (2 degrees)')
st.markdown('- Ridge Regression (5 cv folds)')
st.markdown('- Lasso Regression (5 cv folds)')

st.markdown('#### Evaluation')
st.image('model_training/error_plots/polynomial regression.png')
st.image('model_training/error_plots/multiple linear regression.png')
st.image('model_training/error_plots/ridge regression.png')
st.image('model_training/error_plots/lasso regression.png')
st.markdown('- multiple linear regression, ridge, and lasso has similar values so overfitting is unlikley')

st.markdown('#### Final Selection')
st.markdown('##### Lasso Regression (5 cv folds)')
