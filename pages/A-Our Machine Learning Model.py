import pandas as pd
import streamlit as st
import pickle

st.markdown('# Rat City')

#############################################

st.markdown('### On this page we explore how we built our model')

#############################################

df = pd.read_csv('data/processed_data/feature_data.csv')
model_linear = pickle.load(open('model_training/trained_linear_model.pickle', 'rb'))
model_ridge = pickle.load(open('model_training/trained_ridge_model.pickle', 'rb'))
model_lasso = pickle.load(open('model_training/trained_lasso_model.pickle', 'rb'))

models = {
    'Multiple Linear Regression': model_linear,
    'Ridge Regression': model_ridge,
    'Lasso Regression': model_lasso,
}

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

st.markdown('#### Model Selection')
st.markdown('##### Coefficents from model')
features_lst = ['population', 'avg score', 'critical flag', 'sidewalk dimensions (area)', 'roadway dimensions (area)', 'approved for sidewalk seating', 'approved for roadway seating', 'qualify alcohol', 'total_number_restaurants']
coeff_df = pd.DataFrame(model_linear.coef_, index = features_lst, columns = ['Multiple Linear Regression'])
coeff_df['Ridge Regression'] = model_ridge[-1].best_estimator_.coef_
coeff_df['Lasso Regression'] = model_lasso[-1].best_estimator_.coef_

st.write(coeff_df)

model_select = st.selectbox('What model would you like to use', ['Multiple Linear Regression', 'Polynomial Regression', 'Ridge Regression', 'Lasso Regression'])

if (model_select):
    st.session_state['model'] = models[model_select]

st.markdown('##### Our Recommendation is Lasso Regression (5 cv folds)')
st.markdown('- Multiple Linear Regression, Ridge, and Lasso had similar error values so we selected lasso because it would be the easiest to add new features to incase the ratczar managed to get more data in the future')
st.markdown('- Because population, approval for sidewalk seating, and the number of restaurants that qualify for alcohol are the only features with non-zero coefficents changing the other feature inputs will not have an effect on the final results once deployed')
st.markdown('- Based on our model population is the best indicator of rat count because it has the highest value')
st.markdown('- Our model also showed that number of restaurants that qualify for alcohol is the least correlated value that is non-zero')


