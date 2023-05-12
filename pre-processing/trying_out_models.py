# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
FILEPATH = "/Users/twallacech/Documents/GitHub/rat-city/processed_data/feature_data.csv"
features_df = pd.read_csv(FILEPATH, index_col=0)

# %%
features_df.head()

# %%
features_df["RAT COUNT"].hist()

# %%
# Transform the rat count to be closer to gaussian
(np.log(features_df["RAT COUNT"])).hist()

# %%
(features_df["RAT COUNT"] ** (1/3)).hist()

# %%
(features_df["RAT COUNT"] ** (1/4)).hist()

# %%
(features_df["RAT COUNT"] ** (1/5)).hist()

# %%
(features_df["RAT COUNT"] ** (1/10)).hist()

# %%
# Let's look at outliers
import plotly.express as px

# %%
px.box(
    features_df,
    'Roadway Dimensions (Area)'
)

# %%
features_df["RAT COUNT - Transformed"] = features_df["RAT COUNT"] ** (1/4)

# %%
import seaborn as sns

# %%
sns.pairplot(features_df[[
    "RAT COUNT - Transformed",
    "POPULATION",
    "SCORE",
    "CRITICAL FLAG",
    "ACTION",
    "AVG SCORE",
    "AVG FLAGS"
]])

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


linreg = LinearRegression()

x_features = [
    "POPULATION",
    "SCORE",
    "CRITICAL FLAG",
    "ACTION",
    "AVG SCORE",
    "AVG FLAGS"
]

y_feature = ["RAT COUNT - Transformed"]

test_size = 0.25

x_train, x_test, y_train, y_test = train_test_split(
    features_df[x_features],
    features_df[y_feature],
    test_size=test_size,
    random_state=42
)

linreg.fit(x_train, y_train)

# %%
from sklearn.metrics import mean_absolute_percentage_error

y_pred = linreg.predict(x_test)
y_pred = y_pred ** 4

print(mean_absolute_percentage_error(y_test ** 4, y_pred))

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(x_test.iloc[:, 1], y_pred, color='b')
plt.scatter(x_test.iloc[:, 1], y_test ** 4, color='g')
plt.show()

# %% [markdown]
# #### Linear Regression and 

# %%
import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt         # pip install matplotlib
import streamlit as st                  # pip install streamlit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import random
import plotly.express as px

# %%
def train_multiple_regression(x_train, y_train, regression_methods_options):
    """
    Fit a multiple regression model to data 

    Input: 
        - x_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
    Output: 
        - multi_reg_model: the trained multiple regression model
    """
    try:
        multi_reg_model = LinearRegression()
        multi_reg_model.fit(x_train, y_train)
  
    except Exception as e:
        st.write(f"An error occurred while training the Linear Regression model: {str(e)}")
    
     # Store model in st.session_state[model_name]
    st.session_state['Multiple Linear Regression'] = multi_reg_model

    return multi_reg_model

# %%
def train_polynomial_regression(x_train, y_train, regression_methods_options, poly_degree, poly_include_bias):
    """
    This function trains the model with polynomial regression

    Input: 
        - x_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
        - poly_degree: the degree of polynomial
        - poly_include_bias: whether or not to include bias
    Output: 
        - poly_reg_model: the trained model
    """
    poly_reg_model = None

    # Train model. Handle errors with try/except statement
    try:
        poly_reg_model = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=poly_degree, include_bias=poly_include_bias)), ('polyReg', LinearRegression())])
        poly_reg_model = poly_reg_model.fit(x_train, y_train)

    except Exception as e:
        st.write(f"An error occurred while training the Polynomial Regression model: {str(e)}")

    # Store model in st.session_state[model_name]
    st.session_state['Polynomial Regression'] = poly_reg_model

    return poly_reg_model

# %%
def train_ridge_regression(x_train, y_train, regression_methods_options, ridge_params, ridge_cv_fold):
    """
    This function trains the model with ridge regression and cross-validation

    Input: 
        - x_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
        - ridge_params: a dictionary of the hyperparameters to tune during cross validation
        - ridge_cv_fold: the number of folds for cross validation
    Output: 
        - ridge_cv: the trained model
    """
    # Train model. Handle errors with try/except statement
    ridge_cv = None

    try:
        ridge_cv = Pipeline([('scaler', StandardScaler()), ('ridgeCV', GridSearchCV(estimator=Ridge(), param_grid=ridge_params, cv=ridge_cv_fold))])
        ridge_cv.fit(x_train, y_train)

    except Exception as e:
        st.write(f"An error occurred while training the Ridge Regression model: {str(e)}")

    # Store model in st.session_state[model_name]
    st.session_state["Ridge Regression"] = ridge_cv
    return ridge_cv

# %%
def train_lasso_regression(x_train, y_train, regression_methods_options, lasso_params, lasso_cv_fold):
    """
    This function trains the model with lasso regression and cross-validation

    Input: 
        - x_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
        - lasso_params: a dictionary of the hyperparameters to tune during cross validation
        - lasso_cv_fold: the number of folds for cross validation
    Output: 
        - lasso_cv: the trained model
    """
    lasso_cv = None
    try:
        lasso_cv = Pipeline([('scaler', StandardScaler()), ('lassoCV', GridSearchCV(estimator=Lasso(), param_grid=lasso_params, cv=lasso_cv_fold))])
        lasso_cv.fit(x_train, y_train)
    
    except Exception as e:
        st.write(f"An error occurred while training the Lasso Regression model: {str(e)}")

    # Store model in st.session_state[model_name]
    st.session_state["Lasso Regression"] = lasso_cv
    return lasso_cv

# %%
st.session_state['target'] = 'RAT COUNT'

regression_methods_options = ['Multiple Linear Regression',
                                'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
# Collect ML Models of interests
regression_model_select = st.multiselect(
    label='Select regression model for prediction',
    options=regression_methods_options,
)
st.write('You selected the follow models: {}'.format(
    regression_model_select))

###################### TRAIN REGRESSION MODELS #######################
# # Add parameter options to each regression method

# Multiple Linear Regression
if (regression_methods_options[0] in regression_model_select):
    st.markdown('#### ' + regression_methods_options[0])
    if st.button('Train Multiple Linear Regression Model'):
        train_multiple_regression(
            x_train, y_train, regression_methods_options)

    if regression_methods_options[0] not in st.session_state:
        st.write('Multiple Linear Regression Model is untrained')
    else:
        st.write('Multiple Linear Regression Model trained')

# Polynomial Regression
if (regression_methods_options[1] in regression_model_select):
    st.markdown('#### ' + regression_methods_options[1])

    poly_degree = st.number_input(
        label='Enter the degree of polynomial',
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        key='poly_degree_numberinput'
    )
    st.write('You set the polynomial degree to: {}'.format(poly_degree))

    poly_include_bias = st.checkbox('include bias')
    st.write('You set include_bias to: {}'.format(poly_include_bias))

    if st.button('Train Polynomial Regression Model'):
        train_polynomial_regression(
            x_train, y_train, regression_methods_options, poly_degree, poly_include_bias)

    if regression_methods_options[1] not in st.session_state:
        st.write('Polynomial Regression Model is untrained')
    else:
        st.write('Polynomial Regression Model trained')

# Ridge Regression
if (regression_methods_options[2] in regression_model_select):
    st.markdown('#### ' + regression_methods_options[2])
    ridge_cv_fold = st.number_input(
        label='Enter the number of folds of the cross validation',
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key='ridge_cv_fold_numberinput'
    )
    st.write('You set the number of folds to: {}'.format(ridge_cv_fold))

    solvers = ['auto', 'svd', 'cholesky', 'lsqr',
                'sparse_cg', 'sag', 'saga', 'lbfgs']
    ridge_solvers = st.multiselect(
        label='Select solvers for ridge regression',
        options=solvers,
        default=solvers[0],
        key='ridge_reg_solver_multiselect'
    )
    st.write('You select the following solver(s): {}'.format(ridge_solvers))

    ridge_alphas = st.text_input(
        label='Input a list of alpha values, separated by comma',
        value='1.0,0.5',
        key='ridge_alphas_textinput'
    )
    st.write('You select the following alpha value(s): {}'.format(ridge_alphas))

    ridge_params = {
        'solver': ridge_solvers,
        'alpha': [float(val) for val in ridge_alphas.split(',')]
    }

    if st.button('Train Ridge Regression Model'):
        train_ridge_regression(
            x_train, y_train, regression_methods_options, ridge_params, ridge_cv_fold)

    if regression_methods_options[2] not in st.session_state:
        st.write('Ridge Model is untrained')
    else:
        st.write('Ridge Model trained')

# Lasso Regression
if (regression_methods_options[3] in regression_model_select):
    st.markdown('#### ' + regression_methods_options[3])
    lasso_cv_fold = st.number_input(
        label='Enter the number of folds of the cross validation',
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key='lasso_cv_fold_numberinput'
    )
    st.write('You set the number of folds to: {}'.format(lasso_cv_fold))

    lasso_tol = st.text_input(
        label='Input a list of tolerance values, separated by comma',
        value='0.001,0.0001',
        key='lasso_tol_textinput'
    )
    st.write('You select the following tolerance value(s): {}'.format(lasso_tol))

    lasso_alphas = st.text_input(
        label='Input a list of alpha values, separated by comma',
        value='1.0,0.5',
        key='lasso_alphas_textinput'
    )
    st.write('You select the following alpha value(s): {}'.format(lasso_alphas))

    lasso_params = {
        'tol': [float(val) for val in lasso_tol.split(',')],
        'alpha': [float(val) for val in lasso_alphas.split(',')]
    }

    if st.button('Train Lasso Regression Model'):
        train_lasso_regression(
            x_train, y_train, regression_methods_options, lasso_params, lasso_cv_fold)

    if regression_methods_options[3] not in st.session_state:
        st.write('Lasso Model is untrained')
    else:
        st.write('Lasso Model trained')

regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
trained_models = [
    model for model in regression_methods_options if model in st.session_state]
st.session_state['trained_models'] = trained_models

# Select a model to deploy from the trained models
st.markdown("## Choose your Deployment Model")
model_select = st.selectbox(
    label='Select the model you want to deploy',
    options=st.session_state['trained_models'],
)

if (model_select):
    st.write('You selected the model: {}'.format(model_select))
    st.session_state['deploy_model'] = st.session_state[model_select]


import streamlit as st
import pandas as pd                     
import numpy as np
import random
from sklearn.preprocessing import OrdinalEncoder
random.seed(10)
#############################################

st.markdown("### Final Project - Predicting number of rat complaints based on number of restaurants in an area and the average health inspection grade in the area, using Regression")

#############################################

st.title('Deploy Application')

#############################################
enc = OrdinalEncoder()

# Checkpoint 11
def deploy_model(df):
    """
    Deploy trained regression model trained on 
    Input: 
        - df: pandas dataframe with trained regression model
    Output: 
        - rat_count: predicted rat count
    """
    rat_count = None

    if 'deploy_model' in st.session_state:
        model = st.session_state['deploy_model']

        if model is not None:
            rat_count = model.predict(df)

    return rat_count

# Helper Function
def is_valid_input(input):
    """
    Check if the input string is a valid integer or float.

    Input: 
        - input: string, char, or input from a user
    Output: 
        - True if valid input; otherwise False
    """
    try:
        num = float(input)
        return True
    except ValueError:
        return False
    
# Helper Function
def decode_integer(original_df, decode_df, feature_name):
    """
    Decode integer integer encoded feature

    Input: 
        - original_df: pandas dataframe with feature to decode
        - decode_df: dataframe with feature to decode 
        - feature: feature to decode
    Output: 
        - decode_df: Pandas dataframe with decoded feature
    """
    original_df[[feature_name]]= enc.fit_transform(original_df[[feature_name]])
    decode_df[[feature_name]]= enc.inverse_transform(st.session_state['x_train'][[feature_name]])
    return decode_df



###################### Deploy App #######################

if features_df is not None:
    st.markdown('### Interested in exterminating rats? Predict number of rat complaints based on number of restaurants in an area and the average health inspection grade in the area.')
    st.session_state['x_train'] = x_train

    # Input users input features
    user_input={}

    # Input number of restaurants
    st.markdown('## How many restaurants are in your area')
    number_of_restaurants = st.number_input('Insert the number of restaurants you would like', min_value=1, max_value=100, value=1)
    user_input['number_restaurants_input'] = number_of_restaurants
    st.write('You selected {} restaurants'.format(number_of_restaurants))

    # Input health inspection rating
    st.markdown('## What is their average health rating')
    st.markdown('Choose a health rating from 0 (best) to 40 (worst):')
    health_rating = st.slider('A(0-13) B(14-27) C(28-40)', 1, 40, 1)
    if(health_rating):
        user_input['health_rating_input'] = health_rating
        st.write('You selected a health rating of {}'.format(health_rating))
    
    # Create a DataFrame from the selected features dictionary
    selected_features_df = pd.DataFrame.from_dict(user_input, orient='index').T

    # To get the mean value for unused data
    for col in st.session_state['x_train'].columns:
        if(col not in selected_features_df.columns):
            selected_features_df[col]= st.session_state['x_train'][col].mean()
    
    # Select column order of main DataFrame
    main_df_col_order = st.session_state['x_train'].columns.tolist()
    
    # Reindex the selected_features_df DataFrame with the column order of the Main DataFrame
    selected_features_df = selected_features_df.reindex(columns=main_df_col_order)
    st.write("# Predict Rat Complaints")

    ###################### Predict Rat Complaints #######################
    rat_count=None
    if('deploy_model' in st.session_state and st.button('Predict Rat Complaints')):
        rat_count = deploy_model(selected_features_df)
        if(rat_count is not None):
            # Display rat count
            st.write('The number of rat complaints is {0:.2f}'.format(rat_count[0][0]))


