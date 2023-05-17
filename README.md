# rat-city
Develop a model to predict the number of rats based on restaurant numbers and quality data. Model will be trained on the entire city. Users can then select a single zip and change the number of restaurants and ratings. App will then show how the number of rats will change due to changes in the restaurant landscape.

The mayor is interested in taming the rat population, so this is a relevant project to the city's goals and can have practical applications to the city. It will identify if there are correaltions between specific restaurant features and rat complaints

Technical: regression methods allow us to find relationships between features and the rat population and also evaluate the relative impact of different features on the population

Novelty of approach: combining open-source datasets and tailoring them to focus on rat populations

Impact: Rats cause health problems and tear open trash on streets and make city dirtier, so being able to predict where rats will be will be helpful. Being able to predict amount of rats could help better direct city funds to appropriate locations to pest control

We haven’t found rat machine learning papers relevant to our project (only papers discussing rats and machine learning have been in medical contexts)

# Data Collection, Exploration, & Processing
Datasets using - all from open data, constantly updated, but looking at 2022:
- 311 Service Requests from 2010 to Present (to get rat complaints)
- DOHMH New York City Restaurant Inspection Results (most recent restaurant inspection results)
- Open Restaurants Inspections | NYC Open Data (cityofnewyork.us) (open restaurant inspection results)

Features:
- number of rat complaints 
- number of restaurants, num with alcohol
- health inspection score, critical flags of restaurants, restaurant closures (due to health inspection)
- Amount of sidewalk and roadway seating

Data cleaning:
- Removed irrelevant columns
- Filtered rows with missing or messy data
- Looked at 2022 only
- Joined open restaurant applications with 2022 health inspection, to get operating restaurants
- Convert categorical data to binary or one-hot encoding
- Group by zipcode - to get predictions on geographical zone basis

Dataset exploration:
- visualization options to plot and explore the data:
- Chromoplots to see the number of rats, number of restaurants, average health inspection grade by zip code
- Histograms showing health inspection grade
- Visualizations will be used to see if there are outliers 

# Methods and Model Training
ML techniques - regression

Models we investigated:
- multiple regression
- polynomial regression
- ridge regression
- lasso regression

Cross validation used to address over/underfitting, address the limited number of rows (only 100-something zip codes, so not many data points to train on)

# Model Evaluation
The model evaluation was conducted in `model_training/model_evaluation.ipynb`. There, we created a number of helper functions (inspired by the homework assignments) to train and evaluate different regression models. We evaluated the models based on MAE, RMSE, and R2. We also used cross validation because of the limited number of rows in our dataset and to avoid overfitting our data. We trained over an array of solvers and tolerances.

We found that polynomial regression tended to overfit. Linear, Lasso, and Ridge regression all had similar levels of over/underfitting and had similar error rates across the three error metrics we used. We thus chose to move forward with Lasso regression because it trains and predicts quickly and sets less-important feature coefficients to zero, thus creating a "leaner" model.

# Model Deployment
The code for final deployment can be found in `pages/B-Creat Your Own Neighborhood.py` and `pages/C-RatCzar Model.py`. In `pages/A-Our Machine Learning Model.py` you can select the type of model you want to analyze in `pages/B-Creat Your Own Neighborhood.py`.

`pages/C-RatCzar Model.py` shows the deployment of the Lasso model with only the features that will effect the final results.

# Front-end
In order to run the deployment you first need to install all the packages found in `requirements.txt` Then in your terminal use the command

`streamlit run rat-city.py`

A web browser should pop up with the information on `rat-city.py` and all the files in `pages`

# Results

Our analysis led us to select Lasso regression as the most effective model for predicting rat complaints within the city. Throughout our investigation, we discovered that population density had a more significant impact on the number of rat complaints than the number of restaurants in a given area. This insight allowed us to streamline our model inputs, facilitating a more user-friendly experience.

The final product of our project is a user-friendly Streamlit application that provides a practical tool for city officials and concerned citizens alike. The application enables users to enter specific parameters, such as the population of a specific zip code or the number of restaurants with alcohol licenses, and receive an estimation of the potential number of rat complaints that area could generate.

However, it's important to note that our model and its results may be influenced by the biases in our data source. The 311 data is heavily reliant on citizen reporting, which can introduce variations in the number of rat complaints. Some areas may have more proactive citizens leading to a higher number of complaints, while others may have less reported complaints due to lack of engagement, despite the potential severity of the rat problem.

Looking forward, we aim to integrate additional datasets into our model. A promising candidate for consideration could be municipal trash collection data, including the schedule and frequency of collections. This could potentially improve the accuracy of our rat complaint predictions by providing an additional perspective on factors that might influence rat populations.

In summary, our project has demonstrated the potential of using machine learning models to assist in urban pest control, although our results also highlight the importance of considering data biases and exploring a broader range of influential factors. By continually refining our model and integrating new data sources, we believe we can further improve our predictions and contribute to more effective rat control strategies within the city.
