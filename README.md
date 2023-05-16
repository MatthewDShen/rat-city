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
Used MAE, RMSE, R2 score to evaluate
Specified solver, CV folds, training size hyperparameters
Compared training vs validataion errors

# Model Deployment

# Front-end

# Results

