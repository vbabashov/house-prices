
# house-prices
House Price Prediction (Python) - work in progress

### Problem Statement:

In this project, I'll develop prediction models using the house prices dataset from Aimes, IA. The goal is to demonstrate the 4 steps of the Data Science project lifecycle: Define, Discover, Develop and Deploy. First, I'll establish simple baseline model using the OLS regression, and then I'll develop a few predictive models, namely, random forest, xgboost and lightgbm regression models and compare the performance of these models against the baseline with the aim to get better predictive performance. The implementation of similar price prediction models will potentially allow housing agencies (e.g., CMHC) real-estate companies, central and commercial banks, municipial governments and home buyers to make informed decisions with respect to market pricing.


### 1. Define:

- Define the Problem: The objective of this supervised ML regression exercise is to determine the best model to predict the house prices.

### 2. Discover: 

- Obtain data: The raw dataset comes in train.csv, test.csv and sample_submission.csv including the target values for the test test. Since, the purpose of this exercise is not a competition, I merged the csv files into one file for easier pre-processing steps. The raw dataset comes in train.csv featuring 81 columns and 1460 raws. The test.csv includes the features where the Sale Price is to be predicted.

- Clean data: Four of the columns 'Alley', 'PoolQC', 'Fence', 'MiscFeature' have more than 80% of the values missing, thus, I'll drop them from the dataset in the modelling phase.  The rest of the columns were imputed using median and mode imputers.

- Explore data: To explore the data, we can create visualizations. First, let's take a look at the target. From the Figure below SalePrice variable is skewed to the right and there are several outliers. Log-transformation of the target can help improve the model performance.

![saleprice](https://user-images.githubusercontent.com/26305084/112194684-8200f200-8bdf-11eb-9db5-dec7dc242f72.jpeg)

Next, we can take a look at features. There are ordinal, nominal and numeric (continuous or integer) variables. To explore the relationship between SalePrice and numeric feautures, I'll use scatter plots, and between SalePrice and categoric features, I'll use scatter plots.  

- The box-plots below reveal the spread of the SalePrices across the category levels. Mean sale price and distribuition of prices are similar for most of the variables. We see an inceasing trend as the OverallQual and OverallCond of the house increase. For some levels of the neihbourhood, while some categories explainthe categorical levels don't seem to explain the difference in price.

![selectboxplots](https://user-images.githubusercontent.com/26305084/112189231-31d36100-8bda-11eb-846d-a79159a4a24c.jpeg)


- Scatter plots show that SalePrice increases with LotFrontage, LotArea, BsmtFinSF1, TotalBsmtSF, 1sfFlrSF, GrLivArea,GarageArea, GarageCars and Fireplaces variables.

![selectscatterplots](https://user-images.githubusercontent.com/26305084/112189268-3ac43280-8bda-11eb-9081-e083d1216b17.jpeg)

- Finally, the heatmap show correlation between the features and between features and SalePrice.According to the heatmap, house SalePrice is correlated with GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF (Pearson Correlation Cofficient >=0.6). There seems to be a weaker correlation between SalePrice and rest of the features (Pearson Correlation Cofficient <= 0.5).

![heatmap](https://user-images.githubusercontent.com/26305084/111834061-71900500-88c9-11eb-88ed-b6dd1bdcd737.jpeg)

- Set baseline outcomes: As a baseline, shown in a [notebook](https://github.com/vbabashov/house-prices/blob/main/baseline.ipynb), I built Ordinary Least Squares (OLS), and obtained the Mean Absolute Error (MAE) of with the test dataset. MAE  for the Baseline Model: 24139.18

- Hypothesize solutions: It is my contention that we can obtain better predictive performance compared to baseline using the tree-based models along with feature engineering as following: Random Forest, Xgboost and LightGBM. These (bagging and boosting) models have shown to be successful in different applications. Therefore, I choose them as possible candidate models to explore. The details of Discover stage, can be found in this [notebook](https://github.com/vbabashov/house-prices/blob/main/EDA.ipynb).

### 3. Develop:

- Engineer features: Feature engineering is critical to succesful ML applications. Here, I use feature_engine Python library and sklearn's prepocessing and feature selection. There are ordinal variables in the dataset. I used ordinal encoding to encode the variables. Nominal variables have a lot of categories. Some of these categories don't have any observation. In the preliminary analysis, I noted that one-hot encoding results in poor model peformance due to many columns with sparsity and some categories being having rare values. Instead, I looked at mean Sale Price for each category and encoded them in an increasing order. This showed better model predictive performance. I mapped the month names from numbers to string names to better reflect the nominal nature. I also encoded the categories with rare values and combined them into single category called Rare. This helps to alleviate the rareness problem. And, I encoded the categories with ordering as per increasing mean prices.Also, I engineered four age-related features, total bath count, and total area of the house features. Finally, I log-transformed the SalePrice to minimize the impact of the outliers.
             
- Create Models: I created models using the Pipelines to chain Polynomial Features, Feature Selection wih the models. In addition, I created the parameter grid to be used in GridSearch hyperparameter tuning.

- Test models: I used nested cross-validation approach (5x2Cv) to compare and find the best performing algorithm. Analysis showed that LightGBM performs relatively well.

Train MAE: 10776.56

Test MAE: 16447.13

Train R2: 0.95

Test R2: 0.86

We can see overfitting due to diffences in train and test model predictive performance.

- Select best models: We then performed the GridSearch hyper-parameter tuning on the LightGBM to determine the best set of parameters.

Best CV Score (best RMSE on Log scale): 0.09

Best Parameters: {'reg3__colsample_bytree': 0.3, 
                          'reg3__max_depth': 6, 
                      'reg3__n_estimators': 100, 
                         'reg3__num_leaves': 8}
                         

### 4. Deploy:

- Automate pipeline: At this point, I fit the best model on the entire dataset and generate the predictions on a new dataset.

- Deploy solution: Finally, I save the predictions in a csv file, and save the model, predictions and feature importances to the disk. Below is the figure showing top 25 important features.

![test](https://user-images.githubusercontent.com/26305084/111883088-7b7c3b80-898f-11eb-821a-3772c9aa5a85.jpeg)

As we can see, LotArea, TotalArea, GrLivArea, OverallCond and OverallQual are the top 5 features with the most predictive power.

- Measure efficacy: I'm going to skip this step, since I don't have the actual outcomes of the unseen test data.

The details as well as full implementation of the Develop and Deploy stages can be found in a separate [notebook](https://github.com/vbabashov/house-prices/blob/main/price_prediction.ipynb).    

### Concluding Remarks

The model performance can be further improved by more feature engineering such as reducing the number of categories into fewer categories and/or dropping some unimportant features from the model. This is particularly important, because unimportant and skewed features introduce more noise to the model. Adding polynomial features and interactions didn't imporove the performance. More experiments with the SelectKBest approach is necessary. Alternative feature selection methods can be explored.

### Future Work

