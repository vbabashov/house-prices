
<p align="center">
   <img src="https://user-images.githubusercontent.com/26305084/117038955-35c4c980-acd6-11eb-9a5e-4e98d4d4b764.gif" />
</p>


# house-prices
House Price Prediction (Python)

- **Notebook 1: [Exploratory Data Analysis (EDA)](https://github.com/vbabashov/house-prices/blob/main/notebooks/EDA.ipynb)**
- **Notebook 2: [Baseline Model](https://github.com/vbabashov/house-prices/blob/main/notebooks/baseline.ipynb)**
- **Notebook 3: [ML Model Training and Prediction](https://github.com/vbabashov/house-prices/blob/main/notebooks/price_prediction.ipynb)**
- **Object Oriented Programming: [.py file](https://github.com/vbabashov/house-prices/blob/main/src/model.py)**

### Problem Statement:

In this project, I'll develop prediction models using the house prices dataset from Aimes, IA. The goal is to demonstrate the 4 steps of the Data Science project lifecycle: Define, Discover, Develop and Deploy. First, I'll establish simple baseline model using the OLS regression, and then I'll develop a few predictive models, namely, random forest, xgboost and lightgbm regression models and compare the performance of these models against the baseline with the aim to get better predictive performance. The implementation of similar price prediction models will potentially allow the housing agencies (e.g., CMHC in Canada), real-estate companies, central and commercial banks, municipial governments and home buyers to make informed decisions with respect to market pricing.


### 1. Define:
***

- Define the Problem: The objective of this supervised ML regression exercise is to determine the best model to predict the house prices in Aimes, IA.

### 2. Discover: 
***

- Obtain data: The raw dataset comes in train.csv featuring 81 columns and 1460 raws. The test.csv includes the features where the SalePrice is to be predicted. I'll use the test.csv data as an unseen data and generate predictions.

- Clean data: Four of the columns 'Alley', 'PoolQC', 'Fence', 'MiscFeature' have more than 80% of the values missing, thus, I'll drop them from the dataset in the modelling phase. The rest of the columns were imputed using median and mode imputers available in the sklearn package.

- Explore data: To explore the data, we can create data visualizations. First, let's take a look at the target. From the Figure below SalePrice variable is skewed to the right and there are several outliers. Log-transformation of the target can help improve the model performance. For now, we'll keep the outliers, but it is important to note that outliers in features or target can make the ML model unstable.

     ![saleprice](https://user-images.githubusercontent.com/26305084/112194684-8200f200-8bdf-11eb-9db5-dec7dc242f72.jpeg)

Next, we can take a look at features. There are ordinal, nominal and numeric (continuous or integer) variables. To explore the relationship between SalePrice and numeric feautures, I'll use scatter plots, and to investigate the relationship between SalePrice and categoric features, I'll use scatter plots.  

- The box-plots below reveal the spread of the SalePrices across the category levels. Mean sale price and distribuition of prices are similar for most of the variables. Therefore, I included only a select number of box-plots here. We can see an inceasing trend as the OverallQual and OverallCond of the house increase. While some categories explain the difference in prices for some levels of the Neighbourhood, Condition1, Functional, LotConfig, BsmtExposure and KitchenQual, other levels don't seem to explain the variability in price.

   
     ![selectboxplots](https://user-images.githubusercontent.com/26305084/112189231-31d36100-8bda-11eb-846d-a79159a4a24c.jpeg)


- Scatter plots show that SalePrice increases with LotFrontage, LotArea, BsmtFinSF1, TotalBsmtSF, 1sfFlrSF, GrLivArea, GarageArea, GarageCars and Fireplaces variables.

    
     ![selectscatterplots](https://user-images.githubusercontent.com/26305084/112324815-8c77c600-8c89-11eb-97db-a3d0931f9e0c.jpeg)


- Finally, the heatmap shows the Pearson correlation coefficients between and among the features and SalePrice. According to the heatmap, house SalePrice is correlated with GrLivArea, GarageCars, GarageArea, TotalBsmtSF and 1stFlrSF (all with Pearson Correlation Cofficient >=0.6).

     ![selectheatmap](https://user-images.githubusercontent.com/26305084/112482761-c52da300-8d4e-11eb-8659-fa7e48f39cc7.jpeg)

- Set baseline outcomes: As a baseline, as shown in a [notebook](https://github.com/vbabashov/house-prices/blob/main/notebooks/baseline.ipynb), I built an Ordinary Least Squares (OLS), and obtained the Mean Absolute Error (MAE) of **24139.18** with the test dataset.

- Hypothesize solutions: It is my contention that we can obtain better predictive performance compared to baseline model using the tree-based models with some feature engineering. Random Forest, Xgboost and LightGBM. These (bagging and boosting) models have shown to be successful in different applications. Therefore, I choose them as possible candidate models to explore. 

The details of Discover stage (e.g., EDA) can be found in this [notebook](https://github.com/vbabashov/house-prices/blob/main/notebooks/EDA.ipynb).

### 3. Develop:
***

- Engineer features: Feature engineering is critical to succesful ML applications. Here, I use feature_engine Python library and sklearn's prepocessing and feature selection functions. There are ordinal variables in the dataset. I used ordinal encoding to encode the variables. Nominal variables have a lot of categories. Some of these categories don't have any observation (i.e, rare). In the preliminary analysis, I noted that one-hot encoding results in poor model peformance due to many columns with sparsity and some categories being having rare values. Instead, I looked at mean Sale Price for each category and encoded them in an increasing order. This showed better model predictive performance. I mapped the month names from numbers to string names to better reflect the nominal nature this variable. I also encoded the categories with rare values and combined them into single category called Rare. This helps to alleviate the rareness problem. And, I encoded the categories in order according to increasing mean prices. Also, I engineered four age-related features, total bath count, and total area of the house features. Finally, I log-transformed the SalePrice to minimize the impact of the outliers.
             
- Create Models: I created models using the Pipelines to chain Polynomial Features, Feature Selection wih the models. In addition, I created the parameter grid to be used in GridSearch hyperparameter tuning.

- Test models: I used nested cross-validation approach (5x2Cv) to perform algorithm selection and model selection (i.e., hyperparameter tuning). Based on the GridSearch hyper-parameter tuning and nested cross-validation analysis, LightGBM performs relatively well compared to other alternative algortihms with different hypothesis search space parameters.

      Train MAE: 11789.50
      Test MAE: 16645.64

      Train R2: 0.94
      Test R2: 0.85

We can see some degree of overfitting due to diffence in training and testing model predictive performance.

- Select best models: We can then perform additional hyperparameter tuning on the LightGBM to determine the best set of parameters. This step is optional, as we have already tuned the models inside the cross-validation function using the GridSearchCV. However, one can use this step to do more substantive and wide-randing hyperparameter search. Also, instead of GridSearchCV, one can use RandomSearchCV or other parameter optimization techniques.

      Best CV Score (best RMSE on Log scale): 0.13

      Best Parameters: {'reg3__colsample_bytree': 0.3, 
                                'reg3__max_depth': 4, 
                            'reg3__n_estimators': 100, 
                               'reg3__num_leaves': 6}
                         

### 4. Deploy:
***

- Automate pipeline: At this point, I fit the best model on the entire dataset and generate the predictions on a unseen feature dataset (e.g., test.csv).

- Deploy solution: Finally, I store the predictions in a csv file, and then save the model, predictions and feature importances to the disk. Below is the figure showing top 25 important features. As we can see, GrLivArea, LotArea, TotalArea, OverallCond and OverallQual are the top 5 features with the most predictive power.

     ![features](https://user-images.githubusercontent.com/26305084/113339413-6a6cec00-92f8-11eb-9aee-a7fabbd325fb.jpeg))

- Measure efficacy: I'm going to skip this step, since we don't have the actual outcomes of the unseen test data.

The details as well as full implementation of the Develop and Deploy stages can be found in a separate [notebook](https://github.com/vbabashov/house-prices/blob/main/notebooks/price_prediction.ipynb).    

### Concluding Remarks:

The LightGBM performed slightly better than the Xgboost and Random Forest. This confirms our early contention that tree-based boosting algorithms can be usefull in building interpretable model with better predictive performance. LightGBM is high performance boosting algorithm. If tuned well, it can result in good and useful models. Despite some degree of overfitting, the model (as-is) is still useful and resulted in **32%** improvement in model predictive performance compared to baseline OLS model. However, there is a room for further improvement which I highlight in the next section.

### Future Enhancement:

As a future enhacement, we can try the following steps which will potentially reduce the degree of overfitting and improve the model performance.

- Feature engineering: There are some cateogrical and continious variables (e.g., SaleType, PoolArea) which seem to have a weak predictive power. Combining them into fewer categories and/or dropping the unimportant features from the model can help lift the model. Unimportant and skewed features introduce more noise to the model. 

- LightGBM Tuner:  We can also try [Optuna](https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258) to iteratively optimize the model accuracy across large set of hyperparameter tuning. Particularly,tuning l1 and l2 regularization parameters can help deal with the overfitting problem.
