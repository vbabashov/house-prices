
# house-prices
House Prices (Python) - work in progress

In this project, I'll develop prediction models using the house prices dataset from Kaggle. The goal is to demonstrate the 4 steps of the Data Science project lifecycle: Define, Discover, Develop and Deploy.  I will build several supervised machine learning models and evaulate the performance.

### 1. Define:

- Define the Problem: The objective of this ML exercise is to determine the best model to predict house prices using in the Ames, Iowa. 

### 2. Discover: 

- Obtain data: 
The raw dataset comes in train.csv featuring 81 columns and 1460 raws. The test.csv includes the features where the Sale Price is to be predicted.
 
- Clean data: 
Four of the columns 'Alley', 'PoolQC', 'Fence', 'MiscFeature' have more than 80% of the values missing, thus, I'll drop them from the dataset in the modelling phase.  The rest of the columns were imputed using median and mode imputers.

- Explore data: 
To explore the data, we can create visualizations. First, let's take a look at the target. From the Figure below SalePrice variable is skewed to the right and there are several outliers. Log-transformation of the target can help improve the model performance.

![saleprice](https://user-images.githubusercontent.com/26305084/110682726-43177900-81a9-11eb-9de3-0047b689790d.jpeg)

Next, we can take a look at features. There are ordinal, nominal and numeric (continious or integer) variables. To explore the relationship between SalePrice and numeric feautures, I'll use scatter plots, and between SalePrice and categoric features, I'll use scatter plots.  

The box-plots below reveal the spread of the SalePrices across the category levels. Mean sale price and distribuition of prices are similar for most of the variables. We see an inceasing trend as the OverallQual and OverallCond of the house increase. For some variables, while some categories are rare, most of the categorical levels don't seem to explain the difference in price.

![boxplots](https://user-images.githubusercontent.com/26305084/111832968-d9455080-88c7-11eb-9016-cd800720cce4.jpeg)

Scatter plots show that SalePrie increases with LotFrontage, LotArea, BsmtFinSF1, BsmtFinSF2, TotalBsmtSF, 1sfFlrSF, 2ndFlrSF, GrLivArea and GarageArea variables.

![scatter](https://user-images.githubusercontent.com/26305084/111832940-cdf22500-88c7-11eb-8522-546c7244c420.jpeg)

Finally, the heatmap show correlation between the features and between features and SalePrice.According to the heatmap, house SalePrice is correlated with GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF (Pearson Correlation Cofficient >=0.6). There seems to be a weaker correlation between SalePrice and rest of the features (Pearson Correlation Cofficient <= 0.5).

![heatmap](https://user-images.githubusercontent.com/26305084/111834061-71900500-88c9-11eb-88ed-b6dd1bdcd737.jpeg)

- Set baseline outcomes: 
As a [baseline](https://github.com/vbabashov/house-prices/blob/main/baseline.ipynb), I built Ordinary Least Squares (OLS), and obtained the Mean Absolute Error (MAE) of with the test dataset. MAE  for the Baseline Model: 24139.18

- Hypothesize solutions: 
It is my contention that we can obtain better predictive performance compared to baseline using the tree-based models along with feature engineering as following:

- Random Forest
- Xgboost
- LightGBM

These (bagging and boosting) models have shown to be successful in different applications. Therefore, I choose them as possible candidate models to explore.

### 3. Develop:

- Engineer features: 
Feature engineering is critical to succesful ML applications. Here, I use feature_engine Python library and sklearn's prepocessing and feature selection. There are ordinal variables in the dataset. I used ordinal encoding to encode the variables. Nominal variables have a lot of categories. Some of these categories don't have any observation. In the preliminary analysis, I noted that one-hot encoding results in poor model peformance due to many columns with sparsity and some categories being having rare values. Instead, I looked at mean Sale Price for each category and encoded them in an increasing order. This showed better model predictive performance.
 
I mapped the month names from numbers to string names to better reflect the nominal nature. I also encoded the categories with rare values and combined them into single category called Rare. This helps to alleviate the rareness problem. And, I encoded the categories with ordering as per increasing mean prices.

I engieered four age-related features, total bath count, and total area of the house features.
             
- Create Models
- Test models
- Select best models

### 4. Deploy:
- Automate pipeline
- Deploy solution
- Measure efficacy
