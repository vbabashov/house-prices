# house-prices
House Prices (Python) - work in progress

In this project, I'll develop prediction models using the house prices dataset from Kaggle. The goal is to demonstrate the 4 steps of the Data Science project lifecycle: Define, Discover, Develop and Deploy.  I will build several supervised machine learning models and evaulate the performance.

### 1. Define: 

The objective of this ML exercise is to predict house prices using the Ames, Iowa housing dataset.


### 2. Discover: 

The raw dataset comes in train.csv, test.csv and sample_submission.csv including the target values for the test test. Since, the purpose of this exercise is not a competation, I merged the csv files into one file for easier pre-processing steps.

First, I started with the Exploratory Data Analysis (EDA). 

- Missing Values: There are missing values in this dataset. Four of these columns 'Alley', 'PoolQC', 'Fence', 'MiscFeature' have more than 80% of the values missing, thus we can consider dropping them from the dataset in the modelling phase.

From the Figure below it is evident that SalePrice variable is skewed to the right and there are several outliers.

![saleprice](https://user-images.githubusercontent.com/26305084/110682726-43177900-81a9-11eb-9de3-0047b689790d.jpeg)


### 3. Develop:



### 4. Deploy:
