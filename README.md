
# house-prices
House Prices (Python) - work in progress

In this project, I'll develop prediction models using the house prices dataset from Kaggle. The goal is to demonstrate the 4 steps of the Data Science project lifecycle: Define, Discover, Develop and Deploy.  I will build several supervised machine learning models and evaulate the performance.

### 1. Define:

- Define the Problem:

The objective of this ML exercise is to predict house prices using the Ames, Iowa housing dataset.


### 2. Discover: 

- Obtain data:

The raw dataset comes in train.csv, test.csv and sample_submission.csv including the target values for the test test. Since, the purpose of this exercise is not a competation, I merged the csv files into one file for easier pre-processing steps.
 
- Clean data:

There are missing values in this dataset. Four of these columns 'Alley', 'PoolQC', 'Fence', 'MiscFeature' have more than 80% of the values missing, thus we can consider dropping them from the dataset in the modelling phase.

- Explore data:

To explore the data, we can create visualizations. First, let's take a look at the target. From the Figure below it is evident that SalePrice variable is skewed to the right and there are several outliers.

![saleprice](https://user-images.githubusercontent.com/26305084/110682726-43177900-81a9-11eb-9de3-0047b689790d.jpeg)


Next, we can take a look at features. There are ordinal, nominal and numeric (continious or integer) variables. To explore the relationship between SalePrice and numeric feautures, I'll use scatter plots, and between SalePrice and categoric features, I'll use scatter plots.  

The box-plots below reveal the spread of the SalePrices across the category levels. We see an inceasing trend as the Overall Qual and Overall Cond of the house increase.

![boxplots](https://user-images.githubusercontent.com/26305084/110686259-40b71e00-81ad-11eb-8f2f-e31554a4a769.jpeg)

![scatter](https://user-images.githubusercontent.com/26305084/110687238-49f4ba80-81ae-11eb-828c-6d2fc1ac9b98.jpeg)

![heatmap](https://user-images.githubusercontent.com/26305084/110687048-1580fe80-81ae-11eb-9ca9-839ecb3bdd2e.jpeg)


### 3. Develop:



### 4. Deploy:
