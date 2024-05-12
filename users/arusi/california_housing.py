#The purpose of the exercises performed is to predict median house value based on the data from file housing.csv

import pandas as pd

# path to CSV file with data
path_to_file = 'C:\\Users\\rysru\\Desktop\\housing.csv'

# Upload of data from CSV file to DataFrame
data = pd.read_csv(path_to_file)

print(data.head())

# checking if file has any missing values
missing_values = data.isnull().sum()
print(missing_values)

# checking how much rows there are in DataFrame
rows_number = len(data)
print("Number of rows in CSV file:", rows_number)

# As the missing values in the total_bedrooms column represent only 1% of the total values, I fill them with an average value

# I calculate the average for the ‘total_bedrooms’ column
srednia = data['total_bedrooms'].mean()

# I am replacing missing values in the ‘total_bedrooms’ column with the average
data['total_bedrooms'].fillna(srednia, inplace=True)

# I am checking if there are any missing values in a column after replacing them with the average
if data['total_bedrooms'].isnull().sum() == 0:
    print("Blank values have been replaced by the average in the 'total_bedrooms'.")
else:
    print("It was not possible to replace empty values with the average in the column 'total_bedrooms'.")

import matplotlib.pyplot as plt

# I am creating histograms for the whole data
data.hist(bins=80,figsize=(13, 9))  # Chart size: 10x8
plt.tight_layout()
plt.show()


# Conclusions about the data:
# On first impression, a few outlier groups are present in our data( 'housing_median_age' & 'median_house_value' )
# Housing_median_age has a lot of local peaks but one really odd peak at the maximum value stands out. It has some slight discontinuity in data.
# Feature Median_house_value has an odd peak at its maximum value (around 500k), which seems to be an outlier.
# Population, total_bedrooms and total_rooms represent somewhat connected things, also have similar distribution which is skewed towards smaller values.

# I am removing the outlier for median_house_value by selecting the maximum
# dane_upd : training data without removed outliers
maxval2 = data['median_house_value'].max() # get the maximum value
data_upd = data[data['median_house_value'] != maxval2]

data_upd.hist(bins=80, figsize=(13,9));plt.show() # outlier is completely removed

# To understand more about dataset I am using correlation matrix

# I am selecting only numeric columns
numeric_columns = data_upd.select_dtypes(include=['number'])

# I am calculating the correlation matrix
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt


# I am creating heat map to see visualization of correlation
plt.figure(figsize=(7, 4))  # Chart size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation matrix')
plt.show()

# Feature engineering

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Selection of columns with highly correlated data
skorelowane_kolumny = ['total_bedrooms', 'population', 'households', 'total_rooms']

# Data standarization
scaler = StandardScaler()
X_skorelowane = scaler.fit_transform(data_upd[skorelowane_kolumny])

# PCA
pca = PCA(n_components=2)  # Selecting the number of principal components
X_pca = pca.fit_transform(X_skorelowane)

# Display of results
print("Principal Component Loadings:")
print(pca.components_)
print("\nVariance explained by each principal component:")
print(pca.explained_variance_ratio_)

# Conclusions from PCA:
# The first principal component is primarily positively correlated with each variable, whereas the second principal component has both positive and negative correlations with different variables.
# The first principal component explains 93.59% of the variance in the data, while the second principal component explains only 3.83% of the variance. This means that the first principal component retains significantly more information than the second one.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Selection of columns with highly correlated data
skorelowane_kolumny = ['total_bedrooms', 'population', 'households', 'total_rooms']

# Data standarization
scaler = StandardScaler()
X_skorelowane = scaler.fit_transform(data_upd[skorelowane_kolumny])

# PCA
pca = PCA(n_components=1)  # I retain only the first principal component
X_pca = pca.fit_transform(X_skorelowane)


# Wyświetlenie wyników
print("Principal Component Loadings:")
print(pca.components_)
print("\nVariance explained by first principal component:")
print(pca.explained_variance_ratio_)

# Adding first principal component to data
data_upd_pca = pd.concat([data_upd.drop(columns=skorelowane_kolumny), pd.DataFrame(X_pca, columns=['PCA_1'])], axis=1)

# Checking if there are any missing values in any column
missing_values = data_upd_pca.isnull().sum()
columns_with_missing_values = missing_values[missing_values > 0]

if columns_with_missing_values.empty:
    print("No missing values in any column.")
else:
    print("There are missing values in the following columns:")
    print(columns_with_missing_values)

# Deleting rows containing empty values
data_upd_pca_cleaned = data_upd_pca.dropna()

# Wyświetl informacje o DataFrame po usunięciu pustych wartości
print("Number of rows before empty values are removed:", len(data_upd_pca))
print("Number of rows after empty values are removed:", len(data_upd_pca_cleaned))

# Splitting the data into a training set and a test set

from sklearn.model_selection import train_test_split

# Divide data into features (X) and labels (y)
X = data_upd_pca_cleaned.drop("median_house_value", axis=1)
y = data_upd_pca_cleaned["median_house_value"]

# Divide data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Result:
# X_train - trening features
# X_test - test features
# y_train - trening labels
# y_test - test labels

# Transforming categorical variables using one-hot coding
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialise linear regression model
model = LinearRegression()

# Fit the model to your training data
model.fit(X_train_encoded, y_train)

# Perform prediction on test data
y_pred = model.predict(X_test_encoded)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape)

mspe = np.mean(((y_test - y_pred) / y_test) ** 2)
print("Mean Squared Percentage Error (MSPE):", mspe)

max_err = max_error(y_test, y_pred)
print("Max Error:", max_err)

# Conclusions: Based on the metric results, it seems that the linear regression model might not be appropriate.
# Despite the relatively high R-squared, other metrics such as MSE, MAE, MAPE, MSPE, and max error indicate that the model may not be sufficiently accurate.
# I would like to use Optuna for hyperparameter optimization to potentially improve the performance of my linear regression model

import optuna
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score




