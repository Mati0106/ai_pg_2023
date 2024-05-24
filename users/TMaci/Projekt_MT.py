from math import nan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Optional for heatmap (comment out if not used)
import plotly.express as px  # Optional for scatter plot (comment out if not used)
from scipy.stats import ttest_ind


# Data import from excel fil
data = pd.read_excel('wc_analiza.xlsx')


# Deleting  ML equales 0
data = data[data['ML'] != 0]


# Replacing of wrong signes
data['Prices'] = data['Prices'].str.replace('$', '').str.replace(',', '')


# Conversion ML, Prices i Ratings on float type
data['ML'] = pd.to_numeric(data['ML'], errors='coerce')
data['Prices'] = pd.to_numeric(data['Prices'], errors='coerce')
data['Ratings'] = pd.to_numeric(data['Ratings'], errors='coerce')
data['Ratingsnum'] = pd.to_numeric(data['Ratingsnum'], errors='coerce')


# recalculation frices for 750ml
data['Prices'] = (data['Prices'] / data['ML']) * 750


# droping rows weith Ratingsnum = 0
data = data[data['Ratingsnum'] != 0]


# Adding total rating value
data['Total Rating Value'] = data['Ratings'] * data['Ratingsnum']


# max price
max_price = data['Prices'].max() - 6900


# addjusting number of bins
num_bins = int(max_price // 90) + 1


# Bins creation
bins = np.linspace(0, max_price, num_bins + 1)


# Labels
labels = [f'{x:.2f}' for x in bins[:-1]]


# Adding column Price range DataFrame
data['Price Range'] = pd.cut(data['Prices'], bins=bins, labels=labels)

data = data.dropna()
# calculation of average rating values
data['Average Rating Value'] = data['Total Rating Value'] / data['Ratingsnum']
print(data.head())

data.hist(bins=80,figsize=(13, 9))
plt.tight_layout()
plt.show()
#

missing_values = data.isnull().sum()
print(missing_values)

# selecting numeric columns
numeric_columns = data.select_dtypes(include=['number'])

# calculating the correlation matrix
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)

# checking how much rows there are in DataFrame
rows_number = len(data)
print("Number of rows in xlsx file:", rows_number)

# checking the heat map to see visualization of correlation
plt.figure(figsize=(7, 4))  # Chart size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation matrix')
plt.show()

# Grupping
grouped_data = data.groupby('Price Range').agg({'Total Rating Value': 'sum', 'Ratingsnum': 'sum', 'Average Rating Value': 'mean'}).reset_index()

# Bar
plt.figure(figsize=(15, 6))
plt.bar(grouped_data['Price Range'], grouped_data['Average Rating Value'], width=0.8)  # Use 'Average


plt.xlabel('Przedział cenowy dla uśrednionej pojemności 750ml')
plt.ylabel('Średnia wartość ocen')
plt.title('Zależność ceny produktu od średniej wartości ocen')
plt.xticks(rotation=45)
plt.tight_layout()

data.describe()

data.info()

sns.heatmap(grouped_data.corr(), annot=True)
plt.show()
grouped_data.corr()
data.describe()

import seaborn as sns
import matplotlib.pyplot as plt


maxval2 = data['Prices'].max() # get the maximum value
data_upd = data[data['Prices'] != maxval2]

data_upd.hist(bins=80, figsize=(13,9));plt.show() # outlier is completely removed

# selection of numeric columns
numeric_columns = data_upd.select_dtypes(include=['number'])

# Calculation the correlation matrix
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)

# Heat map to see visualization of correlation
plt.figure(figsize=(7, 4))  # Chart size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation matrix')
plt.show()
#
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Selection of columns with highly correlated data
skorelowane_kolumny = ['Total Rating Value', 'Ratingsnum', 'Average Rating Value']


# Data standarization
scaler = StandardScaler()
X_skorelowane = scaler.fit_transform(data[skorelowane_kolumny])


#---------------------------------------------

# PCA


pca = PCA(n_components=2)  # Selecting the number of principal components (1,2,3)
X_pca = pca.fit_transform(X_skorelowane)

# Display of results

print(pca.components_)
print(pca.explained_variance_ratio_)

#it will be continued

# Adding first principal component to data
data_upd_pca2 = pd.concat([data_upd.drop(columns=skorelowane_kolumny), pd.DataFrame(X_pca, columns=['PCA_1','PCA_2'])], axis=1)



# Deleting rows containing empty values
data_upd_pca_cleaned = data_upd_pca2.dropna()

# Splitting the data into a training set and a test set

from sklearn.model_selection import train_test_split

# Divide data into features (X) and labels (y)
X = data_upd_pca_cleaned.drop('Prices', axis=1)
y = data_upd_pca_cleaned['Prices']

# Data transformation
X_encoded = pd.get_dummies(X)

print("Before one-hot encoding:")
print(X.head())

print("\nAfter one-hot encoding:")
print(X_encoded.head())

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

# Fit the model to your training data with feature names
model = LinearRegression()
model.fit(X_train_encoded, y_train)


# Fit the model to training data
model.fit(X_test_encoded, y_test)

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

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model = LinearRegression()
def objective(trial):
fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)


scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
mse = -scores.mean()
return mse