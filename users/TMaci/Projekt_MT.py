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
skorelowane_kolumny = ['Total Rating Value', 'Ratingsnum', 'Average Rating Value', 'Prices']


# Data standarization
scaler = StandardScaler()
X_skorelowane = scaler.fit_transform(data[skorelowane_kolumny])


#---------------------------------------------

# PCA


pca = PCA(n_components=1)  # Selecting the number of principal components
X_pca = pca.fit_transform(X_skorelowane)

# Display of results

print(pca.components_)
print(pca.explained_variance_ratio_)

#it will be continued
