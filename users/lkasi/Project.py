# 1. Import libraries
# Import analytics libraries
import pandas as pd

# Import visualizations packages
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

# Import Scikit-learn for modeling LR (logistic Regression)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 2. Import dataset (diabetes.csv)
data = pd.read_csv('diabetes.csv')

# 3. Exploratory Data Analysis (EDA)

# Check shape of dataset (df.shape)
print(data.shape)

# Preview of dataset (df.head())
print(data.head())

# Generally info of dataset (df.info())
print(data.info())

# Print basic info about dataset.
print("Column names and data types: ")
print(data.dtypes)

print("Summary Statistic: ")
print(data.describe())

# Checking for missing values in each column.
print("Missing values: ")
print(data.isnull().sum())

# Conclusion: There are no missing values.

# Histogram of all columns

plt.figure(figsize=(12, 6))
data.hist(bins=20, color='skyblue', edgecolor='black', grid=False, layout=(3, 3))
plt.tight_layout()


# Conclusion: There are many of "0" values.

# Function for counting zero values in each column.
def count_zero(df):
    zero_count = {}
    for column in df.columns:
        num_zero = (df[column] == 0).sum()
        zero_count[column] = num_zero
    return zero_count

zero_counts = count_zero(data)
print("""Number of "0" values in each columns.""")
for column, count in zero_counts.items():
    print(column, ":", count)

# There isn't missing values, but there is some of outlier values, especially in columns "SkinThickness",
# "Insulin", "Glucose", "BloodPressure" and "BMI".
# There are also a lot of "0" values "SkinThickness"= 227 and "Insulin" = 374. I'm leaving it this way,
# because maybe these valuse are ok.

# Removing outliers
def removing_outliers(df, columns):
    all_indices_to_keep = np.ones(len(df), dtype=bool)
    for column in columns:
        column_data = df[column]
        Q1 = np.percentile(column_data, 25)
        Q3 = np.percentile(column_data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        indices_to_keep = (column_data >= lower_bound) & (column_data <= upper_bound)
        all_indices_to_keep &= indices_to_keep
    return df[all_indices_to_keep]

cleaned_data = removing_outliers(data, columns=['Glucose', 'BloodPressure', 'BMI'])

# Counting 0 values after cleaning
zero_counts = count_zero(cleaned_data)
print("Liczba zer w każdej kolumnie")
for column, count in zero_counts.items():
    print(column, ":", count)

# Histogram of all CLEANED columns, after removing outliers
plt.figure(figsize=(12, 6))
cleaned_data.hist(bins=20, color='skyblue', edgecolor='black', grid=False, layout=(3, 3))
plt.tight_layout()

print("Shape of cleaning dataset: ")
print(cleaned_data.shape)

#Corellaion matrix
correlation_matrix = cleaned_data.corr()
print(correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation matrix')
plt.show()

# 4. Preprocessing the dataset
# Split the data into features and target variable

X = cleaned_data.drop(["Outcome"], axis=1)
Y = cleaned_data["Outcome"]

# Print some info about X and Y

X.head()
Y.head()

print(X.info())

# Standardize features (X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split teh data into training (70%) and test (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y,test_size=0.3, random_state=42)

print("Number of training samples: ", len(X_train))
print("Number of test samples: ", len(X_test))

# 5. Modeling LR (Logistic Regression)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

accuracy = model.score(X_test, Y_test)
print("Model accuracy: ", accuracy)

# Conclusion: Model accuracy is 78%. I think that's quite good.

preds = model.predict(X_test)
con_matrix = metrics.confusion_matrix(Y_test, preds)
print(con_matrix)
sns.heatmap(con_matrix, annot=True, fmt=".0f")

#Conclusion:
# Model accuracy is 78%, it's a quite good score.
# However, 28 samples were incorrectly classified as negative - for disease classification
# this value should be as low as possible. This is 10% of all sample tested.
# This result should be worked on.
