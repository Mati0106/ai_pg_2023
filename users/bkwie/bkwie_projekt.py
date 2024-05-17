# importing libraries
import pandas as pd
import numpy as np
import scipy
from scipy.stats import zscore
import matplotlib.pyplot as plt

#EDA part
# read data
# dynamic: 2002-2023; 2024: 2024 year only
df_dynamic= pd.read_csv("users/bkwie/World-happiness-report-updated_2024.csv", encoding='latin-1')
df_2024= pd.read_csv("users/bkwie/World-happiness-report-2024.csv", encoding='latin-1')
print(df_dynamic.head(5))
print(df_2024.head(5))

# Display the column names and data types
print("Column Names and Data Types:")
print(df_dynamic.dtypes)
print(df_2024.dtypes)

# Display summary statistics
print("\nSummary Statistics:")
print(df_dynamic.describe())
print(df_2024.describe())

#display info
df_dynamic.info()
df_2024.info()

# Handling Missing Values: Fill non-numeric columns in the dynamic dataset
numeric_cols = df_dynamic.select_dtypes(include=np.number).columns
df_dynamic[numeric_cols] = df_dynamic[numeric_cols].fillna(df_dynamic[numeric_cols].mean())

# Handling Missing Values: Fill non-numeric columns in the 2024 dataset
numeric_cols = df_2024.select_dtypes(include=np.number).columns
df_2024[numeric_cols] = df_2024[numeric_cols].fillna(df_2024[numeric_cols].mean())

#stats
agg_df = df_2024.pivot_table(
    values=['Country name'],
    index=['Regional indicator'],
    aggfunc=['count']
).reset_index()

print(agg_df)

# Display the shape of the DataFrame
print("\nDataFrame Dynamic Shape:", df_dynamic.shape)
print("\nDataFrame 2024 Shape:", df_2024.shape)

# Identify outliers using z-score
# from scipy.stats import zscore
# Remove outliers based on z-score threshold - 2024 data
# We use z-score to identify and remove outliers from numeric data in the DataFrame.
threshold = 3
help(zscore)
z_score = zscore(df_dynamic['Healthy life expectancy at birth'])
df_dynamic['Z_Score_Healthy life expectancy at birth'] = z_score
df_dynamic_filtered = df_dynamic[df_dynamic['Z_Score_Healthy life expectancy at birth'].abs() < threshold]
# df_2024_filtered = df_2024[df_2024['Z_Score'].abs() < threshold]


