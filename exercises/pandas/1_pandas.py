# Example 1: Loading a CSV file into a DataFrame

import pandas as pd

# Load a CSV file into a DataFrame
df = pd.read_csv('datasets/cars.csv')

# Display the first few rows of the DataFrame
print(df.head())
# In this example, we use the read_csv function from Pandas to load a CSV file named data.csv into a DataFrame.

# Example 2: Exploring DataFrame Information

# Display the column names and data types
print("Column Names and Data Types:")
print(df.dtypes)

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Display the shape of the DataFrame
print("\nDataFrame Shape:", df.shape)
# Here, we explore information about the DataFrame, such as column names, data types, summary statistics, and the shape of the DataFrame.

# Example 3: Selecting and Filtering Data

# Select a single column
column_data = df['Column_Name']

# Select multiple columns
multiple_columns = df[['Column1', 'Column2']]

# Filter rows based on a condition
filtered_data = df[df['Column_Name'] > 10]
# In this example, we demonstrate how to select specific columns and filter rows based on conditions in the DataFrame.

# Example 4: Grouping and Aggregating Data

# Group data by a column and calculate mean
mean_by_group = df.groupby('Column_Name').mean()

# Perform multiple aggregations
agg_results = df.groupby('Column_Name').agg({'Column1': 'mean', 'Column2': 'sum'})
# Here, we group the data by a column and perform aggregations such as mean calculation and summing across multiple columns.

# Example 5: Handling Missing Values

# Drop rows with missing values
df_dropna = df.dropna()

# Fill missing values with a specified value
df_fillna = df.fillna(0)
# In this example, we demonstrate how to handle missing values by either dropping rows with missing values or filling them with a specified value.