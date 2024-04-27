# Importing required packages in one place to keep it tidy and clean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import os
print(os.listdir("../MO_project"))
# Any results written in the current directory will be saved as output.

import missingno as msno


energy = pd.read_csv("ai_pg_2023/MO_project/all_energy_statistics.csv", parse_dates=True)
energy.head()

# Find more information about the shape, features and unique values
print("Number of rows and columns are:", energy.shape)
print("List of countries and Area are: {}".format(energy.country_or_area.unique()))

# Find more about the data from the imported file
energy.info()

# Based on the above data, it seems that there is information missing for the column 'quantity_footnotes'.
# We also need to check if all the names under 'country_or_area' are unique or not.
# We need to check the same thing for other columns such as year, category etc.

#checking for the unique values for 'country_or_area' column
len(energy['country_or_area'].unique())

#checking for the unique values for 'year' column
len(energy['year'].unique())

#checking for the unique values for 'category' column
len(energy['category'].unique())