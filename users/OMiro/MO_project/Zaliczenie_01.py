# Importing required packages in one place to keep it tidy and clean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import os
print(os.listdir('ai_pg_2023/users/OMiro/MO_Project'))
# Any results written in the current directory will be saved as output.

import missingno as msno


energy = pd.read_csv('ai_pg_2023/users/OMiro/MO_Project/all_energy_statistics.csv', parse_dates=True)
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

# Now we are going to focus on finding the missing values
missing = pd.concat([energy.isnull().sum(), 100 * energy.isnull().mean()], axis=1)
missing.columns=['Count', '%']
missing.sort_values(by='Count', ascending=False)

# using missingno package to check for the missingness
msno.matrix(energy)
plt.show()

# checking the quality of footnotes to see if we need those or not
energy['quantity_footnotes'].unique()

# Since quantity_footnotes is missing lots of value, we are going to drop this column and also it is not adding any value to our analysis
df1 = energy.drop(['quantity_footnotes'], axis=1)
df1.columns

# To avoid any problem on the country_or_area names in the upper and lower, we are going to convert all the names in this & other column to lower case
df1['country_or_area'] = df1['country_or_area'].str.lower()
df1['commodity_transaction'] = df1['commodity_transaction'].str.lower()
df1['unit'] = df1['unit'].str.lower()
df1['category'] = df1['category'].str.lower()
df1.head()

# Simplifying the column names by renaming it
df1.columns = ['country', 'commodity', 'year', 'unit', 'quantity', 'category']
df1.head()

# finding the unique years for the data (to check if that data is from 1990 to 2014)
df1['year'].unique()

# exploring more about the category
df1['category'].unique()

# exploring more about the commodity
df1['commodity'].unique()

# Since we are going to focus on the G7 (and Poland) countries, so we are going to create a function to select a country.
def select_country(countryname):
    country_data = df1[df1.country.isin(countryname)].sort_values('year').reset_index()
    country_data.drop('index', axis=1, inplace=True)
    return country_data

# Generating data for each country

us_data = select_country(['united states'])
canada_data = select_country(['canada'])
germany_data = select_country(['germany'])
france_data = select_country(['france'])
italy_data = select_country(['italy'])
japan_data = select_country(['japan'])
uk_data = select_country(['united kingdom'])
poland_data = select_country(['poland'])

# now selecting the category by creating a function
def select_category(country_data, categoryname):
    category_country = country_data[country_data.category.isin(categoryname)].sort_values('year').reset_index()
    category_country.drop('index', axis=1, inplace=True)
    return category_country

# EXPLORATORY DATA ANALYSIS AND INSIGHTS

# Generating category for each country for the conventional crude oil
crude_us = select_category(us_data, ['conventional_crude_oil'])
crude_canada = select_category(canada_data, ['conventional_crude_oil'])
crude_germany = select_category(germany_data, ['conventional_crude_oil'])
crude_france = select_category(france_data, ['conventional_crude_oil'])
crude_italy = select_category(italy_data, ['conventional_crude_oil'])
crude_japan = select_category(japan_data, ['conventional_crude_oil'])
crude_uk = select_category(uk_data, ['conventional_crude_oil'])
crude_poland = select_category(poland_data, ['conventional_crude_oil'])

# Filtering further for the commodity type
def select_commodity(category_country, commodityname):
    commodity_country = category_country[category_country.commodity.isin(commodityname)].sort_values('year').reset_index()
    commodity_country.drop('index', axis=1, inplace=True )
    return commodity_country

# Filtering data for the conventional crude oil production for the G7 (+Poland) countries

crudeprod_us = select_commodity(crude_us,['conventional crude oil - production'])
crudeprod_canada = select_commodity(crude_canada,['conventional crude oil - production'])
crudeprod_germany = select_commodity(crude_germany,['conventional crude oil - production'])
crudeprod_france = select_commodity(crude_france,['conventional crude oil - production'])
crudeprod_italy = select_commodity(crude_italy,['conventional crude oil - production'])
crudeprod_japan = select_commodity(crude_japan,['conventional crude oil - production'])
crudeprod_uk = select_commodity(crude_uk,['conventional crude oil - production'])
crudeprod_poland = select_commodity(crude_poland,['conventional crude oil - production'])

# plotting line graph for each G7 (+Poland) country for the Conventional Crude Production

plt.figure(figsize=(15, 10))
x1 = crudeprod_us.year
y1 = crudeprod_us.quantity
plt.plot(x1, y1, label='United States')
x2 = crudeprod_canada.year
y2 = crudeprod_canada.quantity
plt.plot(x2, y2, label='Canada')
x3 = crudeprod_germany.year
y3 = crudeprod_germany.quantity
plt.plot(x3, y3, label='Germany')
x4 = crudeprod_france.year
y4 = crudeprod_france.quantity
plt.plot(x4, y4, label='France')
x5 = crudeprod_italy.year
y5 = crudeprod_italy.quantity
plt.plot(x5, y5, label='Italy')
x6 = crudeprod_japan.year
y6 = crudeprod_japan.quantity
plt.plot(x6, y6, label='Japan')
x7 = crudeprod_uk.year
y7 = crudeprod_uk.quantity
plt.plot(x7, y7, label='United Kingdom')
x8 = crudeprod_poland.year
y8 = crudeprod_poland.quantity
plt.plot(x8, y8, label='Poland')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Year")
plt.ylabel("Conventional Crude Oil Production (metric tons, thousand)")
plt.title("Conventional Crude Oil Production for the G7 (+Poland) Countries")
plt.legend(loc='best')
plt.show()