# Loading and Exploring a Dataset

# task - use the historical data solar plant production and weather parameters and try to predict production using
# weather forecast.
# I use the data from kaggle - /kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv
# /kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

generation_data = pd.read_csv('users/WWojc/Data/Plant_2_Generation_Data.csv')
weather_data = pd.read_csv('users/WWojc/Data/Plant_2_Weather_Sensor_Data.csv')

generation_data.info()
weather_data.info()
generation_data.head()
generation_data.describe()

# check is there any missing value
# we have complete data, without any null

# check, how many inverters we have in this plant
print('The number of inverter for data_time {} is {}'.format('16-05-2020 00:15',
generation_data[generation_data.DATE_TIME == '16-05-2020 00:15']['SOURCE_KEY'].nunique()))


# In this solar plant, we have 22 inverters, used to change DC power to AC power.
# Data set including data about energy production [kW] from every inverters (each inverters has name "source key")
# recorded in 15 minutes interval,
# weather data including parameters from one weather station on this PV plant, recorded at 15 minutes intervals:
# ambient temperature [st. C],
# module temperature [st. C]
# irradiation [Wh/m2]

# sum up values from every inverters;
# thanks to this, we have total plant productions in every 15 minutes,
generation_data = generation_data.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']].agg('sum')
generation_data = generation_data.reset_index()
generation_data.head()

# Adjust datetime format - we have two, new columns with separated information about date and time,
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], errors='coerce')
generation_data['time'] = generation_data['DATE_TIME'].dt.time
generation_data['date'] = pd.to_datetime(generation_data['DATE_TIME'].dt.date)

generation_data.info()

weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], errors='coerce')
weather_data['date'] = pd.to_datetime(weather_data['DATE_TIME'].dt.date)
weather_data['time'] = weather_data['DATE_TIME'].dt.time

weather_data.info()
generation_data.head()
weather_data.head()

# erase unneeded rows
del weather_data['PLANT_ID']
del weather_data['SOURCE_KEY']
weather_data.tail()

# DC Production analyse
generation_data.plot(x= 'time', y='DC_POWER', style='.', figsize = (15, 8))
generation_data.groupby('time')['DC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.ylabel('DC Power')
plt.title('DC POWER plot')
plt.show()

# PV plant produces energy only between 05:33:20 and 18:00:00. Over of that time, production is null - that caused to
# weak sunlight power.

# PV production analyse in every day at the same time,
calendar_dc = generation_data.pivot_table(values='DC_POWER', index='time', columns='date')
calendar_dc.tail()

def multi_plot(data=None, row=None, col=None, title='DC Power'):
    cols = data.columns  # take all column
    gp = plt.figure(figsize=(20, 20))

    gp.subplots_adjust(wspace=0.2, hspace=0.8)
    for i in range(1, len(cols) + 1):
        ax = gp.add_subplot(row, col, i)
        data[cols[i - 1]].plot(ax=ax, style='k.')
        ax.set_title('{} {}'.format(title, cols[i - 1]))

multi_plot(data=calendar_dc, row=9, col=4)

# every day production is almost similar. From time to time the production during the day is lower - it's
# might have been caused by cloudy skies


# Total daily production:
daily_dc = generation_data.groupby('date')['DC_POWER'].agg('sum')
daily_dc.plot.bar(figsize=(15, 5), legend=True)
plt.title('Daily DC Power')
plt.show()

# Total Daily production in each hours:
generation_data.plot(x='time', y='DAILY_YIELD', style='b.', figsize=(15, 5))
generation_data.groupby('time')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('DAILY YIELD')
plt.ylabel('Yield')
plt.show()

# Total Daily production increasing between 05:33:20 and 18:00:00,
daily_yield = generation_data.pivot_table(values='DAILY_YIELD', index='time', columns='date')
daily_yield.boxplot(figsize=(18, 5), rot=90, grid=False)
plt.title('DAILY YIELD IN EACH DAY')
plt.show()

# every day total production changes, but we haven't outliers


# Temperature analyse

weather_data.plot(x='time', y = 'AMBIENT_TEMPERATURE' , style='b.', figsize=(15,5))
weather_data.groupby('time')['AMBIENT_TEMPERATURE'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('Daily AMBIENT TEMPERATURE MEAN (RED)')
plt.ylabel('Temperature (°C)')
plt.show()

ambient = weather_data.pivot_table(values='AMBIENT_TEMPERATURE', index='time', columns='date')
ambient.tail()

ambient.boxplot(figsize=(15,5), grid=False, rot=90)
plt.title('AMBIENT TEMPERATURE BOXES')
plt.ylabel('Temperature (°C)')

# three days contains outliers
ambient_temp = weather_data.groupby('date')['AMBIENT_TEMPERATURE'].agg('mean')
ambient_temp.plot(grid=True, figsize=(15,5), legend=True, colormap='Oranges_r')
plt.title('AMBIENT TEMPERATURE 15 MAY- 17 JUNE')
plt.ylabel('Temperature (°C)')

# In this time, the ambient temperature in this plant was between 24 and 31 st. C. June was little coldest


weather_data.plot(x='time', y='MODULE_TEMPERATURE', figsize=(15,8), style='b.')
weather_data.groupby('time')['MODULE_TEMPERATURE'].agg('mean').plot(colormap='Reds_r', legend=True)
plt.title('DAILY MODULE TEMPERATURE & MEAN(red)')
plt.ylabel('Temperature(°C)')


# Irradiation analyse

weather_data.plot(x='time', y = 'IRRADIATION', style='.', legend=True, figsize=(15,5))
weather_data.groupby('time')['IRRADIATION'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('IRRADIATION')

irradiation = weather_data.pivot_table(values='IRRADIATION', index='time', columns='date')
irradiation.tail()

irradiation.boxplot(figsize=(15,5), rot = 90, grid=False)
plt.title('IRRADIATION BOXES')

rad = weather_data.groupby('date')['IRRADIATION'].agg('sum')
rad.plot(grid=True, figsize=(15,5), legend=True)
plt.title('IRRADIATION 15 MAY- 17 JUNE')

# Correlation

# merge power generation data with weather data

gen_wea = weather_data.merge(generation_data, left_on='DATE_TIME', right_on='DATE_TIME')
gen_wea.tail(3)

#we remove the columns that we do not need
del gen_wea['date_x']
del gen_wea['date_y']
del gen_wea['time_x']
del gen_wea['time_y']

gen_wea.tail(3)
gen_wea.info()

# spearman correlation

gen_wea.corr(method = 'spearman')

# DAILY_YIELD and TOTAL_YIELD it's not correlated with any feature.

correlation = gen_wea.drop(columns=['DAILY_YIELD', 'TOTAL_YIELD']).corr(method = 'spearman')
plt.figure(dpi=100)
sns.heatmap(correlation, robust=True, annot=True, fmt='0.3f', linewidths=.5, square=True)
plt.show()

# we can observe that:
# AC_POWER and DC_POWER are correlated, but it's normal, beacuse this two parameters we measured in input and output
# on inverter. In the further analysis, we can use only AC_POWER.
# AC_POWER is correlated with AMBIENT_TEMPERATURE, MODULE_TEMPERATURE and IRRADIATION


# we make pairplot


plt.figure(dpi=100)
sns.lmplot(x='AMBIENT_TEMPERATURE', y='AC_POWER', data=gen_wea)
plt.title('Regression plot')
plt.show()
# AC_POWER increases non linearly with an Ambient_Temperature.

plt.figure(dpi=100)
sns.lmplot(x='MODULE_TEMPERATURE', y='AC_POWER', data=gen_wea)
plt.title('Regression plot')
plt.show()

# AC_POWER is produced linearly by MODULE_TEMPERATURE with some variability.

plt.figure(dpi=100)
sns.lmplot(x='IRRADIATION', y='AC_POWER', data=gen_wea)
plt.title('Regression plot')
plt.show()

# AC_POWER increase with IRRADIATION.

# IRRADIATION of Module and Heat Transfert between ambient air and Module are very well correlated.
# AC_POWER is indeed influenced by the ambient temperature, by the temperature of the module
# and by the irradiation.


# Predicting AC_POWER

# AC_POWER generation of the plant, depend on irradiation, ambient temperature and module temperature.
# To predict AC_POWER by the weather forecast, we can use only irradiation data and ambient temperature.
# Module temperature linear depend of ambient temperature.

predict_reg = gen_wea[['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE']]
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

y=predict_reg['AC_POWER']
X=predict_reg[['IRRADIATION','AMBIENT_TEMPERATURE']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
pred_y_train = lm.predict(X_train)
pred_y_test = lm.predict(X_test)

#plt.scatter(y_test, predictions)
from sklearn.metrics import r2_score, mean_squared_error

#Model Evaluation on Training data

R2_train = r2_score(y_train, pred_y_train)
mse_train = mean_squared_error(y_train, pred_y_train)
print('R2 for Train dataset:', R2_train, '  '   'MSE for Train dataset:', mse_train)

#Model Evaluation on Testing data

R2_test = r2_score(y_test, pred_y_test)
mse_test = mean_squared_error(y_test, pred_y_test)
print('R2 for Test dataset:', R2_test, '  '   'MSE for Test dataset:', mse_test)

# R2 value of Train and Test Datasets are almost equal. Model is valid and approx. 83% of AC Power variation
# is explained by Irradiation and Ambient Temperature.

print('Slope:' ,model.coef_)
print('Intercept:', model.intercept_)


# AC Power output is highly dependent on Irradition. With 1 unit increase in Irradiation,
# AC Power output increases by approx 17.5MW With 1 deg increase in Ambient Temperature AC Power output
# increases by 120 kW.
# Using the Weather forecast data (Irradiation and Ambient Temperature), AC Power output of the plant can be
# predicted with a good accuracy.