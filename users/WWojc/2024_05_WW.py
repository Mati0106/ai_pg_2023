# Loading and Exploring a Dataset


# 22 inervters, production data [Wh] from every, every inverters has name "source key"
# wheather sensors - one weather station, we meassure ambient temperature [st. C],
# module temperature [st. C] and irradiation [Wh/m2]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




generation_data = pd.read_csv('2024_05_WW_Praca_zal/Data/Plant_1_Generation_Data.csv')
weather_data = pd.read_csv('2024_05_WW_Praca_zal/Data/Plant_1_Weather_Sensor_Data.csv')
generation_data.info()
weather_data.info()

# check, how many inverters we have in this plant
print('The number of inverter for data_time {} is {}'.format('15-05-2020 23:00', generation_data[generation_data.DATE_TIME == '15-05-2020 23:00']['SOURCE_KEY'].nunique()))

# check is there any missing value
generation_data.info()
weather_data.info()
generation_data.head()

# sum up values from every inverters
generation_data = generation_data.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']].agg('sum')
generation_data = generation_data.reset_index()
generation_data.head()

# Adjust datetime format
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

del weather_data['PLANT_ID']
del weather_data['SOURCE_KEY']
weather_data.tail()

# Production analyse

generation_data.plot(x='time', y='DC_POWER', style='.', figsize=(15, 8))
generation_data.groupby('time')['DC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')
# plt.label('DC Power')
# plt.title('DC POWER plot')
plt.show()

# show to dc power producted by Plant in each day
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

daily_dc = generation_data.groupby('date')['DC_POWER'].agg('sum')
daily_dc.plot.bar(figsize=(15, 5), legend=True)
plt.title('Daily DC Power')
plt.show()

# daily production

generation_data.plot(x='time', y='DAILY_YIELD', style='b.', figsize=(15, 5))
generation_data.groupby('time')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('DAILY YIELD')
plt.ylabel('Yield')
plt.show()

daily_yield = generation_data.pivot_table(values='DAILY_YIELD', index='time', columns='date')
# we plot all daily yield
multi_plot(data=daily_yield.interpolate(), row=9, col=4, title='DAILY YIELD')
multi_plot(data=daily_yield.diff()[daily_yield.diff() > 0], row=9, col=4, title='new yield')

daily_yield.boxplot(figsize=(18, 5), rot=90, grid=False)
plt.title('DAILY YIELD IN EACH DAY')
plt.show()

daily_yield.diff()[daily_yield.diff() > 0].boxplot(figsize=(18, 5), rot=90, grid=False)
plt.title('DAILY YIELD CHANGE RATE EACH 15 MIN EACH DAY')
plt.show(block = True)

# Only two days have an outlier 2020-03-06 and 2020-05-21.
#we compute a daily yield for each date.
dyield = generation_data.groupby('date')['DAILY_YIELD'].agg('sum')
dyield.plot.bar(figsize=(15,5), legend=True)
plt.title('Daily YIELD')
plt.show()


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

ambient_temp = weather_data.groupby('date')['AMBIENT_TEMPERATURE'].agg('mean')
ambient_temp.plot(grid=True, figsize=(15,5), legend=True, colormap='Oranges_r')
plt.title('AMBIENT TEMPERATURE 15 MAY- 17 JUNE')
plt.ylabel('Temperature (°C)')

ambient_change_temp = (ambient_temp.diff()/ambient_temp)*100
ambient_change_temp.plot(figsize=(15,5), grid=True, legend=True)
plt.ylabel('%change')
plt.title('AMBIENT TEMPERATURE %change')

# Comment
# Sunday 17 May 2020 to Monday 18 May 2020, the ambient Temperature decreases to 10%.
# Monday 18 May 2020 to Tuesday 19 May 2020, the ambient Temperature increases to 15% and tomorrow decreases to 5%.
# Wednesday 20 May 2020 to Thursday 21 May 2020, the ambient Temperature increases to 10% and tomorrow decreases to 15%.
# June month's, the ambiant Temperature %change stabilize between -2.5 and 2.5%.

from scipy.signal import periodogram

decomp = sm.tsa.seasonal_decompose(ambient_temp)
cols = ['trend', 'seasonal', 'resid']  # take all column
data = [decomp.trend, decomp.seasonal, decomp.resid]
gp = plt.figure(figsize=(15, 15))

gp.subplots_adjust(hspace=0.5)
for i in range(1, len(cols) + 1):
    ax = gp.add_subplot(3, 1, i)
    data[i - 1].plot(ax=ax)
    ax.set_title('{}'.format(cols[i - 1]))

weather_data.plot(x='time', y='MODULE_TEMPERATURE', figsize=(15,8), style='b.')
weather_data.groupby('time')['MODULE_TEMPERATURE'].agg('mean').plot(colormap='Reds_r', legend=True)
plt.title('DAILY MODULE TEMPERATURE & MEAN(red)')
plt.ylabel('Temperature(°C)')

module_temp = weather_data.pivot_table(values='MODULE_TEMPERATURE', index='time', columns='date')
module_temp.boxplot(figsize=(15,5), grid=False, rot=90)
plt.title('MODULE TEMPERATURE BOXES')
plt.ylabel('Temperature (°C)')

mod_temp = weather_data.groupby('date')['MODULE_TEMPERATURE'].agg('mean')
mod_temp.plot(grid=True, figsize=(15,5), legend=True)
plt.title('MODULE TEMPERATURE 15 MAY- 17 JUNE')
plt.ylabel('Temperature (°C)')

# May month's have: 2 huges hot date 21 and 29.
chan_mod_temp = (mod_temp.diff()/mod_temp)*100
chan_mod_temp.plot(grid=True, legend=True, figsize=(15,5))
plt.ylabel('%change')
plt.title('MODULE TEMPERATURE %change')

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

# N.B Thursday 21 May 2020 is a date where plant 1 are:
# more produce dc power.
# ambient temperature, module temperature are maximun.
# This date is very special.

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

# DAILY_YIELD is not correlated with all feature but AMBIENT_TEMPERATURE is moreless correlated.
# TOTAL_YIELD is also not correlated with all feature. I remove it in the correlation matrix.

correlation = gen_wea.drop(columns=['DAILY_YIELD', 'TOTAL_YIELD']).corr(method = 'spearman')
plt.figure(dpi=100)
sns.heatmap(correlation, robust=True, annot=True, fmt='0.3f', linewidths=.5, square=True)
plt.show()

# we make pairplot
sns.pairplot(gen_wea.drop(columns=['DAILY_YIELD', 'TOTAL_YIELD']))
plt.show()

#we plot dc power vs ac power
plt.figure(dpi=100)
sns.lmplot(x='DC_POWER', y='AC_POWER', data=gen_wea)
plt.title('Regression plot')
plt.show()

#This graph said that inverter convert dc power to ac power linearly.  dcpower=10∗acpower
 # inverter lost 90% of their power when it convert.

plt.figure(dpi=100)
sns.lmplot(x='AMBIENT_TEMPERATURE', y='DC_POWER', data=gen_wea)
plt.title('Regression plot')
plt.show()
# DC_power increases non linearly with an Ambient_Temperature.

plt.figure(dpi=100)
sns.lmplot(x='MODULE_TEMPERATURE', y='DC_POWER', data=gen_wea)
plt.title('Regression plot')
plt.show()

# DC_POWER is produced linearly by MODULE_TEMPERATURE with some variability.

plt.figure(dpi=100)
sns.lmplot(x='IRRADIATION', y='DC_POWER', data=gen_wea)
plt.title('Regression plot')
plt.show()

# DC_Power increase with IRRADIATION.

# What happens if I introuduce a difference Temperature between AMBIENT_TEMPERATURE AND MODULE_TEMPERATURE.
# we introduce DELTA_TEMPERATURE
gen_wea['DELTA_TEMPERATURE'] = abs(gen_wea.AMBIENT_TEMPERATURE - gen_wea.MODULE_TEMPERATURE)
# we check if all is ok
gen_wea.tail(3)

# now we use correlation
gen_wea.corr(method='spearman')['DELTA_TEMPERATURE']
# we remark that YIELD does not depend on DELTA_TEMPERATURE also.
sns.lmplot(x='DELTA_TEMPERATURE', y='DC_POWER', data=gen_wea)
plt.title('correlation between DC_POWER and DELTA_TEMPERATURE')

sns.lmplot(x='DELTA_TEMPERATURE', y='IRRADIATION', data=gen_wea)
plt.title('Regression plot')

# IRRADIATION of Module and Heat Transfert between ambient air and Module are very well correlated.
#In this section, we conclude that:
#Yield does not depend on the Temperature, the dc/ac power and irradiation.
#the transfert function between dc and ac power is linear.
#dc power is indeed influenced by the ambient temperature, by the temperature of the module, by the irradiation and finally by the heat transfer between the module and the air.
#all 22 Inverters of Plant I lost 90% of their dc power when it convert.

