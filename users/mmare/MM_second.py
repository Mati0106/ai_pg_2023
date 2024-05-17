import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('./users/mmare/HR.csv')

df.head()
df.info()
df.EmployeeCount.value_counts()
# Droping EmployeeCount column as it has only one type of value and EmployeeNumber as it is not useful for the analysis
df.drop(columns=['EmployeeCount' , 'EmployeeNumber'] , inplace = True)
pd.set_option('future.no_silent_downcasting', True)
# Converting Attrition from object to numerical
df['Attrition'] = df['Attrition'].replace({'Yes':1 , 'No':0})
df.head()