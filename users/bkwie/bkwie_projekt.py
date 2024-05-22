# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#dataset Kaggle source: World Happiness Report- 2024
# World Happiness Report for 2024 & World Happiness Report from 2005-2024
# https://www.kaggle.com/datasets/jainaru/world-happiness-report-2024-yearly-updated/data

# About Dataset
# Context:
#
# The World Happiness Report is a landmark survey of the state of global happiness . The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.
#
# Content:
#
# Here's a brief explanation of each column in the dataset:
#
# Country name: Name of the country.
# Regional indicator: Region to which the country belongs.
# Ladder score: The happiness score for each country, based on responses to the Cantril Ladder question that asks respondents to think of a ladder, with the best possible life for them being a 10, and the worst possible life being a 0.
# Upper whisker: Upper bound of the happiness score.
# Lower whisker: Lower bound of the happiness score.
# Log GDP per capita: The natural logarithm of the country's GDP per capita, adjusted for purchasing power parity (PPP) to account for differences in the cost of living between countries.
# Social support: The national average of binary responses(either 0 or 1 representing No/Yes) to the question about having relatives or friends to count on in times of trouble.
# Healthy life expectancy: The average number of years a newborn infant would live in good health, based on mortality rates and life expectancy at different ages.
# Freedom to make life choices: The national average of responses to the question about satisfaction with freedom to choose what to do with one's life.
# Generosity: The residual of regressing the national average of responses to the question about donating money to charity on GDP per capita.
# Perceptions of corruption: The national average of survey responses to questions about the perceived extent of corruption in the government and businesses.
# Dystopia + residual: Dystopia is an imaginary country with the world’s least-happy people, used as a benchmark for comparison. The dystopia + residual score is a combination of the Dystopia score and the unexplained residual for each country, ensuring that the combined score is always positive. Each of these factors contributes to the overall happiness score, but the Dystopia + residual value is a benchmark that ensures no country has a lower score than the hypothetical Dystopia.
# Positive affect: The national average of responses to questions about positive emotions experienced yesterday.
# Negative affect: The national average of responses to questions about negative emotions experienced yesterday.

# read data
# dynamic: 2002-2023; 2024: 2024 year only
# discrepancies in column names & columns between 2 datasets

# I updated Türkiye to Turkiye in dynamic dataset manually!! (as it wasn't displaying properly in Notepad++)

df_dynamic= pd.read_csv("users/bkwie/World-happiness-report-updated_2024.csv", encoding='latin-1')
df_2024= pd.read_csv("users/bkwie/World-happiness-report-2024.csv", encoding='latin-1')
ctry_map= pd.read_csv("users/bkwie/filled_mappings.csv", encoding='latin-1')
print(df_dynamic.head(5))
print(df_2024.head(5))
# print(ctry_map)
# MAPPING CREATED MANUALLY FOR MISSING CNTRY (+ therefore REGION) IN 2024 dataset:
# Country name,Regional indicator
# Angola,Sub-Saharan Africa
# Belarus,Central and Eastern Europe
# Belize,Latin America and Caribbean
# Bhutan,South Asia
# Burundi,Sub-Saharan Africa
# Central African Republic,Sub-Saharan Africa
# Cuba,Latin America and Caribbean
# Djibouti,Sub-Saharan Africa
# Guyana,Latin America and Caribbean
# Haiti,Latin America and Caribbean
# Maldives,South Asia
# Oman,Middle East and North Africa
# Qatar,Middle East and North Africa
# Rwanda,Sub-Saharan Africa
# Somalia,Sub-Saharan Africa
# Somaliland region,Sub-Saharan Africa
# South Sudan,Sub-Saharan Africa
# Sudan,Sub-Saharan Africa
# Suriname,Latin America and Caribbean
# Syria,Middle East and North Africa
# Trinidad and Tobago,Latin America and Caribbean
# Turkmenistan,Commonwealth of Independent States

# PLUS:
# I updated Türkiye to Turkiye in dynamic dataset manually!! as it wasn't displaying correctly in Notepad++

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

#correlation matrix - df_2024 - between Ladder, upperwhisker & lowerwhisker
df_2024.iloc[:, 2:5].corr()

#the correlation is almost 1, dropping upperwhisker & lowerwhisker columns from 2024 dataset
#additionally - dropping 'Dystopia + residual' column, as it's not present in the dynamic dataset
df_2024 = df_2024.drop(['upperwhisker', 'lowerwhisker','Dystopia + residual'], axis=1)

#adding year to 2024 dataset
df_2024.insert(2, "year", 2024, True)
df_2024.info()

#removing 2 columns from historical data - they are not shown in 2024 data
df_dynamic = df_dynamic.drop(['Positive affect', 'Negative affect'], axis=1)
#adding Regional indicator to previous data
df_dynamic.insert(1, "Regional indicator", "", True)
#checking Regional indicators for countries - region & underlying countries list
# region_mappings = df_2024.groupby('Regional indicator')['Country name'].unique().reset_index()
# print(region_mappings)
# mappings from 2024 data for region:
# country_region = df_2024.iloc[:, 0:2]

# Merge df_dynamic with df_2024 on the 'country' column - to retrieve Region indicator
merged_dynamic = df_dynamic.merge(df_2024, on='Country name', how='left', suffixes=('', '_ref'))

# Fill missing values in the 'region' column of df1 with corresponding values from 'region_ref' column from df2
merged_dynamic['Regional indicator'] = merged_dynamic['Regional indicator_ref']

# Drop the 'region_ref' column as it was only used for filling missing values
merged_dynamic = merged_dynamic.drop(['Perceptions of corruption_ref', 'Generosity_ref', 'Freedom to make life choices_ref','Healthy life expectancy', 'Regional indicator_ref',  \
                                    'Social support_ref','Log GDP per capita_ref', 'Ladder score', 'year_ref'], axis=1)

# Merge merged_dynamic with df_2024 on the 'country' column - to retrieve Region indicator
new_merged_dynamic = merged_dynamic.merge(ctry_map, on='Country name', how='left', suffixes=('', '_ref'))

# Fill missing values in the 'region' column of df1 with corresponding values from 'region_ref' column from additional ctry region mappings
new_merged_dynamic.insert(1, "Regional indicator_temp", "", True)
new_merged_dynamic['Regional indicator_temp'] = new_merged_dynamic['Regional indicator'].str.cat(new_merged_dynamic['Regional indicator_ref'], sep='', na_rep='')
new_merged_dynamic['Regional indicator'] = new_merged_dynamic['Regional indicator_temp']
new_merged_dynamic = new_merged_dynamic.drop(['Regional indicator_temp', 'Regional indicator_ref'], axis=1)

# Drop the 'region_ref' column as it was only used for filling missing values
#merged_dynamic = merged_dynamic.drop(['Regional indicator_ref'], axis=1)

new_merged_dynamic.info()
df_2024.info()
#now we have the same number of columns in both datasets, we need to rename 2 of them in df_2024 to make it same as in dynamic:
# Ladder score >> Life Ladder
# Healthy life expectancy >> Healthy life expectancy at birth
df_2024.rename(columns={
    "Ladder score": "Life Ladder",
    "Healthy life expectancy": "Healthy life expectancy at birth"
},inplace=True)

#now, we have 2 dataframes with same columns - let's merge them:
df_all = pd.concat([new_merged_dynamic, df_2024], axis=0)
df_all.info()

#missing values count
df_all.isnull().sum()
#output:
# Country name                          0
# Regional indicator                    0
# year                                  0
# Life Ladder                           0
# Log GDP per capita                   31
# Social support                       16
# Healthy life expectancy at birth     66
# Freedom to make life choices         39
# Generosity                           84
# Perceptions of corruption           128

# # Handling Missing Values: Fill non-numeric columns in the dynamic dataset
numeric_cols = df_all.select_dtypes(include=np.number).columns
df_all[numeric_cols] = df_all[numeric_cols].fillna(df_all[numeric_cols].mean())

# #stats
agg_df = df_all.pivot_table(
    values=['Country name'],
    index=['Regional indicator'],
    aggfunc=['count']
).reset_index()
print(agg_df)

# # Display the shape of the DataFrame
print("\nJoint dataframes Shape:", df_all.shape)

# Boxplots - showing error in pycharm, but the boxplot is displayed
plt_all.boxplot = df_all.boxplot(column=['Log GDP per capita','Social support','Healthy life expectancy at birth','Freedom to make life choices','Generosity','Perceptions of corruption'])

# Show the plot
plt_all.show()
# visual analysis: most outliers found in below columns:
# 'Log GDP per capita'
# 'Healthy life expectancy at birth'

#correlation matrix
corr_all = df_all.iloc[:, 3:11].corr()
#corr_all.to_csv('corr_all.csv', index=False)

# plt.matshow(df_all.iloc[:, 2:11].corr())
# plt.show()

# correlation heatmap - seaborn heatmap for correlation
f, ax = plt.subplots(figsize=(10, 8))
corr = df_all.iloc[:, 2:11].corr()
sns.heatmap(corr,
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True, ax=ax)

#visualizing each feature's skewness with 2:11 / without year 3:11
sns.pairplot(df_all.iloc[:, 3:11])
#most of the distributions seem slightly skewed
#outliers - also visualized on these pairplots charts

#Life Ladder - per Regional Indicators - SNS Kernel Density Estimate plot
#showing Life Ladder by Regional Indicator - all historical data
plt.figure(figsize = (15,8))
sns.kdeplot(x=df_all['Life Ladder'], hue = df_all['Regional indicator'], fill = False, linewidth = 1)
plt.axvline(df_all['Life Ladder'].mean(),c= 'Orange')
plt.title('Life Ladder by Regional Indicator - all historical data')
plt.show()

#LL_mean: 5.486077414205905 - our benchmark for regression
LL_mean = df_all['Life Ladder'].mean()

# handling outliers - cap / floor approach selected
# previous boxplot & pairplot visual analysis: most outliers found in below columns:
# 'Log GDP per capita' - wouldn't make sense to change, statistical data
# 'Healthy life expectancy at birth' - can impact model

df_all['Healthy life expectancy at birth'].describe()
# Results:
# count    2506.000000
# mean       59.793905
# std        15.851697
# min         0.000000
# 25%        57.670000
# 50%        64.432500
# 75%        68.042500
# max        74.600000
# IQR = 75% - 25% = 68.042500 - 57.670000 = 10.372500000000002

# flooring & capping the outliers - replace them with the nearest non-outlier value.
# This approach maintains the overall distribution while reducing the influence of extreme values.
Q1 = df_all['Healthy life expectancy at birth'].quantile(0.25)
Q3 = df_all['Healthy life expectancy at birth'].quantile(0.75)
IQR = Q3 - Q1
lower_bound_iqr = Q1 - 1.5 * IQR
upper_bound_iqr = Q3 + 1.5 * IQR

df_all['Healthy life expectancy at birth_capped'] = df_all['Healthy life expectancy at birth'].apply(lambda x: lower_bound_iqr if x < lower_bound_iqr else upper_bound_iqr if x > upper_bound_iqr else x)
df_all['Healthy life expectancy at birth_capped'].describe()
# count    2506.000000
# mean       62.153288
# std         8.043498
# min        42.111250 = lower_bound_iqr
# 25%        57.670000
# 50%        64.432500
# 75%        68.042500
# max        74.600000 = upper_bound_iqr

#split the dataset into X and Y and train and test datasets
y = df_all['Life Ladder']
y.describe()
X = df_all.drop(['Life Ladder', 'Healthy life expectancy at birth'], axis=1)
X.describe()
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_test.to_csv('X_test_columns.csv', index=False)

#LINEAR REGRESSION MODEL - START OF SECTION:
#define the columns to be standardized
Cols = ['year','Log GDP per capita','Social support','Freedom to make life choices','Generosity','Perceptions of corruption','Healthy life expectancy at birth_capped']
#standardize on dataframe copies
X_train_01 = X_train
X_test_01 = X_test
Std_Scaler = StandardScaler()
X_train_01[Cols] = Std_Scaler.fit_transform(X_train[Cols])
X_test_01[Cols] = Std_Scaler.fit_transform(X_test[Cols])

#linear regression - after standardizing numerical columns

# Create a Linear Regression model - for numerical standardized columns
modelLinReg = LinearRegression()

# Train the model on the stadardizeda data
modelLinReg.fit(X_train_01[Cols], y_train)

#display coeficcients - similar to Shapley values
print(f"Intercept: {modelLinReg.intercept_}")
print(f"Coefficients: {modelLinReg.coef_}")
#Intercept: 5.482826347305389 - expected Life ladder with all variables at their mean, LL mean from previous statistics = 5.486077414205905 - very similar value - expected
#Coefficients: [-0.0111051   0.23932288  0.58133361  0.16964888  0.06956047 -0.22980423
 # 0.35923068]
# this means the highest impoact on the model results is from the 3rd variable: Social support
# and the highest negative impact on the outcome is from Perceptions of corruption
# note all the variables have been standardized, so to compute total impact we'll need to reverse standardization
# and compute further

y_pred = modelLinReg.predict(X_test_01[Cols])

#LL_mean: 5.486077414205905 - our benchmark for regression
#LL_mean = df_all['Life Ladder'].mean() - "benchmark" predictions vector:
y_benchmark = y_pred #same length
y_benchmark[:] = LL_mean

# Calculate Mean Squared Error - This measures the average squared difference between the observed actual outcomes
# and the predictions. A lower value indicates a better fit.
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
#Mean Squared Error: 0.4130467453586691
#This indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
#An R² value close to 1 indicates a good fit.

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")
#0.6673770479884817 - not a bad fit, but could be better

mse_vs_benchmark = mean_squared_error(y_benchmark, y_test)
print(f"Mean Squared Error ytest vs benchmark: {mse_vs_benchmark}")
#Mean Squared Error - ytest vs benchmark (mean Ladder score): 1.2419551001256066 -
# way higher than from linear regression model: Mean Squared Error: 0.4130467453586691

#END OF LINEAR REGRESSION SECTION - with standard scaler in use, coefficients & verification vs benchmark


#XGBoost - start of section

# Convert data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train[Cols], label=y_train)
dtest = xgb.DMatrix(X_test[Cols], label=y_test)

# Define parameters for XGBoost
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the XGBoost model
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
y_predXGB = xgb_model.predict(dtest)

# Evaluate the model
XGBmse = mean_squared_error(y_test, y_predXGB)
print("Mean Squared Error XBGOOST:", XGBmse)
#Mean Squared Error XBGOOST: 0.30690809800412866 - lower than linear regression Mean Squared Error: 0.4130467453586691
#and way lower than Mean Squared Error - ytest vs benchmark (mean Ladder score): 1.2419551001256066 -
#XGBOOST - END OF SECTION

#OPTUNA
#how to make XGBOOST RESULTS EVEN BETTER?
def objective(trial):
    # Suggest values for `max_depth` and `eta`
    max_depth = trial.suggest_int('max_depth', 3, 10)
    eta = trial.suggest_loguniform('eta', 0.05, 0.5)

    # Set up XGBoost parameters
    param = {
        'max_depth': max_depth,
        'eta': eta,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # Train the model
    dtrain = xgb.DMatrix(X_train[Cols], label=y_train)
    dtest = xgb.DMatrix(X_test[Cols], label=y_test)
    model = xgb.train(param, dtrain, evals=[(dtest, "validation")], early_stopping_rounds=10, verbose_eval=False)

    # Predict and calculate RMSE
    preds = model.predict(dtest)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return rmse

# Create and run the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# Display the best trial
print("Best trial:")
trial = study.best_trial

print(f"  RMSE: {trial.value}")
print(f"  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train final model with the best parameters
best_params = trial.params
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)
# Best trial results:
#   RMSE: 0.4990366074397528 - higher RMSE than using the single XGBOOST MODEL ABOVE, SOMETHING WENT WRONG?
#   Params:
#     max_depth: 9
#     eta: 0.3158467951775049
