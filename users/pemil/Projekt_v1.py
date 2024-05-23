import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from scipy import stats
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')
%matplotlib inline

df = pd.read_csv('CarPrice_Assignment.csv')
# Display the first few rows of the DataFrame
print(df.head())
df_all = pd.DataFrame(df)
print(df_all)
## Dataframe has 205 rows x 26 columns

df.info()
#Checking the number of cars in a dataset
df["CarName"].describe()
#count               205
#top       toyota corona

# Summary statistics
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(df.describe())

#Data Vis
df.hist(bins=200,figsize=[20,10])

def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize = 12)
    plt.show()

pie_plot(df, ['fueltype', 'aspiration', 'doornumber', 'cylindernumber', 'carbody', 'enginetype', 'fuelsystem', 'enginelocation', 'drivewheel'], 3, 3)
#In this dataset, we have the most gas cars
#56% of the cars in this dataset have four doors
#The most popular carbody is sedan, next hatchback, wagon, hardtop and the last is convertible

#Checking missing values
df.duplicated().sum()
df.isna().sum()
#The dataset doesn't contain missing values or duplicates.

# Converting text values to numeric values - text values in doornumber and cylindernumber columns are converted to numeric values
df['doornumber'].replace({'two': 2, 'four': 4}, inplace=True)
df['cylindernumber'].replace({'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}, inplace=True)
df = pd.get_dummies(df, columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem'])

#Encoding the names of the cars
car_names = df['CarName'].tolist()
unique_names = []
for car in car_names:
    unique_names.append(car.split()[0])
unique_names = set(unique_names)
for i, c in enumerate(df["CarName"]):
    for u in unique_names:
        if u in c:
            df['CarName'].iloc[i] = u
df = pd.get_dummies(df, columns=['CarName'])

#Correlation
correlation_matrix = df.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation')
plt.show()

#Outliers
num_cols = ['symboling','doornumber', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
            'cylindernumber', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
            'peakrpm', 'citympg', 'highwaympg', 'price']

plt.figure(figsize=(20, 15))
for i in range(len(num_cols)):
    plt.subplot(5, 4, i + 1)
    sns.boxplot(df[num_cols[i]], palette="flare")
    plt.title(num_cols[i])

plt.tight_layout()
plt.show()

#Remove outliers
def remove_outliers(df, num_cols):
    for col in num_cols:
        z_scores = stats.zscore(df[col])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3) #Remove values with a deviation greater than 3 from the mean
        df = df[filtered_entries]
    return df

df_clean = remove_outliers(df.copy(), num_cols)

#Split the dataset into features and target variable
X = df_clean.drop(['car_ID', 'price'], axis=1)
y = df_clean['price']

#StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f'Original number of features: {X_scaled.shape[1]}')
print(f'Reduced number of features: {X_pca.shape[1]}')

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#I have continuous variables in the model so I need to use a regression model such as LinearRegression or RandomForestRegressor
#RandomForestRegressor to forecast car prices
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)
#Predicting on a test set
y_pred_rf = rf_regressor.predict(X_test)
#Evaluation of the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor:")
print("Mean Squared Error:", mse_rf)
print("R2 Score:", r2_rf)
#Random Forest Regressor:
#Mean Squared Error: 5454509.474176818
#R2 Score: 0.8308496217662109
#The R2 Score is about 0.83, which means that about 83% of the variation in car prices is explained by the model, that is, the model is able to explain and predict about 83% of the variation in car prices based on the available characteristics.

#XGBoost
xgb_regressor = xg.XGBRegressor()
xgb_regressor.fit(X_train, y_train)
#Predicting on a test set
y_pred_xgb = xgb_regressor.predict(X_test)
#Evaluation of the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Regressor:")
print("Mean Squared Error:", mse_xgb)
print("R2 Score:", r2_xgb)
#XGBoost Regressor:
#Mean Squared Error: 6780072.961453662
#R2 Score: 0.789742430311636
#The R2 Score is about 0.79. This means that about 79% of the variation in car prices is explained by the XGBoost model, so the model is able to explain and predict about 79% of the variation in car prices based on the available features.

#We define the parameter space for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("The best params:", grid_search.best_params_)

best_xgb_model = grid_search.best_estimator_
y_pred_best = best_xgb_model.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("XGBoost Regressor:")
print("Mean Squared Error:", mse_best)
print("R2 Score:", r2_best)

#The results show that after tuning the hyperparameters, the XGBoost model has an MSE = 5051816.64, suggesting that predicted prices differ on average by this squared difference from actual prices.
#R^2 = 0.84, which means that the model explains 84% of the variation in the data, which is a very good result for a regression model.
#This means that the XGBoost model, after hyperparameter tuning, is a good fit and effectively predicts car prices based on available characteristics.