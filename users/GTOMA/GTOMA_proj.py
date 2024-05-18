# PROJECT:
# Download data from kaggle.com - analyze the problem with data

# The data set ifood_df.csv consists customers of XYZ company with data on:
#   Customer profiles
#   Product preferences
#   Campaign successes/failures
#   Channel performance

#   ID=Customer's unique identifier
#   Year_Birth=Customer's birth year
#   Education=Customer's education level
#   Marital_Status=Customer's marital status
#   Income=Customer's yearly household income
#   Kidhome=Number of children in customer's household
#   Teenhome=Number of teenagers in customer's household
#   Dt_Customer=Date of customer's enrollment with the company
#   Recency=Number of days since customer's last purchase
#   MntWines=Amount spent on wine in the last 2 years
#   MntFruits=Amount spent on fruits in the last 2 years
#   MntMeatProducts=Amount spent on meat in the last 2 years
#   MntFishProducts=Amount spent on fish in the last 2 years
#   MntSweetProducts=Amount spent on sweets in the last 2 years
#   MntGoldProds=Amount spent on gold in the last 2 years
#   NumDealsPurchases=Number of purchases made with a discount
#   NumWebPurchases=Number of purchases made through the company's web site
#   NumCatalogPurchases=Number of purchases made using a catalogue
#   NumStorePurchases=Number of purchases made directly in stores
#   NumWebVisitsMonth=Number of visits to company's web site in the last month
#   AcceptedCmp3=1 if customer accepted the offer in the 3rd campaign, 0 otherwise
#   AcceptedCmp4=1 if customer accepted the offer in the 4th campaign, 0 otherwise
#   AcceptedCmp5=1 if customer accepted the offer in the 5th campaign, 0 otherwise
#   AcceptedCmp1=1 if customer accepted the offer in the 1st campaign, 0 otherwise
#   AcceptedCmp2=1 if customer accepted the offer in the 2nd campaign, 0 otherwise
#   Response=1 if customer accepted the offer in the last campaign, 0 otherwise
#   Complain=1 if customer complained in the last 2 years, 0 otherwise
#   Country=Customer's location


# Zadanie 1 - upload of data + analyze

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('datasets/ifood_df.csv')

# Data analysis
print(df.head())  # Preview first few rows
print(df.info())   # Information about the data, checking types and missing values

# The target variable AcceptedCmpOverall consists of binary (0 or 1) values,
# it indicates a classification problem. Therefore, clf is performing classification.

# Check for null values
print(df.isnull().sum())  # Sum of missing values in each column

# Outlier detection
# Methods such as boxplot or defining custom rules like z-score

# Correlations
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# What is the problem:
# 1 Missing Data: Some cells in the CSV file are empty or contain null values.
# 2 Imbalanced Classes: Imbalanced classes can lead to improper model learning, favoring the dominant class.
# 3 Outliers: It will be necessary to identify and potentially process these outlier values.
# 4 High Dimensionality: The data may be high-dimensional, meaning it has a large number of features.
# Considering these issues and applying appropriate data processing techniques is crucial for effective machine learning model training.

# Conclusion
# Analyzing the correlation heatmap helps identify highly correlated variables
# which may be redundant. That's why, I can decide to remove one of the variables
# if they are very highly correlated

# Zadanie 2 - Feature engennering

#Age variable instead of year of birth
#Numb_Days variable counts days a customer is in the company
#Family variable counts the number of family members
#TotalPurchases total purchase places
#Variable Spending is a sum of the amount spent on product categories
#TotalCampaignsAcc is total acceptance of advertising campaigns


data['Age'] = 2023-data['Year_Birth']
data['Customer_Days'] = data['Dt_Customer'].max() - data['Dt_Customer']
data['Family'] = data['Marital_Status']+data['Kidhome']+data['Teenhome']
data['TotalPurchases'] = data['NumWebPurchases']+data['NumCatalogPurchases']+data['NumStorePurchases']
data['Spending'] = data.filter(like='Mnt').sum(axis=1)
data['TotalCampaignsAcc'] = data.filter(like='Accepted').sum(axis=1)+data['Response']

data[['Age','Customer_Days','Family','TotalPurchases','Spending','TotalCampaignsAcc']].head(3)

data['Customer_Days'] = data['Customer_Days'].astype(str).str.replace(' days', '')
# Convert the column to integer data type
data['Customer_Days'] = pd.to_numeric(data['Customer_Days'], downcast='integer')

data.drop(['Year_Birth','Dt_Customer'],axis=1,inplace=True)

#3. Zadanie 3  - Modelling

# Importing models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# Splitting data into train and test sets
X = df.drop(columns=['Response'])  # Model input
y = df['Response']  # Model output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Benchmark model
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
roc_lr = roc_auc_score(y_test, y_pred_lr)

print("Accuracy for Logistic Regression:", acc_lr)
print("ROC AUC for Logistic Regression:", roc_lr)

# Main model
# XGBoost
model_xgb = xgb.XGBClassifier()
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
roc_xgb = roc_auc_score(y_test, y_pred_xgb)

print("Accuracy for XGBoost:", acc_xgb)
print("ROC AUC for XGBoost:", roc_xgb)

#4. Zadanie 4. - Optymalisation of model

# Import libraries
import optuna

# Define objective function
def objective(trial):
    # Parameters to optimize
    max_depth = trial.suggest_int('max_depth', 2, 10)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.1)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)

    # XGBoost model with optimized parameters
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)

# Setting up optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Displaying the best parameters
print("Best parameters:", study.best_params)

#5. Zadanie 5. - Shap'ley values and interpretation

# Import libraries
import shap

# Creating XGBoost model with selected parameters
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=500)
model.fit(X_train, y_train)

# Creating explainer object
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test)

# Shapley values
shap.summary_plot(shap_values, X_test)

# Partial Dependence Plot

from pdpbox import pdp, get_dataset, info_plots
feature_names = X.columns.tolist()
pdp_goals = pdp.pdp_isolate(model=model, dataset=X_test, model_features=feature_names, feature='MntWines')
pdp.pdp_plot(pdp_goals, 'MntWines')
plt.show()
