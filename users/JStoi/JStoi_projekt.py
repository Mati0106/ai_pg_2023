import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import shap

# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
df = pd.read_csv('diabetes.csv')

# Chcecking if there are any null values or values equal to 0
df = df.dropna()  # Deleting rows with only null values
# Chcecking if there are any text columns
print(df.dtypes)  # There are only integer and float values

# Looking for outliers using interquartile range
columns = df.columns
for _ in columns:
    Q1 = df[_].quantile(0.25)
    Q3 = df[_].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[_] < lower_bound) | (df[_] > upper_bound)]
    print(_)
    print(outliers[_])
    # We can see that there are some values that that fall below Q1 − 1.5 IQR or above Q3 + 1.5 IQR. We can see some values equal to 0.
    # We can see that we have values = 0 for Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin and BMI.
for _ in columns:
    print(_)
    print(df[df[_] == 0])
    # As for Pregnancies 0 is normal as you can have 0 kids, for other columns values are not correct, e.g. you can't have such Blood Pressure (unless you're dead :))
    # Replacing these values with mean value
columns_mean = df.columns.drop(['Pregnancies', 'Outcome'])
for _ in columns_mean:
    mean_value = df[df[_] != 0][_].mean()
    df[_] = df[_].replace(0, mean_value)

    # Marking outliers with a flag column
df['Outlier_flag'] = 0
for _ in columns:
    Q1 = df[_].quantile(0.25)
    Q3 = df[_].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df.loc[df[_] < lower_bound, 'Outlier_flag'] = 1
    df.loc[df[_] > upper_bound, 'Outlier_flag'] = 1

# Checking on a heatmap if there is a strong correlaction between features to see if we should use PCA, as we can see there is no need for tha
df_heatmap = df.drop(columns=['Outlier_flag'])
correlation_matrix = df_heatmap.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap')
plt.show()
# The highest correlaction is 0.54 between Age and Pregnancies and also BMI and Skin Thickness. For other features the correlation is close to none
# thus there is no need for using PCA

# Splitting the dataset into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# No need for standarizing y as it has only 0 and 1 values

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Creating a model and fitting with training data
# We would like to classify people as positive or negative in the context of having diabities so we will use a classificatior, Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluating the model on the test data
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)
# AUC, ROC dodac!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Looking at confusion matrix will help evaluate the model deeper
predictions = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
# In the context of any disease detection the most important condition we should focus on is False negative.
# We can see that condition is about 11% of all test observation, that seems high.

# Performing cross-validation
cv_scores = cross_val_score(model, X, y, cv=10)
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Shap values i interpretacja wynikow, shap - partial dependence plots!!!!!!!!
# Optymalizacja modeli za pomoca optuny


# import optuna
# import sklearn
# import sklearn.datasets
# import sklearn.ensemble
# import sklearn.model_selection
#
#
# def objective(trial):
#     n_estimators = trial.suggest_int('n_estimators', 2, 20)
#     max_depth = int(trial.suggest_float('max_depth', 1, 32, log=True))
#
#     # clf = sklearn.ensemble.RandomForestClassifier(
#     #     n_estimators=n_estimators, max_depth=max_depth)
#
#     return sklearn.model_selection.cross_val_score(
#         model, X, y, n_jobs=-1, cv=3).mean()
#
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)
#
# trial = study.best_trial
#
# print('Accuracy: {}'.format(trial.value))
# print("Best hyperparameters: {}".format(trial.params))
# optuna.visualization.plot_optimization_history(study)

