# Dataset from kaggle.com Fake Currency Data. Contains synthetic data representing fake currency samples. Classification

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('users/zanna/fake_currency_data.csv')

#Colums:
#Country: Country of origin for the currency.
#Denomination: Currency denomination.
#Counterfeit: Binary indicator (0 for genuine, 1 for counterfeit).
#SerialNumber: Serial number of the currency.
#Security Features: Security features present in the currency.
#Weight: Weight of the currency in grams.
#Length: Length of the currency in mm.
#Width: Width of the currency in mm.
#Thickness: Thickness of the currency in mm

df.dtypes #checking the data types, Country, Denomination, SecurityFeatures are object, rest is int/float
df[['Country','Denomination','SecurityFeatures']] #checking if these 3 features should be 'object', Denomination could be split into value and the currency
df.describe() #counterfeit min:0,max:1, mean:0.499, median:0, Q75:1 -> there are more genuine banknotes, but the difference in quantity small
df['Counterfeit'].value_counts()
df['Country'].value_counts()
#as the general describe() didn't show all columns, the range had to be narrowed
df[['Weight','Length','Width','Thickness']].describe() #judging by the statistics there are no outliers

df.Counterfeit.unique() #Counterfeit contains only 0 and 1
df.Country.unique() #USA, EU, UK -> one-hot encoding, then the currency wont be needed
df.SecurityFeatures.unique() #Hologram, Security Thread, Microprint, Watermark ->one-hot encoding
df.isnull().sum() #there are no nulls

df['D'] = df['Denomination'].str[1:]

#one-hot encoding for Country and SecurityFeatures
df_dummies=pd.get_dummies(df[['Country','SecurityFeatures']])
df_encoded = pd.concat([df, df_dummies], axis=1)
df_encoded.dtypes
df_encoded = df_encoded.drop(['Country', 'SecurityFeatures','Denomination','SerialNumber'], axis=1)
df_encoded.D = df_encoded.D.str.replace(',', '').astype(int) #converting coulmn D (denomination) into int

corr = df_encoded.corr()
sns.heatmap(corr, annot=True)
#correlation is small

#Setting the features and target variable in the dataset
X = df_encoded.drop('Counterfeit', axis=1)
y = df_encoded['Counterfeit']

#Creating training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Features standarization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#Model 1: Logistic Regression

from sklearn.linear_model import LogisticRegression

# Initialize and train a logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
accuracy_LR = log_reg.score(X_test_scaled, y_test)
print("Model accuracy:", accuracy_LR)
#accuracy: 0.50188

import optuna

def objective_lr(trial):
    C = trial.suggest_loguniform('C', 0.00001, 10000.0)
    model = LogisticRegression(C=C)
    model.fit(X_train_scaled, y_train)
    return model.score(X_test_scaled, y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective_lr, n_trials=100, show_progress_bar=True)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
#Accuracy: 0.502
#Best hyperparameters: {'C': 4.770917583949285e-05}
#Best trial: 93

#Predictions
predictions = log_reg.predict(X_test_scaled)
print("Predictions:")
for i in range(5):
    print("Predicted:", predictions[i], "| Actual:", y_test.iloc[i])

#Evaluating Model Performance
from sklearn.metrics import classification_report, confusion_matrix

# Generate a classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


#Model 2: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Initialize and train a RandomForestClassifier model
rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy_clf = rnd_clf.score(X_test, y_test)
print("Model accuracy:", accuracy_clf)
#accuracy: 0.501405

def objective_rf(trial):

    n_estimators = trial.suggest_int('n_estimators', 2, 20)
    max_depth = int(trial.suggest_float('max_depth', 1, 32, log=True))

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train,y_train)

    # Make predictions and calculate RMSE
    y_pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae


study = optuna.create_study(direction='minimize')
study.optimize(objective_rf, n_trials=100, show_progress_bar=True)

trial = study.best_trial

print('Best trial: ')
print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
#Accuracy: 0.496915
#Best hyperparameters: {'n_estimators': 9, 'max_depth': 3.643172400379503}


#Both models Logistic regression and random forest classifier have similar accuracy LR(0.50188) RF(0.501405









