import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load a CSV file into a DataFrame
df = pd.read_csv('users/bike_buyers.csv')

#DATASET ANALYSING
df.head()
df.shape
df.info()
df.describe()

#How many null value is in my DF?

rows_with_nulls = df.isnull().any(axis=1).sum()
print("There are:", rows_with_nulls, "rows with nulls")

df.isna().sum()

#The 48 rows represent 4.80% of the value from DF. In view of this, I remove these rows
df_cleaned = df.dropna()
print(df_cleaned)

df_cleaned.info()

#EXPLORATORY DATA ANALYSIS

def create_plot(df_cleaned, x_label,y_label):
    # Purchased bike distribution
    plt.figure(figsize = (4, 2))
    sns.histplot(data = df_cleaned, x=x_label, bins = 'auto', hue = 'Purchased Bike')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
#Relation between Income and Purchased Bike
create_plot(df_cleaned, 'Income', 'Count')
#Relation between Age and Purchased Bike
create_plot(df_cleaned, 'Age', 'Count')
#Relation between having children and Purchased Bike
create_plot(df_cleaned, 'Children', 'Count')
#Relation between marital status and Purchased Bike
create_plot(df_cleaned, 'Marital Status', 'Count')
#Relation between gender and Purchased Bike
create_plot(df_cleaned, 'Gender', 'Count')
#Relation between education and Purchased Bike
create_plot(df_cleaned, 'Education, 'Count')
#Relation between occupation and Purchased Bike
create_plot(df_cleaned, 'Occupation', 'Count')
#Relation between having home and Purchased Bike
create_plot(df_cleaned, 'Homr Owner', 'Count')
#Relation between commute distance and Purchased Bike
create_plot(df_cleaned, 'Commute Distance', 'Count')
#Relation between region of residence and Purchased Bike
create_plot(df_cleaned, 'Region', 'Count')


#Correlation
numeric_columns = df_cleaned.select_dtypes(include=['int', 'float']).columns
correlation = df_cleaned[numeric_columns].corr()
correlation
#The highest correlation is between children and age

df_cleaned.columns
df_cleaned['Marital Status'].value_counts()
df_cleaned['Gender'].value_counts()
df_cleaned['Home Owner'].value_counts()
df_cleaned['Children'].value_counts()
df_cleaned['Cars'].value_counts()
df_cleaned['Purchased Bike'].value_counts()

#DATA PREPROCESSING & FEATURE ENGINEERING

df_cleaned_copy = df_cleaned.copy()
# ID column isn't use so I can drop it (better model cause of that?)
df_cleaned_copy= df_cleaned_copy.drop(columns=['ID'], axis=1, inplace=False)
df_cleaned_copy

#Rename name of columns with space (that's can make some errors)
df_cleaned_copy.rename(columns = {'Marital Status' : 'Marital_Status', 'Home Owner' : 'Home_Owner','Purchased Bike':'Purchased_Bike','Commute Distance':'Commute_Distance'}, inplace = True)
df_cleaned_copy

#Convers float type to int
df_cleaned_copy.info()
df_cleaned_copy['Cars'] = df_cleaned['Cars'].astype(int)
df_cleaned_copy['Age'] = df_cleaned['Age'].astype(int)
df_cleaned_copy['Children'] = df_cleaned['Children'].astype(int)
df_cleaned_copy.info()

#Change data in binary data

df_cleaned_copy['Marital_Status'] = np.where(df_cleaned_copy['Marital_Status'] == 'Married', 1, 0)
df_cleaned_copy['Gender'] = np.where(df_cleaned_copy['Gender'] == 'Female', 1, 0)
df_cleaned_copy['Purchased_Bike'] = np.where(df_cleaned_copy['Purchased_Bike'] == 'Yes', 1, 0)
df_cleaned_copy['Home_Owner'] = np.where(df_cleaned_copy['Home_Owner'] == 'Yes', 1, 0)

#In data below is more than 0-1 selection
a = {'Bachelors' :0, 'Partial College' :1,'High School' :2, 'Graduate Degree' :3, 'Partial High School' :4}
b = {'Professional' :0, 'Skilled Manual' :1,'Clerical' :2, 'Management' :3, 'Manual' :4}
c = {'0-1 Miles' :0, '5-10 Miles' :1,'1-2 Miles' :2, '2-5 Miles' :3, '10+ Miles' :4}
d = {'North America' :0, 'Europe' :1, 'Pacific' :2}
# Replacing the the values In Dataframe.
df_cleaned_copy= df_cleaned_copy.replace({'Education': a})
df_cleaned_copy= df_cleaned_copy.replace({'Occupation': b})
df_cleaned_copy= df_cleaned_copy.replace({'Commute_Distance': c})
df_cleaned_copy= df_cleaned_copy.replace({'Region': d})

df_cleaned_copy.info()


corr =df_cleaned_copy.corr()
corr

# BUILDING MODEL

from sklearn.model_selection import train_test_split

# the features - new variable
feature_cols=['Marital_Status', 'Gender', 'Income', 'Children','Education','Occupation','Home_Owner', 'Cars', 'Commute_Distance', 'Region', 'Age']
# X Variable is the features that effects Purchased
X = df_cleaned_copy[feature_cols]
# Y Variable is the Purchased_Bike
y = df_cleaned_copy['Purchased_Bike']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test

#Tree Classifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

tr = tree.DecisionTreeClassifier(random_state = 40)
tr.fit(X_train, y_train)
# Predicting Test Dataset
y_pred = tr.predict(X_test)
y_pred
# Calculating Accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
#Calculate Matrix
print(confusion_matrix(y_test, y_pred))
tr_acc = accuracy_score(y_test, y_pred)

#RandomForestClassifier

# From Sklearn Importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# taking tr Variable.
rf = RandomForestClassifier(n_estimators = 100)
# Fitting RandomForestClassifier in The Train Dataset
rf.fit(X_train, y_train)
# Predicting Test Dataset
y_pred = rf.predict(X_test)
y_pred
# Calculating Accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
#Calculate Matrix
print(confusion_matrix(y_test, y_pred))
rf_acc = accuracy_score(y_test, y_pred)

#Support Vector Machine

from sklearn import preprocessing
from sklearn.svm import SVC

svm = SVC(kernel ='linear')
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
svm.fit(X_train, y_train)
# Predicting Test Dataset
y_pred = svm.predict(X_test)
y_pred
# Calculating Accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
#Calculate Matrix
print(confusion_matrix(y_test, y_pred))
svm_acc = accuracy_score(y_test, y_pred)

#Xgboost

from xgboost import XGBClassifier
import xgboost as xgb

# Conwersion to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.3,
    'silent': 1,
    'eval_metric': 'mlogloss'
}
num_round = 100  # Liczba iteracji (epok)
bst = xgb.train(params, dtrain, num_round)

# Calculating Accuracy
preds = bst.predict(dtest)
accuracy = accuracy_score(y_test, preds)
print(f'Accuracy: {accuracy * 100:.2f}%')


#Summary
summary=pd.DataFrame(data={'Model': ['Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Machine (SVM)', 'Xgboost'], 'Accuracy': [tr_acc, rf_acc,svm_acc, 0.6178]})
print(summary)

#The accuracy is fairly less due the the limitations of the dataset - the best model is Random Forest Classifier