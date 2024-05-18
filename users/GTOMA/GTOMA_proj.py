# PROJECT:
# Download data from kaggle.com - analyze the problem with data

Lung Cancer Prediction

# Classification

# This dataset contains information on patients with lung cancer, including their age, gender, air pollution exposure,
#  alcohol use, dust allergy, occupational hazards, genetic risk, chronic lung disease, balanced diet, obesity, smoking,
# passive smoker, chest pain, coughing of blood, fatigue, weight loss ,shortness of breath ,wheezing ,
# swallowing difficulty ,clubbing of finger nails and snoring

# Basing on data above I will build a mode to asset Lung Cancer Prediction (Cancer/ No_Cancer)


# EXERCISE 1. Data read and analyze

# Import packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data from the file cancer_data.csv to DataFrame "df"
df = pd.read_csv('datasets/cancer_data.csv')

# Data analysis
# Preview first few rows
print(df.head())
# Information about the data, checking types and missing values
print(df.info())

# Find more information about the shape, features and unique values
print("Number of rows and columns are:", df.shape)

# Preview the column names and data types
print("Column Names and Data Types:")
print(df.dtypes)

# Preview charts
import matplotlib.pyplot as plt
df.hist(figsize=(12,12))
plt.show()

#Preview data heat map
plt.figure(figsize=(16,16))
sns.heatmap(df.corr(),annot=True,fmt=".0%")
plt.show()

# Preview statistics
print("Summary Statistics:")
print(df.describe())

puste=df.isnull().sum()
print("Suma pustych wierszy:",puste)


# EXERCISE 2. Feature engennering

# Mapping

mapping={"Cancer":0, "No_Cancer":1}

df["Output"]=[mapping[value] for value in df["Lung Cancer"]]
df.drop("Lung Cancer", axis=1, inplace=True)

print(df.describe())


# Exploring correlations between features

plt.figure(figsize=(14,12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# High correlation for Couching of Blood


# EXERCISE 3. Modelling

# Data splitting into features (X) and target variable (y)
X = df.drop('Output', axis=1)
y = df['Output']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initializing and train a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)

print("KNN classifier accuracy :", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

# Initializing and train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluating the model on the test set
accuracy = model.score(X_test, y_test)
print("Model accuracy Logistic regression:", accuracy)

# We see how to train a machine learning logistic regression model using scikit-learn which gives better accuracy than KNN.

# Using trained model to make predictions on the test set
predictions = model.predict(X_test)

# Preview sample predictions and corresponding true labels
print("Sample predictions:")
for i in range(5):
    print("Predicted:", predictions[i], "| Actual:", y_test.iloc[i])

# Trained model was used to make predictions on the test set and comparison with the actual labels.


# EXERCISE 4. - OPTUNA Optimalisation of model

# Encoding target variable
label_encoder = LabelEncoder()
df['Level'] = label_encoder.fit_transform(df['Level'])

# Spliting data into features and target
X = df.drop(columns=['index', 'Patient Id', 'Level'])
y = df['Level']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna optimization
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)

    # Define the model with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Train the final model with the best hyperparameters
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)


# EXERCISE 5: Shap'ley values and evaluating model performance

import shap
from sklearn.metrics import classification_report, confusion_matrix

# Generating classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Calculating Shap'ley values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Interpret Shap'ley values
feature_importance = pd.DataFrame(list(zip(X_test.columns, np.mean(np.abs(shap_values), axis=0))),
                                  columns=['Feature', 'Importance'])
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(feature_importance)

# Evaluating the model's performance
y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

