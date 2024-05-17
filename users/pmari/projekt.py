# Classification
# From kaggle.com I have choosen data related to Heart disease.
# based on features: Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120, EKG results, Max HR, Exercise angina, ST depression, Slope of ST, Number of vessels fluro, Thallium
# will build a model to assess Heart disease (Presence/Absence)

# Loading a CSV file into a DataFrame

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 1. Data read and analysis
# Load a CSV file into a DataFrame
df = pd.read_csv('C:/AI/ML/Heart_Disease_Prediction.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Age - The age of the patient, which can be a significant risk factor for heart diseases. The older a person is, the higher the risk of developing heart diseases.
# Sex - The gender of the patient, which also can influence the risk of heart diseases. Men tend to have a higher risk than women, but the risk for women increases after menopause. 1 is male, 0 female
# Chest pain type - Description of pain or discomfort in the chest, which may be associated with heart diseases. It can be described as pressure, squeezing, burning, or tightness.
# BP - Blood pressure measurement, which is an important indicator of heart health. High blood pressure (hypertension) is a risk factor for heart diseases.
# Cholesterol - The level of cholesterol in the blood, which also can be a risk factor for heart diseases. High levels of "bad" cholesterol (LDL) and low levels of "good" cholesterol (HDL) increase the risk of heart diseases.
# FBS over 120 - High fasting blood sugar level, which may indicate diabetes or prediabetes, thus increasing the risk of heart diseases.
# EKG results - Results of an electrocardiogram, a diagnostic test assessing the electrical activity of the heart. Abnormalities in the EKG may indicate heart problems.
# Max HR - The maximum heart rate per minute during intense physical activity, which may be relevant for diagnosing heart diseases.
# Exercise angina - Chest pain or discomfort associated with physical exertion, which may be a symptom of heart diseases.
# ST depression - Depression of the ST segment on an electrocardiogram, which may be associated with heart ischemia.
# Slope of ST - The slope angle of the ST segment on an electrocardiogram, which may be an indicator of heart health.
# Number of vessels fluro - The number of coronary vessels observed during fluoroscopy, an imaging test showing blood flow through the coronary arteries. Evaluating coronary vessels is important for diagnosing heart diseases.
# Thallium - A chemical substance used in imaging tests (such as scintigraphy) to assess blood flow in the heart. It is used to detect issues related to coronary arteries.
# Heart Disease - A general term describing various heart conditions, including coronary artery disease, valvular diseases, arrhythmias, heart failure, etc.

# Display the column names and data types
print("Column Names and Data Types:")
print(df.dtypes)

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

puste=df.isnull().sum()
print("Ilość pustych wierszy:\n",puste)

# 2. Feature engineering
# Mapping 
mapping={"Absence":0, "Presence":1}

df["Output"]=[mapping[value] for value in df["Heart Disease"]]
df.drop("Heart Disease", axis=1, inplace=True)
print(df.describe())

# Explore correlations between features
plt.figure(figsize=(14,12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# High correlation for Max HR

# 3. Modeling
# Split the data into features (X) and target variable (y)
X = df.drop('Output', axis=1)
y = df['Output']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy KNN classifier:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

# Initialize and train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print("Model accuracy Logistic regression:", accuracy)
# This example illustrates how to train a simple machine learning model (logistic regression) using scikit-learn.
# Logistic regression gives better accuracy then KNN

# Use the trained model to make predictions on the test set
predictions = model.predict(X_test)

# Display some sample predictions and corresponding true labels
print("Sample predictions:")
for i in range(5):
    print("Predicted:", predictions[i], "| Actual:", y_test.iloc[i])
# Here, we use the trained model to make predictions on the test set and compare them with the actual labels.

# Example 5: Evaluating Model Performance

from sklearn.metrics import classification_report, confusion_matrix

# Generate a classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
# Finally, this example demonstrates how to evaluate the performance of the trained model using metrics such as classification report and confusion matrix.

import shap
import sklearn

X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
print("Model coefficients:\n")
for i in range(X.shape[1]):
    print(X.columns[i], "=", model.coef_[i].round(5))

# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]

# compute the SHAP values for the linear model
explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X)
sample_ind = 20
shap.plots.waterfall(shap_values[sample_ind], max_display=14)

import interpret.glassbox
# import matplotlib.pyplot as plt
model_ebm = interpret.glassbox.ExplainableBoostingRegressor(interactions=0)
model_ebm.fit(X, y)
explainer_ebm = shap.Explainer(model_ebm.predict, X100)
shap_values_ebm = explainer_ebm(X)
shap.plots.scatter(shap_values_ebm[:, "Number of vessels fluro"])
shap.plots.beeswarm(shap_values_ebm)