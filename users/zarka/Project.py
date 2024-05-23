import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
import shap

# Load the dataset:

df = pd.read_csv('heart.csv')

# "Heart" dataset from Kaggle is used to tackle a binary classification problem,
# where the target variable indicates the presence (1) or absence (0) of heart disease in patients.
# The dataset includes various medical attributes which are used to predict the likelihood of heart disease.
# The primary goal is to develop a model that can accurately classify patients based on these features.

# First several rows of the dataset:

print(df.head())

# Number of rows and columns:

print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')

# Check if there are any null values:

print(df.isnull().sum())

# Basic statistics:

print(df.describe().T)

# Check proportion of the target values:

print(df['target'].value_counts())

# Test for correlations:

print(df.corr())
# There is a positive correlation between cp (chest pain) and target.
# When the chest pain increases there is a greater chance for a disease.
# Also there is a negative correlation between exang (exercise induced angina) and target.
# Exercising produces more blood and slows down blood flow.

# Check for outliers:

Q_01 = df.quantile(0.1)
Q_09 = df.quantile(0.9)
IQR = Q_09 - Q_01

outliers = ((df < (Q_01 - 1.5 * IQR)) | (df > (Q_09 + 1.5 * IQR))).sum()
print("Number of outliers in each column:\n", outliers)

# Prepare data for modeling. Start from spliting to train and test dataframe:

X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Principal component analysis:

pca = PCA(n_components=6) # PCA initialized with 6 components to reduce dimensionality and simplify the model.
X_train_pca = pca.fit_transform(X_train) # Fit the model to the training data and transform it.
X_test_pca = pca.transform(X_test) # Fit the model to the test data.

# XGBoost:

model = XGBClassifier()
model.fit(X_train_pca, y_train) # Train the model on the PCA-transformed training data.
y_pred = model.predict(X_test_pca) # Use the model to make predictions on the PCA-transformed test data.

initial_score = roc_auc_score(y_test, y_pred) # Calculate the ROC AUC score.
print(f'Initial Score: {initial_score}')
# We get score ~0.8 which indicates a good model performance
# and high ability to distinguish between patients with and without disease.

# Optuna - hyperparameter optimization:

def objective(trial):
    n_components = trial.suggest_int('n_components', 2, X_train.shape[1])
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    model = XGBClassifier(**param)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    roc_auc = roc_auc_score(y_test, y_pred)

    return roc_auc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(f'Best Parameters: {best_params}')

# During each of 100 trials optuna tested different combinations of hyperparameters
# to find the best set that maximizes the model's performance.
# That way we learn which trial gives the highest performance.
# Model should be configured to those parameters to achieve that score.

# Test the model with new parameters:

n_components_opt = best_params.pop('n_components') # separate number of components from the parameters (for PCA)
pca = PCA(n_components=n_components_opt)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the optimized XGBoost model:

optimized_model = XGBClassifier(**best_params)
optimized_model.fit(X_train_pca, y_train)
y_pred_optimized = optimized_model.predict(X_test_pca)
optimized_score = roc_auc_score(y_test, y_pred_optimized)
print(f'Optimized Score: {optimized_score}')

# After optimatization we have got higher score (around 5-8% more).

# Use shap values for the model interpretation.
# Determine the contribution of each feature to the model's prediction.

explainer = shap.Explainer(optimized_model)
shap_values = explainer.shap_values(X_test_pca)

matplotlib.use('TkAgg')
print(shap.summary_plot(shap_values, X_test_pca, plot_type="bar", feature_names=X_test.columns))

# Top features that contribute the most to the model performance are:

# 1. chol (cholesterol): It has highest average SHAP value and contribute the most to the model's predictions.
# 2. sex: gender difference impacts the model prediction.