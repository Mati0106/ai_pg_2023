import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import zscore
from sklearn.decomposition import PCA
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, classification_report
import shap

# Load the dataset
df = pd.read_csv('wine.csv')
print(df.head())
df.info()  # Confirm data is numeric

# Compute summary statistics
summary_stats = df.describe()
additional_stats = df.agg(['median', 'skew', 'kurtosis'])
additional_stats.index = ['median', 'skew', 'kurtosis']
combined_stats = pd.concat([summary_stats, additional_stats])
pd.set_option('display.max_columns', None)
print(combined_stats.T)

# Check and remove duplicates
if df.duplicated().any():
    print("\nThere are duplicates in this set.")
    df = df.drop_duplicates()
    print("\nDuplicates removed.")
else:
    print("\nThere are no duplicates in this set.")

# Check for NULLs
print("\nNumber of NULLs for each column:\n", df.isnull().sum())

# Checking outliers
features = df.columns.tolist()
plt.figure(figsize=(15, 11))
for i in range(len(features)):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x=df[features[i]], color='green', orient='h')
plt.suptitle('Boxplot Distribution of All Columns')
plt.show()

z_scores = zscore(df)
outliers = (z_scores > 3)
outliers_cat = outliers.sum(axis=0)
print("Sum of outliers (per category): ", outliers_cat)
total_outliers = outliers_cat.sum()
print("Total number of outliers:", total_outliers)

# Normality test
explan_var = df.columns[:-1]  # Excluding the target column
for var in explan_var:
    norm_test = stats.normaltest(df[var])
    if norm_test.pvalue > 0.05:
        print(f'p-Value of {var} is : {norm_test.pvalue} - Normally distributed')
    else:
        print(f'p-Value of {var} is : {norm_test.pvalue} - Not normally distributed')

# Correlation heatmap
plt.figure(figsize=(15, 9))
sns.heatmap(df.corr(method='spearman'), cmap="Spectral", annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Significance of features
corr_coeff = []
is_significant = []
for col in df.columns:
    c, p = stats.spearmanr(df[col], df['quality'])
    corr_coeff.append(c)
    is_significant.append('No' if p > 0.05 else 'Yes')
results = pd.DataFrame({"columns": df.columns, "correlation_coeff": corr_coeff, "is_significant": is_significant})
print(results)

# Feature Engineering - encoding
df['quality'] = df['quality'].apply(lambda x: 1 if x > 6.5 else 0)
print(df.head())

# Split the dataset
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=7)
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y:", Y.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Define models
def get_models():
    models = [
        ('LR', LogisticRegression()),
        ('DT', DecisionTreeClassifier()),
        ('SVC', SVC()),
        ('RC', RandomForestClassifier()),
        ('XGB', XGBClassifier())
    ]
    return models

cv = KFold(n_splits=11, shuffle=True, random_state=1)
models = get_models()

# Collect results from cross-validation
for name, model in models:
    results = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f'{name}: {results.mean()} ({results.std()})')

# Train and evaluate XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test)
plt.show()

# SHAP dependence plots
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()
for i, feature in enumerate(explan_var):
    shap.dependence_plot(feature, shap_values, X_test, feature_names=df.columns[:-1], ax=axes[i])
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Optuna for hyperparameter optimization
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 200)
    max_depth = int(trial.suggest_float('max_depth', 1, 32, log=True))
    clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return sklearn.model_selection.cross_val_score(clf, X, Y, n_jobs=-1, cv=3).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

trial = study.best_trial
print('Best Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

import plotly
optuna.visualization.plot_optimization_history(study)
