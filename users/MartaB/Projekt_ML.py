import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import zscore, describe
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import xgboost as xgb
import shap
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, classification_report

#Our file provide informarion about wine quaity. As we determine wine quality on a scale from 3 to 8 (where 8 means best and 3 worst), we have a classification problem.
df = pd.read_csv('wine.csv')
print(df.head())

df.info()
#We can see we have all numeric data, no need to transform.

#Check for duplicates
if not df.duplicated().any():
    print("\nThere are no duplicates in this set.")
else:
    print("\nThere are duplicates in this set.")

#The dataset shows 240 duplicate values, they can affect model performance, hence will be removed.
df = df.drop_duplicates()
df.duplicated()

#Check again for duplicates
if not df.duplicated().any():
    print("\nThere are no duplicates in this set.")
else:
    print("\nThere are duplicates in this set.")

#No more duplicates in our dataset.

#Check for NULL's
print("\nNumber of NULL's for each column:\n", df.isnull().sum())

#No NULLs present in this set
df.info()

#Checking outliers
#Boxplot distribution
features = df.columns.tolist()
plt.figure(figsize=(15,11))
for i in range(0, len(features)):
  plt.subplot(3, 4, i+1)
  sns.boxplot(x=df[features[i]], color='green', orient='h')
plt.suptitle('Boxplot Distribution of All Columns')


z_scores = zscore(df)

# Identify outliers
outliers = (z_scores > 3)

# Sum outliers for each category (column)
outliers_cat = outliers.sum(axis=0)
print("Sum of outliers (per category): ", outliers_cat)

total_outliers = outliers_cat.sum()
print("Total number of outliers:", total_outliers)
print('A total of 147 outliers. Removing them will result in narrowing our dataset to 1212 observations, which will be around 76% of our initial set. Removing so many observations can impact future results, hence it was decided to not remove them.')

# Compute summary statistics using describe()
summary_stats = df.describe()

# Compute additional statistics: median, skewness, and kurtosis
additional_stats = df.agg(['median', 'skew', 'kurtosis'])

# Rename the index to match with the statistics
additional_stats.index = ['median', 'skew', 'kurtosis']

# Concatenate summary statistics and additional statistics into a single DataFrame
combined_stats = pd.concat([summary_stats, additional_stats])

# Set the display options to show all columns
pd.set_option('display.max_columns', None)

# Display the combined statistics
print(combined_stats.T)

#Statistics suggest that distribution of all variables is not normal. Let's check this further.

# Normality test
explan_var = ['fixed acidity',
  'volatile acidity',
  'citric acid',
  'residual sugar',
  'chlorides',
  'free sulfur dioxide',
  'total sulfur dioxide',
  'density',
  'pH',
  'sulphates',
  'alcohol']

for i in range(len(explan_var)):
  norm = stats.normaltest(np.array(df[explan_var[i]]))
  if norm.pvalue > 0.05:
    print('p-Value of', explan_var[i], 'is :', norm.pvalue, 'can be categorize as Normal')
  else:
    print('p-Value of', explan_var[i], 'is :', norm.pvalue, 'can be categorize as Not Normal')

#All variables distribution is not normal, hence we will calculate Spearman's correlation.

#Correlation heatmap
plt.figure(figsize=(15, 9))
sns.heatmap(df.corr(method='spearman'), cmap="Spectral", annot=True)
plt.title('Correlation Heatmap')
plt.show()
print('Correlation heatmap created')

# Heatmap shows that quality correlations with residual sugar, free sulfur dioxide and pH are near 0, but should we remove them from our model?
corr_coeff= []
is_significant = []

for col in df.columns:
    c, p = stats.spearmanr(df[col], df['quality'])
    corr_coeff.append(c)
    if p > 0.5:
        is_significant.append('No')
    else:
        is_significant.append('Yes')
new_f = {
    "columns": df.columns,
    "correlation_coeff": corr_coeff,
    "is_significant": is_significant
}
results = pd.DataFrame(new_f)
print(results)

# Checking the significance of all features shows that all of them are significant, which advocates for leaving all feauters in our model.

# Example 2: Data Preprocessing

#Feautre Engineering - encoding
#Upon reading the dataset desription, grade above 6.5 is conidered good and grade below 6.5 is considered bad, hence we will scale the data (good ->1 bad ->0)
df['quality'] = df['quality'].apply(lambda x: 1 if x > 6.5 else 0)
print(df.head())
quality_counts = df['quality'].value_counts()

# Display the counts
print("Counts of each class in 'quality' variable:")
print(quality_counts)
#We can see that our feature values are very imbalanced, with vast majority of bad wines. For imbalanced datasets, boost or tree-based models are usually a very good choice.

X = df.iloc[:,0:11]
Y = df.iloc[:,11]

seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y:", X.shape)
print("Shape of y_train:", X_train.shape)
print("Shape of y_test:", X_test.shape)

def get_models():
    models = []
    models.append(('DT',DecisionTreeClassifier()))
    models.append(('SVC',SVC()))
    models.append(('XGB',XGBClassifier()))
    return models

# define test conditions
cv = KFold(n_splits=11, shuffle=True, random_state=1)

# get the list of models to consider
models = get_models()

# collect results
names = []
cv_results = []

for name, model in models:
    results = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results.append(results)
    names.append(name)
    print('%s: %f (%f)' % (name, results.mean(), results.std()))

#Modeling
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)


# Make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: %.2f%%" % (accuracy * 100.0))

log_f1 = f1_score(y_test, y_pred)
log_acc = accuracy_score(y_test, y_pred)
log_recall = recall_score(y_test, y_pred)
log_auc = roc_auc_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


# Initialize the TreeExplainer with the XGBoost model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)
print(shap_values)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test)
plt.show()

#High alcohol content increases the likelihood that the wine will be good.
#High values of total sulfur dioxide decrease the likelihood that the wine will be good.
#There is a high chance that very low values of volatile acidity will classify the wine as good (lack of red dots on the right side, which means either we are not sure if it is good or it is good).

#Optuna
def objective(trial):

    n_estimators = trial.suggest_int('n_estimators', 2, 200)
    max_depth = int(trial.suggest_float('max_depth', 1, 32, log=True))

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth)

    return sklearn.model_selection.cross_val_score(
        clf, X_train, y_train, n_jobs=-1, cv=3).mean()



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
import plotly
optuna.visualization.plot_optimization_history(study)

# Partial dependence plot
shap.dependence_plot(0, shap_values, X_test, feature_names=df.columns[:-1])