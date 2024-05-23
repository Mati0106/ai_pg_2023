from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import shap
import optuna
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
df = pd.read_csv('diabetes.csv')

# Deleting rows with only null values
df = df.dropna(how='all')
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
X = df.drop(['Outcome', 'Outlier_flag'], axis=1)
y = df['Outcome']

column_names = X.columns.tolist()
# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# No need for standarizing y as it has only 0 and 1 values
X_scaled = pd.DataFrame(X_scaled, columns=column_names)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Creating a very simple benchmark model for reference
benchmark_model = DummyClassifier(strategy="most_frequent")
benchmark_model.fit(X_train, y_train)
accuracy = benchmark_model.score(X_test, y_test)
print("Benchmark model accuracy:", accuracy)
predictions = benchmark_model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
cv_scores = cross_val_score(benchmark_model, X, y, cv=10)
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Creating a model and fitting with training data
# We would like to classify people as positive or negative in the context of having diabities so we will use a classificatior, Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluating the model on the test data
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Looking at confusion matrix will help evaluate the model deeper
predictions = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
# In the context of any disease detection the most important condition we should focus on is False negative.
# We can see that condition around 20 times for 154 observations, that is a lot

# Performing cross-validation
cv_scores = cross_val_score(model, X, y, cv=10)
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Defining objective function for optimalization
def objective(trial):
    # Choosing hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    # Creating a model with these hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42    )
    # Fitting with training data
    model.fit(X_train, y_train)

    # Evaluating the model on the test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Creating study Optuna object
study = optuna.create_study(direction='maximize')  # We want to maximize accuracy

# Optimalization process
study.optimize(objective, n_trials=100)
print("Best parameters:", study.best_params)
best_params = study.best_params
best_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42)
best_model.fit(X_train, y_train)

# Evaluating the model on the test data
accuracy = best_model.score(X_test, y_test)
print("Model accuracy:", accuracy)
# Looking at confusion matrix will help evaluate the model deeper
predictions = best_model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
# In the context of any disease detection the most important condition we should focus on is False negative
# We can see that now this condition happened around 10 times for 154 observations, that is still high but much better than before optimalization
# Accuracy doesn't differ that much
# Performing cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=10)
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values[0], X_test)
# We can see that Glucose and BMI have the biggest impact on the model
# On the other hand Age has the lowset impact
shap.dependence_plot('Glucose', shap_values[1], X_test,interaction_index='BMI')

