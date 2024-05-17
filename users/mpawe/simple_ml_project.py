import pandas as pd
from pandas_profiling import ProfileReport

data = pd.read_csv('C:/Users/pmord/PodyplomoweAI/ML/archive/student-por.csv')

print(data)
print(data.describe())
print(data.info())

# No matrix correlation because string

# Dummy variables because there is some categorical variables
df = pd.get_dummies(data)

print("Data with dummies:",  df)

# Choosing target and features
target='G3'
X = df.drop(target, axis=1)
Y = df[target]

print(X)
print(Y)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBRegressor()
model.fit(X_train, y_train)

print(model)

# Make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# Benchmarking
print("\n======== Benchmarking ===========\n")
print("Prediction: ", predictions)

# Evaluate predictions using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("MSE: {0}".format(mse))

# Evaluate predictions using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: {0}".format(rmse))

# Evaluate predictions using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print("MAE: {0}".format(mae))

# Evaluate predictions using R-squared
r2 = r2_score(y_test, predictions)
print("R-squared: {0}".format(r2))

# Evaluate predictions using Adjusted R-squared
n = len(y_test)  # Number of samples
p = X_test.shape[1]  # Number of predictors

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R-squared: {0}".format(adj_r2))

# Define the Huber loss function
def huber_loss(y_true, y_pred, delta=1.0):
    residual = y_true - y_pred
    condition = np.abs(residual) <= delta
    squared_loss = 0.5 * residual ** 2
    linear_loss = delta * (np.abs(residual) - 0.5 * delta)
    return np.where(condition, squared_loss, linear_loss).mean()


# Evaluate predictions using Huber Loss
huber = huber_loss(y_test, y_pred, delta=1.0)
print("Huber Loss function: {0}".format(huber))

# Using optuna
import optuna


def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': trial.suggest_int('random_state', 1, 1000)
    }
    model = XGBRegressor(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)


# Create the study
study = optuna.create_study(direction='minimize', study_name='regression')
study.optimize(objective, n_trials=100)

# Print the best parameters
print('Best parameters', study.best_params)
# Print the best value
print('Best value', study.best_value)

# Print the best trial
print('Best trial', study.best_trial)

# Train new model

model = XGBRegressor(**study.best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("\n========= New model parameters ===========\n")
print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# Shap values
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
