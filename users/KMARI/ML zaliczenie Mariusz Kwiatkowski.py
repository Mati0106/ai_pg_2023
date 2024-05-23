# Import libraries:
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import optuna
import shap
import matplotlib.pyplot as plt

# 1. Download And Analysis Data

# Import Youtube Sports Channel As DF

df = pd.read_csv('yt_sports_channels_stats.csv')
# Display the column names and data types
print("Column Names and Data Types:")
print(df.dtypes)

# Remove "channel_id" column
NumericData = df.drop(columns=['channel_id', 'channel_title', 'start_date'])
print(NumericData)
print(NumericData.dtypes)

# 2 Principal Component Analysis - PCA
# Step 1: Perform PCA to reduce to 3 dimensions
pca = PCA(n_components=2)
pca_data = pca.fit_transform(NumericData)

# Convert PCA results back to a DataFrame
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])

print(pca_df)
# Prepare data for modeling
X = pca_df
y = df['view_count']  # Assuming 'view_count' is the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Create a benchmark model
benchmark_model = RandomForestRegressor(n_estimators=100, random_state=42)
benchmark_model.fit(X_train, y_train)
benchmark_predictions = benchmark_model.predict(X_test)
benchmark_mse = mean_squared_error(y_test, benchmark_predictions)
print(f'Benchmark Model MSE: {benchmark_mse}')

# Step 3: Use Optuna to optimize the model


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 4, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse


study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler())
study.optimize(objective, n_trials=50)

print('Best parameters:', study.best_params)

# Train the optimized model# Import libraries:

optimized_model = RandomForestRegressor(**study.best_params, random_state=42)
optimized_model.fit(X_train, y_train)
optimized_predictions = optimized_model.predict(X_test)
optimized_mse = mean_squared_error(y_test, optimized_predictions)
print(f'Optimized Model MSE: {optimized_mse}')

# Step 4: Use SHAP to create a partial dependence plot
explainer = shap.TreeExplainer(optimized_model)
shap_values = explainer.shap_values(X_test)

# Create the SHAP summary plot
shap.summary_plot(shap_values, X_test)

# Plot partial dependence for one of the principal components
shap.dependence_plot('PC1', shap_values, X_test)
plt.show()
