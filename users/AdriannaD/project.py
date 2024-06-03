import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as st
from scipy.stats import zscore, describe
from sklearn.decomposition import PCA
import warnings
import optuna
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# 1. Wczytanie danych + analiza
df = pd.read_csv("heart.csv")
print(df.head())
print(df.info())
print(df.describe())
#The Data Set consists of 303 Rows and 14 Columns.
#The type of all variables in the data set are in numerical format. (Integer or Float)
#There is no missing value (NaN Value) in the data set.


# Missing values
isnull_number = df.isnull().sum()
miss = pd.DataFrame(isnull_number, columns=["Total Missing Values"])
print(miss)
#There is no missing value in the data set.

# Analiza wartości odstających (box plots)
#for col in df.columns:
#    sns.boxplot(df[col])
#    plt.title(f"Boxplot for {col}")
#    plt.show()

# Korelacja
cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns, figsize=(10, 8), cmap="Reds")
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()
#The correlation matrix in the attached image illustrates the relationships between various medical variables and their association with the "output" variable. Key observations include:
#"cp" (chest pain type) and "thalachh" (maximum heart rate achieved) have the strongest positive correlations with "output" (0.43 and 0.42, respectively), indicating that higher values in these variables are associated with a higher output.
#"exng" (exercise-induced angina), "oldpeak" (ST depression induced by exercise), and "caa" (number of major vessels colored by fluoroscopy) exhibit the strongest negative correlations with "output" (-0.44, -0.43, and -0.39, respectively), suggesting that higher values in these variables are associated with a lower output.
#Age and sex have weaker correlations with output (-0.23 and -0.28, respectively).



# 2. Feature Engineering (PCA)
X = df.drop("output", axis=1)
y = df["output"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Heart Disease Dataset')
plt.show()

#PCA reduces the dataset to two principal components.
#Class Separation: Blue points (one class) are concentrated on the right. Red points (another class) are concentrated on the left.
#Overlap: Some overlap exists, suggesting the need for further classification methods for precise separation.


# 3. Model (benchmarkowy - średnia)
# Benchmarkowy model: przewidywanie większościowej klasy
y_pred_benchmark = np.full_like(y, fill_value=y.mode()[0])
accuracy_benchmark = accuracy_score(y, y_pred_benchmark)
print(f"Benchmark model accuracy: {accuracy_benchmark}")

# Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Logistic Regression accuracy: {accuracy}")

# AUC-ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'Logistic Regression (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#The ROC curve for the Logistic Regression model shows an AUC of 0.93, indicating excellent discriminative ability. This means the model is highly effective at distinguishing between positive and negative classes.


# 4. Model XGBoost
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost accuracy: {accuracy_xgb}")

# AUC-ROC for XGBoost
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_pred_proba_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (area = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for XGBoost')
plt.legend(loc='lower right')
plt.show()

# 4. Optymalizacja za pomocą Optuny
def objective(trial):
    param = {
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
    }

    model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best parameters: {study.best_params}")

# 5. Training XGBoost with best parameters from Optuna
model_optimized = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss')
model_optimized.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred_optimized = model_optimized.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"Optimized XGBoost accuracy: {accuracy_optimized}")

# AUC-ROC for Optimized XGBoost
y_pred_proba_optimized = model_optimized.predict_proba(X_test)[:, 1]
fpr_optimized, tpr_optimized, thresholds_optimized = roc_curve(y_test, y_pred_proba_optimized)
roc_auc_optimized = auc(fpr_optimized, tpr_optimized)

plt.plot(fpr_optimized, tpr_optimized, label=f'Optimized XGBoost (area = {roc_auc_optimized:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Optimized XGBoost')
plt.legend(loc='lower right')
plt.show()

#The optimized XGBoost model with an AUC of 0.93 performs better than the baseline model with an AUC of 0.91.
#The improvement in the AUC demonstrates that the optimization process has successfully enhanced the model's ability to discriminate between the positive and negative classes, albeit the improvement is marginal. This suggests that the optimized model is slightly more effective in prediction tasks.


# 5. SHAP values + partial dependence plot
model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=df.columns[:-1])

# Partial dependence plot
shap.dependence_plot(0, shap_values.values, X_test, feature_names=df.columns[:-1])



