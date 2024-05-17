import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, classification_report
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

# Missing values
isnull_number = df.isnull().sum()
miss = pd.DataFrame(isnull_number, columns=["Total Missing Values"])
print(miss)

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

# 3. Model (benchmarkowy - średnia)
# Benchmarkowy model: przewidywanie większościowej klasy
y_pred_benchmark = np.full_like(y, fill_value=y.mode()[0])
accuracy_benchmark = accuracy_score(y, y_pred_benchmark)
print(f"Benchmark model accuracy: {accuracy_benchmark}")

# Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression accuracy: {accuracy_lr}")

# AUC-ROC for Logistic Regression
y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (area = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()
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

# 4. Optymalizacja za pomocą Optuny dla XGBoost
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

    model_xgb = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    return accuracy_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best parameters: {study.best_params}")

# 5. SHAP values + Partial Dependence Plot for XGBoost
model_xgb = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train, y_train)

explainer = shap.Explainer(model_xgb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=df.columns[:-1])

# Partial dependence plot
shap.dependence_plot(0, shap_values.values, X_test, feature_names=df.columns[:-1])

# 6. Analiza krzywej uczenia dla Logistic Regression
train_sizes, train_scores, test_scores = learning_curve(model_lr, X_scaled, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Learning curves for Logistic Regression')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# 7. Porównanie z innymi algorytmami



# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest accuracy: {accuracy_rf}")

# AUC-ROC for Random Forest
y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()

# Gradient Boosting Machine (GBM)
from sklearn.ensemble import GradientBoostingClassifier

model_gbm = GradientBoostingClassifier()
model_gbm.fit(X_train, y_train)
y_pred_gbm = model_gbm.predict(X_test)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
print(f"Gradient Boosting Machine accuracy: {accuracy_gbm}")

# AUC-ROC for Gradient Boosting Machine
y_pred_proba_gbm = model_gbm.predict_proba(X_test)[:, 1]
fpr_gbm, tpr_gbm, thresholds_gbm = roc_curve(y_test, y_pred_proba_gbm)
roc_auc_gbm = auc(fpr_gbm, tpr_gbm)

plt.plot(fpr_gbm, tpr_gbm, label=f'GBM (area = {roc_auc_gbm:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for GBM')
plt.legend(loc='lower right')
plt.show()

# Porównanie wyników różnych modeli
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (area = {roc_auc_lr:.2f})')

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (area = {roc_auc_rf:.2f})')
plt.plot(fpr_gbm, tpr_gbm, label=f'GBM (area = {roc_auc_gbm:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc='lower right')
plt.show()
