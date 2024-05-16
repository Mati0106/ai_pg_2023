import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy
from scipy.stats import chi2_contingency
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from statistics import stdev
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
sns.set_context("notebook")

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load csv file

hr_data = pd.read_csv('.\\users\\mmare\\HR.csv')

# Shape of the DataFrame / how many rows and columns do we have
print("\nDataFrame Shape:", hr_data.shape)

# basic information about the data / column names, data types, count
hr_data.info()

#check if there are some missing data

hr_data.describe().transpose().round(2)
hr_data.describe(include="all")
# Get the list of categorical columns
cat_cols = hr_data.select_dtypes(include='object').columns.tolist()
print(cat_cols)
# Create a DataFrame containing counts of unique values for each categorical column
cat_hr_data = pd.DataFrame(hr_data[cat_cols].melt(var_name='column', value_name='value')
                      .value_counts()).rename(columns={0: 'count'}).sort_values(by=['column', 'count'])
print(cat_hr_data)

# Display summary statistics of categorical variables

display(hr_data[cat_cols].describe())

# Display counts of unique values for each categorical column
display(cat_hr_data)
#duplicated records
hr_data[hr_data.duplicated(keep=False)]
#missing values
missing_hr_data = hr_data.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing Values"})
missing_hr_data["% of Missing Values"] = round((missing_hr_data["Total No. of Missing Values"]/len(hr_data))*100,2)
missing_hr_data

print("\nMissing data:")
hr_data.isna().sum()
# Conclusion : The data has no missing values, but one column has a naming error and some entries start with a capital letter.

# display summary statistics
print("\nSummary Statistics:")
print(hr_data.describe())

# Rename columns
hr_data = hr_data.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})

hr_data.columns

#my TRIAL

# Calculate skewness for numerical columns
skewness = hr_data.select_dtypes(include=['int64', 'float64']).skew()
skewness
# Count the number of numerical columns
num_cols_count = len(hr_data.select_dtypes(include=['int64', 'float64']).columns)
num_cols_count

# Determine the layout for subplots
num_rows = (num_cols_count + 3) // 4  # Adjust the number of columns in each row
num_cols = min(4, num_cols_count)  # Maximum of 4 columns in each row
# Plot histograms for numerical columns to visualize distributions and identify anomalies
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
for i in range(num_rows):
    for j in range(num_cols):
        col_idx = i * num_cols + j
        if col_idx < num_cols_count:
            col = hr_data.select_dtypes(include=['int64', 'float64']).columns[col_idx]
            axes[i, j].hist(hr_data[col], bins=15, color='skyblue', alpha=0.7)
            axes[i, j].set_title(f'{col}')
            axes[i, j].set_xlabel(col)
            axes[i, j].set_ylabel('Frequency')

            # Compute skewness
            skew_val = skewness[col]

            # Plot skewness value in the center of plot
            axes[i, j].text(0.5, 0.5, f'Skewness: {skew_val:.2f}', horizontalalignment='center',
                            verticalalignment='center', transform=axes[i, j].transAxes, fontsize=10, color='red')

plt.tight_layout()
plt.show()

# Print skewness values
print("Skewness:")
print(skewness)

# Plot the boxplot with rotated text labels
hr_data.plot(kind='box', rot=45)

# Show the plot
plt.show()

# Correlation matrix

# Select only the numeric columns from the DataFrame
numeric_columns = hr_data.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Heatmap Plotting
# Select only numeric columns
numeric_columns = hr_data.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix
corr_matrix = numeric_columns.corr()

# Filter correlation matrix to include values greater than 0.5 or less than -0.5
corr_matrix_filtered = corr_matrix[(corr_matrix > 0.5) | (corr_matrix < -0.5)]

# Plot the heatmap with filtered correlation values
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_filtered, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features (|Correlation| > 0.5)')
plt.show()

# Explore categorical features
for column in hr_data.select_dtypes(include=['object']):
    sns.countplot(x=column, data=hr_data)
    plt.tight_layout()
    plt.show()

# Explore categorical features
class XGBClassifier:
    pass


for column in hr_data.select_dtypes(include=['object']):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=column, data=hr_data)

    # Add count and percentage annotations to each bar
    total = len(hr_data[column])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        count = p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(f'{count}\n{percentage}', (x, y), ha='center', va='bottom')

    plt.title(f'Count Plot for {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Get the value counts for the 'Attrition' column
    value_counts = hr_data['Attrition'].value_counts()

    # Calculate the percentage
    percentage = (value_counts / len(hr_data)) * 100

    # Concatenate the count and percentage into a DataFrame
    count_and_percentage = pd.concat([value_counts, percentage], axis=1)
    count_and_percentage.columns = ['Count', 'Percentage']

    # Round the values in the DataFrame to two decimal places
    count_and_percentage_rounded = count_and_percentage.round(2)

    # Calculate the total count
    total_count = len(hr_data)

    # Create a DataFrame for total count
    total_employee_data = pd.DataFrame({'Count': [total_count], 'Percentage': [100]}, index=['Total'])

    # Concatenate the total count DataFrame with the counts and percentages DataFrame
    result_employee_data = pd.concat([count_and_percentage_rounded, total_employee_data])

    # Display the result DataFrame
    print(result_employee_data)

    # Get the value counts for the 'Attrition' column
    value_counts = hr_data['Attrition'].value_counts()

    # Calculate the percentage
    percentage = (value_counts / len(hr_data)) * 100

    # Concatenate the count and percentage into a DataFrame
    count_and_percentage = pd.concat([value_counts, percentage], axis=1)
    count_and_percentage.columns = ['Count', 'Percentage']

    # Round the values in the DataFrame to two decimal places
    count_and_percentage_rounded = count_and_percentage.round(2)

    # Calculate the total count
    total_count = len(hr_data)

    # Create a DataFrame for total count
    total_employee_data = pd.DataFrame({'Count': [total_count], 'Percentage': [100]}, index=['Total'])

    # Concatenate the total count DataFrame with the counts and percentages DataFrame
    result_employee_data = pd.concat([count_and_percentage_rounded, total_employee_data])

    # Display the result DataFrame
    print(result_employee_data)

#Employee attrition by gender
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    att1 = hr_data.groupby(by='Attrition', as_index=False)['Age'].count()
    att1['Count'] = att1['Age']
    att1.drop('Age', axis=1, inplace=True)
    att2 = hr_data.groupby(['Gender', 'Attrition'], as_index=False)['Age'].count()
    att2['Count'] = att2['Age']
    att2.drop('Age', axis=1, inplace=True)
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=3)
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=('<b>Employee Attrition', '<b>Female Attrition', '<b>Male Attrition'))

    fig.add_trace(
        o.Pie(values=att1['Count'], labels=att1['Attrition'], hole=0.7,marker_colors=['DeepSkyBlue', 'LightCoral'],
               name='Employee Attrition', showlegend=False), row=1, col=1)
    fig.add_trace(go.Pie(values=att2[(att2['Gender'] == 'Female')]['Count'],
                         labels=att2[(att2['Gender'] == 'Female')]['Attrition'], hole=0.7,
                         marker_colors=['DeepSkyBlue','LightCoral'], name='Female Attrition', showlegend=False), row=1,
                  col=2)
    fig.add_trace(
        go.Pie(values=att2[(att2['Gender'] == 'Male')]['Count'], labels=att2[(att2['Gender'] == 'Male')]['Attrition'],
               hole=0.7,marker_colors=['DeepSkyBlue','LightCoral'], name='Male Attrition', showlegend=True), row=1,
        col=3)
    fig.update_layout(title_x=0, template='simple_white', showlegend=True,
                      legend_title_text="<b style=\"font-size:90%;\">Attrition",
                      title_text='<b style="color:black; font-size:120%;"></b>', font_family="Times New Roman",
                      title_font_family="Times New Roman")
    fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))

#LABEL ENCODING
    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    hr_data["Attrition"] = label.fit_transform(hr_data.Attrition)
    hr_data.info()


    #Data processing

    dummy_col = [column for column in hr_data.drop('Attrition', axis=1).columns if
                 hr_data[column].nunique() < 20]
    data = pd.get_dummies(hr_data, columns=dummy_col, drop_first=True, dtype='uint8')
    data.info()
    print(data.shape)

    # Remove duplicate Features
    data = data.T.drop_duplicates()
    data = data.T

    # Remove Duplicate Rows
    data.drop_duplicates(inplace=True)

    print(data.shape)



    data.drop('Attrition', axis=1).corrwith(data.Attrition).sort_values().plot(kind='barh', figsize=(10, 75))
 #   data.drop('Attrition', axis=1).corrwith(data.Attrition).sort_values().plot(kind='barh', figsize=(10, 30))

#Attrition based on Perfomance Rating
    bus = hr_data.groupby(['PerformanceRating', 'Attrition'], as_index=False)['Age'].count()
    bus.rename(columns={'Age': 'Count'}, inplace=True)
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
    fig.add_trace(go.Pie(values=bus[bus['PerformanceRating'] == 'Excellent']['Count'],
                         labels=bus[bus['PerformanceRating'] == 'Excellent']['Attrition'], hole=0.7,
                         marker_colors=['DeepSkyBlue', 'LightCoral'], name='Excellent', showlegend=False), row=1, col=1)
    fig.add_trace(go.Pie(values=bus[bus['PerformanceRating'] == 'Outstanding']['Count'],
                         labels=bus[bus['PerformanceRating'] == 'Outstanding']['Attrition'], hole=0.7,
                         marker_colors=['DeepSkyBlue', 'LightCoral'], name='Outstanding', showlegend=True), row=1,
                  col=2)
    fig.add_annotation(dict(x=0.18, y=0.5, align='center',
                            xref="paper", yref="paper",
                            showarrow=False, font_size=15,
                            text="<b>Excellent</b>"))
    fig.add_annotation(dict(x=0.83, y=0.5, align='center',
                            xref="paper", yref="paper",
                            showarrow=False, font_size=15,
                            text="<b>Outstanding</b>"))
    fig.update_traces(textposition='outside', textinfo='percent+label')
    fig.update_layout(title_x=0.5, template='simple_white', showlegend=True, legend_title_text="Attrition",
                      title_text='<b style="color:black; font-size:100%;">Employee Attrition bd on Performance Rating',
                      font_family="Times New Roman", title_font_family="Times New Roman")
    fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
    feature_correlation = data.drop('Attrition', axis=1).corrwith(data.Attrition).sort_values()
    model_col = feature_correlation[np.abs(feature_correlation) > 0.02].index
    len(model_col)


    # Train and data test
    X = data.drop('Attrition', axis=1)
    y = data.Attrition

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                        stratify=y)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    X_std = scaler.transform(X)
 #   (X_test)
  #  X_std = scaler.transform(X)X = hr_data.drop('Attrition', axis=1)
 #   y = hr_data.Attrition

 #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                        stratify=y)


    def feature_imp(df, model):
        fi = pd.DataFrame()
        fi["feature"] = df.columns
        fi["importance"] = model.feature_importances_
        return fi.sort_values(by="importance", ascending=False)


    y_test.value_counts()[0] / y_test.shape[0]

    stay = (y_train.value_counts()[0] / y_train.shape)[0]
    leave = (y_train.value_counts()[1] / y_train.shape)[0]

    print("===============TRAIN=================")
    print(f"Staying Rate: {stay * 100:.2f}%")
    print(f"Leaving Rate: {leave * 100 :.2f}%")

    stay = (y_test.value_counts()[0] / y_test.shape)[0]
    leave = (y_test.value_counts()[1] / y_test.shape)[0]

    print("===============TEST=================")
    print(f"Staying Rate: {stay * 100:.2f}%")
    print(f"Leaving Rate: {leave * 100 :.2f}%")


    def evaluate(model, X_train, X_test, y_train, y_test):
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        print("TRAINIG RESULTS: \n===============================")
        clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
        print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
        print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")

        print("TESTING RESULTS: \n===============================")
        clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
        print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
        print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print(clf_report)

    #logistic regression



    lr_clf = LogisticRegression(solver='liblinear', penalty='l1')
    lr_clf.fit(X_train_std, y_train)

    evaluate(lr_clf, X_train_std, X_test_std, y_train, y_test)


    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.title("Precision/Recall Tradeoff")


    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], "k--")
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')


    precisions, recalls, thresholds = precision_recall_curve(y_test, lr_clf.predict(X_test_std))
    plt.figure(figsize=(14, 25))
    plt.subplot(4, 2, 1)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    plt.subplot(4, 2, 2)
    plt.plot(precisions, recalls)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("PR Curve: precisions/recalls tradeoff");

    plt.subplot(4, 2, 3)
    fpr, tpr, thresholds = roc_curve(y_test, lr_clf.predict(X_test_std))
    plot_roc_curve(fpr, tpr)

#tested OK 20240515-13:42

    scores_dict = {
        'Logistic Regression': {
            'Train': roc_auc_score(y_train, lr_clf.predict(X_train)),
            'Test': roc_auc_score(y_test, lr_clf.predict(X_test)),
        },
    }
    # Get feature coefficients from the logistic regression model
    feature_importance = np.abs(lr_clf.coef_[0])

    # Create DataFrame for feature importance
    df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})

    # Sort DataFrame by importance in descending order
    df = df.sort_values(by='importance', ascending=False)

    # Select top 40 features
    df = df[:40]

    # Plotting top 40 feature importance
    plt.figure(figsize=(10, 10))
    df.set_index('feature').plot(kind='barh', figsize=(10, 10))
    plt.title('Feature Importance according to Logistic Regression')
    plt.xlabel('Importance')
    plt.show()

    #RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    rf_clf = RandomForestClassifier(n_estimators=100, bootstrap=False,
                                    #                                      class_weight={0:stay, 1:leave}
                                    )
    rf_clf.fit(X_train, y_train)
    evaluate(rf_clf, X_train, X_test, y_train, y_test)
    param_grid = dict(
        n_estimators=[100, 500, 900],
        max_features=['auto', 'sqrt'],
        max_depth=[2, 3, 5, 10, 15, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        bootstrap=[True, False]
    )

    rf_clf = RandomForestClassifier(random_state=42)
    search = GridSearchCV(rf_clf, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)

    rf_clf = RandomForestClassifier(**search.best_params_, random_state=42)
    rf_clf.fit(X_train, y_train)
    evaluate(rf_clf, X_train, X_test, y_train, y_test)
    #a long long time

    precisions, recalls, thresholds = precision_recall_curve(y_test, rf_clf.predict(X_test))
    plt.figure(figsize=(14, 25))
    plt.subplot(4, 2, 1)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    plt.subplot(4, 2, 2)
    plt.plot(precisions, recalls)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("PR Curve: precisions/recalls tradeoff");

    plt.subplot(4, 2, 3)
    fpr, tpr, thresholds = roc_curve(y_test, rf_clf.predict(X_test))
    plot_roc_curve(fpr, tpr)

    scores_dict['Random Forest'] = {
        'Train': roc_auc_score(y_train, rf_clf.predict(X_train)),
        'Test': roc_auc_score(y_test, rf_clf.predict(X_test)),
    }

    df = feature_imp(X, rf_clf)[:40]
    df.set_index('feature', inplace=True)
    df.plot(kind='barh', figsize=(10, 10))
    plt.title('Feature Importance according to Random Forest')



    #Really what is the xgboost?
    from xgboost import XGBClassifier
    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)

    evaluate(xgb_clf, X_train, X_test, y_train, y_test)

    scores_dict['XGBoost'] = {
        'Train': roc_auc_score(y_train, xgb_clf.predict(X_train)),
        'Test': roc_auc_score(y_test, xgb_clf.predict(X_test)),
    }

    precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_clf.predict(X_test))
    plt.figure(figsize=(14, 25))
    plt.subplot(4, 2, 1)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    plt.subplot(4, 2, 2)
    plt.plot(precisions, recalls)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("PR Curve: precisions/recalls tradeoff");

    plt.subplot(4, 2, 3)
    fpr, tpr, thresholds = roc_curve(y_test, xgb_clf.predict(X_test))
    plot_roc_curve(fpr, tpr)

    df = feature_imp(X, xgb_clf)[:35]
    df.set_index('feature', inplace=True)
    df.plot(kind='barh', figsize=(10, 8))
    plt.title('Feature Importance according to XGBoost')

    

#    X = hr_data.iloc[:, 0:8]
#   Y = hr_data.iloc[:, 8]
#    seed = 7
#    test_size = 0.33
#    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#    model = XGBClassifier()
#    model.fit(X_train, y_train)

    print(model)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

