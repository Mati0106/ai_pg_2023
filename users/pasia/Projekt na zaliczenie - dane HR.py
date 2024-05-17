import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#1. Data exploration

#Load a CSV file into a DataFrame from file "pasia"
hr_data = pd.read_csv('./users/pasia/HR_data.csv')

# Display the shape of the DataFrame / how many rows and columns do we have
print("\nDataFrame Shape:", hr_data.shape)

# show basic information about the data / column names, data types, count
hr_data.info()

#check if there are some missing data
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

#correlation map

plt.figure(figsize=(8, 6))
sns.heatmap(hr_data[['satisfaction_level', 'left', 'number_project', 'average_monthly_hours', 'tenure']]
            .corr(), annot=True, cmap="crest")
plt.title('Mapa korelacji')
plt.show()

# diagram - number of employees vs tenure
plt.figure(figsize=(6, 6))
plt.title('Distribution of Tenure', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
sns.countplot(x='tenure', data=hr_data)
plt.show()
# Conclusion: There is too little data for tenures equal to 8, 9, and 10. I think I should consider them outliers

#calculate persentile to identify outliers and check how much rows contains outliers
p25 = hr_data['tenure'].quantile(0.25)
p75 = hr_data['tenure'].quantile(0.75)

# calculate  IQR to see how spread out the middle half of data (without getting fooled by extreme values)
iqr = p75 - p25

# count upper limit and lower limit for non-outlier values in `tenure`
upper_limit = p75 + 1.5 * iqr
lower_limit = p25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

wartosci_odstajace = hr_data[(hr_data['tenure'] > upper_limit) | (hr_data['tenure'] < lower_limit)] # Identify subset of data containing outliers in `tenure`

print("Liczba wierszy z wartościami odstającymi:", len(wartosci_odstajace))

###

# check how many employees works same amount of years
print(hr_data['tenure'].value_counts())
print()

# check how many people work in each department
print(hr_data['department'].value_counts())
print()

# check how many left
print(hr_data['left'].value_counts())
print()

#group data and count how many people left from each department
odejscia_per_dzial = hr_data[hr_data["left"] == 1].groupby("department").size().reset_index(name="Liczba odejść")

# visualize data
plt.bar(odejscia_per_dzial["department"], odejscia_per_dzial["Liczba odejść"])

#show amount for each department
for index, row in odejscia_per_dzial.iterrows():
    plt.text(index, row["Liczba odejść"] + 0.1, row["Liczba odejść"], ha='center', va='bottom', fontsize=10)

plt.xlabel("Nazwa działu")
plt.ylabel("Liczba odejść")
plt.title("Ilość odejść per dział")
plt.xticks(rotation=20, ha="right")
plt.show()
# Conclusion : The sales, technical, and support departments have the highest number of departures.

#Visualisation of data : how many people stay and left from each department
wszyscy_per_dzial = hr_data.groupby("department").size().reset_index(name="Liczba pracowników")
liczba_odjsc_per_dzial = hr_data[hr_data["left"] == 1].groupby("department").size().reset_index(name="Liczba odejść")
#create one dataframe from above data
df = wszyscy_per_dzial.merge(liczba_odjsc_per_dzial, on="department", how="left").fillna(0)
#determine the ratio
df["Stosunek"] = df["Liczba pracowników"] / df["Liczba odejść"]
#create diagram
plt.plot(df["department"], df["Stosunek"], marker='o', linestyle='-')

plt.xlabel("Nazwa działu")
plt.ylabel("Stosunek ilości pracowników do odejść")
plt.title("Stosunek pracowników do odejść w działach")
plt.xticks(rotation=20, ha="right")
plt.show()
#Conclusion: The higher the ratio, the fewer people left. The management and R&D departments have the fewest departures.


#show how many projects are in each department
liczba_projektow = hr_data.groupby("department")["number_project"].sum().reset_index()
# visualize data
plt.bar(liczba_projektow["department"], liczba_projektow["number_project"])
plt.xlabel("Nazwa działu")
plt.ylabel("Liczba projektów")
plt.title("Liczba projektów per dział")
plt.xticks(rotation=20, ha="right")
plt.show()
# Conclusion : Our data shows that the Sales department has the highest number of projects, while the Management department has the least.

print(hr_data['salary›'].value_counts())
print()

#show average satisfaction by salary category
avg_satisfaction = hr_data.groupby(['department', 'salary›'])['satisfaction_level'].mean().unstack()

plt.figure(figsize=(15, 10))
avg_satisfaction.plot(kind='bar', colormap='tab10')
plt.xlabel('Salary')
plt.ylabel('Average Satisfaction')
plt.title('Poziom satysfakcji vs poziom wynagrodzenia per departament')
plt.xticks(rotation=35)
plt.legend(title='Department', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()




#2. Logistic Regression Model

#make a copy of hr_data to save original data
hr_data_1 = hr_data.copy()

#change data in column "salary" into numerical codes
hr_data_1['salary›'] = (
    hr_data_1['salary›'].astype('category')  #astype -> change type of data into categorical type
    .cat.set_categories(['low', 'medium', 'high']) #define specific categories
    .cat.codes #assign a unique integer code to each category, for example: low = 0
)

#create separate column for each department with valuse true/false
hr_data_1 = pd.get_dummies(hr_data_1, drop_first=False)

hr_data_1.head()
hr_data_1.info() #now we have 18columns : separate for each department and changed type of data in column "salary"


# Create a heatmap to visualize correlaction
plt.figure(figsize=(8, 6))
sns.heatmap(hr_data_1[['satisfaction_level', 'left', 'number_project', 'average_monthly_hours', 'tenure']]
            .corr(), annot=True, cmap="crest")
plt.title('Mapa korelacji nowego df')
plt.show()

#visualisation emplyees stayed/left with old df
pd.crosstab(hr_data['department'], hr_data['left']).plot(kind ='bar',color='mr')
plt.title('Liczba pracowników vs ci którzy odeszli per dział')
plt.ylabel('Liczba pracowników')
plt.xlabel('Departament')
plt.show()

#remove the outliers from the tenure column
# Select rows without outliers in `tenure` and save the as new dataframe
hr_data_clean = hr_data_1[(hr_data_1['tenure'] >= lower_limit) & (hr_data_1['tenure'] <= upper_limit)]

#check new df
hr_data_clean.head()
hr_data_clean.columns

# Isolating the outcome variable, which is the variable we want our model to predict
y = hr_data_clean['left']
y.head()

#Select features to use in a model which will help to predict the outcome variable
X = hr_data_clean.drop('left', axis=1)
X.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Construct a logistic regression model and fit it to the training dataset
model = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)

#testing the logistic regression model
y_pred = model.predict(X_test)

#visualization of  results of the logistic regression model
log_cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,   #confusion matrix
                                  display_labels=model.classes_)

log_disp.plot(values_format='')
plt.show()
#Conslusion: my model is very good because he predicted 2344(true negatives) and 466(true positives) records correctly. He made mistakes in 245(false positives) and 375(false negatives) record.
