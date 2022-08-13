# May Sholomit portnoy - 208417964
# Shahar shtokhamer- 318162112
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv("votersdata.csv")

# Q1
RSEED = 123
# print(df.head())
# Q2.a
my_crosstab_status = pd.crosstab(df["status"], df["vote"])
my_crosstab_status.plot.bar(stacked=True, title="status")

my_crosstab_passtime = pd.crosstab(df["passtime"], df["vote"])
my_crosstab_passtime.plot.bar(stacked=True, title="passtime")

my_crosstab_sex = pd.crosstab(df["sex"], df["vote"])
my_crosstab_sex.plot.bar(stacked=True, title="sex")

# Q2.b
df.boxplot(column=["age"], by="vote", grid=False)
df.boxplot(column=["salary"], by="vote", grid=False)
df.boxplot(column=["volunteering"], by="vote", grid=False)
# plt.show()

# Q3
# print(df.isnull().sum())
# Age- We assume that the minimum age of vote is 16,So we changed the age under 16 and the missing values to the mean .
df["age"] = df["age"].mask(df["age"] < 16, np.nan)
df["age"] = df["age"].replace(to_replace=np.nan, value=df.age.mean())
# print(df.isnull().sum())
# salary column
# we replaced the extreme value to null, and changed the missing values with the average number
extreme_val = df['salary'].max()
# print(df.salary.describe())

df['salary'] = df['salary'].mask(df['salary'] == extreme_val, np.NaN)
df['salary'] = df['salary'].replace(to_replace=np.nan, value=df.salary.mean())
# print(df.salary.describe())
# checking if the last null value was change:
# print(df.salary.tail(10))

# Passtime- We replaced the missing value to the most common value.
# print(df["passtime"].value_counts())
df["passtime"] = df["passtime"].replace(to_replace=np.nan, value="fishing")
# print(df.isnull().sum())

# Creating a new dataframe for normlize
dfnum = df.copy(deep=True)
# encode the categorical variables
le = LabelEncoder()
le.fit(dfnum['sex'])
dfnum['new_sex'] = le.transform(df['sex'])

le.fit(dfnum['passtime'])
dfnum['new_passtime'] = le.transform(df['passtime'])

le.fit(dfnum['status'])
dfnum['new_status'] = le.transform(df['status'])

# Normalize
zscores = stats.zscore(df["salary"])
dfnum["salary"] = zscores
zscores = stats.zscore(df["volunteering"])
dfnum["volunteering"] = zscores
zscores = stats.zscore(df["age"])
dfnum["age"] = zscores
zscores = stats.zscore(dfnum["new_sex"])
dfnum["new_sex"] = zscores
zscores = stats.zscore(dfnum["new_passtime"])
dfnum["new_passtime"] = zscores
zscores = stats.zscore(dfnum["new_status"])
dfnum["new_status"] = zscores
print(dfnum.head())

#print(df.head())

# Q4
dfnum = dfnum.drop(["passtime", "sex"], axis=1)
x = dfnum.drop(["vote", "status"], axis=1)
y = dfnum["vote"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=RSEED)

# Q5
clf = DecisionTreeClassifier( random_state=RSEED)
clf = clf.fit(X_train, y_train)
# making the tree
plt.figure(figsize=(14, 10), dpi=200)
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=["Democrat","Republican" ])
# plt.show()

# Q6
y_test_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
# Accuracy
ac = pd.crosstab(y_test, y_test_pred, colnames=["pred"], margins=True)
print(ac)
tp=ac.iloc[0,0]
fp=ac.iloc[0,1]
fn=ac.iloc[1,0]
tn=ac.iloc[1,1]
Accuracy=(tp+tn)/(tp+fp+fn+tn)
# Recall
Recall = tp/(tp+fn)
# Precision
Precision= tp/(tp+fp)
print("test set:")
print(" Accuracy:", Accuracy)
print("Recall:",Recall)
print("Precision:",Precision)

# Q7
# checking the train set
y_train_pred = clf.predict(X_train)
ac = pd.crosstab(y_train, y_train_pred , colnames=["pred"], margins=True)
print(ac)
tp=ac.iloc[0,0]
fp=ac.iloc[0,1]
fn=ac.iloc[1,0]
tn=ac.iloc[1,1]
Accuracy=(tp+tn)/(tp+fp+fn+tn)
# Recall
Recall = tp/(tp+fn)
# Precision
Precision= tp/(tp+fp)
print("training set:")
print(" Accuracy:", Accuracy)
print("Recall:",Recall)
print("Precision:",Precision)

# There is an overwriting because the indices of the training set are all 100% -
# it is meen that the machine learned exactly the training set and wont recognize other similar data.


# Q8
# limit the tree to avoid overflow
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=40,random_state=RSEED)
clf = clf.fit(X_train, y_train)
plt.figure(figsize=(14, 10), dpi=200)
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=["Democrat","Republican" ], fontsize=4)
# plt.show()

# Q8.a-3
# Q8.b- 5 leaf
# Q8.c-volunteering
# Q8.d- Age and passtime.
# Q8.e-
print("row 68:", dfnum.iloc[67:68,3:4])
y_test_pred68 = clf.predict(X_test)[68]
print(np.column_stack(y_test_pred68))
# The prediction went wrong because the vote in row 68 should be "republican" and the model predicted "democrat" .

# Q9
y_test_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))



# Q10
# the accuracy at the test was not very high- not over-training
# the results of the test set very similar to the train set, it means that the machine learned enough the behavior of the model
# to predict the same accuracy level  on the test.
# the high recall value means that an algorithm returns most of the relevant results- which in our case are the 'Democrat'

# Decision tree - Multiclass -10
# Normalize vote
le.fit(dfnum['vote'])
dfnum['new_vote'] = le.transform(df['vote'])
zscores = stats.zscore(dfnum["new_vote"])
dfnum["new_vote"] = zscores

dfnum = dfnum.drop(["vote"], axis=1)
x = dfnum.drop(["new_status", "status"], axis=1)
y = dfnum["status"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=RSEED)
# building the tree
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=40,random_state=RSEED)
clf = clf.fit(X_train, y_train)
plt.figure(figsize=(18, 15), dpi=100)
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=[ "family","couple", "single"], fontsize=4)
# confusion matrix
y_test_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
# Accuracy
ac = pd.crosstab(y_test, y_test_pred, colnames=["pred"], margins=True)
tp = ac.iloc[0,0]
fp = ac.iloc[0,1]
fn = ac.iloc[1,0]
tn = ac.iloc[1,1]
Accuracy=(tp+tn)/(tp+fp+fn+tn)
print(Accuracy)
# 0.6666666666666666
# The model wont predict the status so good because the accuracy is very low .
# Q11- the precision of single is : 38/72=0.527. It was calculated according to the last confusion matrix.



