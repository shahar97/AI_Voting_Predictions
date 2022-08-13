import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stat
df = pd.read_csv('votersdata.csv')
# hist of age
#age_hist= df.age.plot.hist(bins=20)
#sex_bar =df["age"].value_counts().plot(kind="bar")
plt.show()
df["age"]=df['age'].replace(to_replace=np.nan, value=0)
df["salary"]=df['salary'].replace(to_replace=np.nan, value=0)
df.boxplot(column=["age"] )
le = LabelEncoder()
le.fit(df["vote"])
df["new vote"]=le.transform(df["vote"])

le.fit(df["sex"])
df["new sex"]=le.transform(df["sex"])

le.fit(df["passtime"])
df["new passtime"]=le.transform(df["passtime"])

df1 = df.drop(["vote","passtime","sex","status"], axis=1)

zscores = stat.zscore(df1["salary"])
df1["salary"] = zscores
zscores = stat.zscore(df1["new vote"])
df1["new vote"] = zscores

zscores = stat.zscore(df1["new sex"])
df1["new sex"] = zscores
zscores = stat.zscore(df1["new passtime"])
df1["new passtime"] = zscores
zscores = stat.zscore(df1["age"])
df1["age"] = zscores
zscores = stat.zscore(df1["volunteering"])
df1["volunteering"] = zscores
print(df1.head())

import scipy.cluster.hierarchy as sch
model= sch.linkage(df1,method="complete")
plt.figure(figsize=(25,20))
dan=sch.dendrogram(model)
plt.show()
cluster=sch.fcluster(model,t=4.2,criterion="distance")
df1["cluster"]=cluster


