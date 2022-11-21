import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

df = pd.read_csv('Social_Network_Ads.csv')
df = df.set_index('User ID')

X = df[['Gender','Age','EstimatedSalary']]
y = df['Purchased']
X['Gender'].replace(['Male','Female'],[0,1],inplace=True)

XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size=0.15, stratify=y, random_state=20)
clf = RandomForestClassifier(n_estimators=10, max_depth=3)
clf.fit(XTrain,yTrain)

dump(clf, '/modelStorageMap/model.joblib')