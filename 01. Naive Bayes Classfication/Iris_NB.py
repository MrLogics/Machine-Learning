import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


names= ['sepal-length', 'sepal-width', 'petal-lenth', 'petal-width', 'species']

dataset=pd.read_csv('..\Data\iris.data',names=names,sep=',')

dataset.head()

X=dataset.iloc[:, 0:3].values
Y=dataset.iloc[:, 4].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

gnb=GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))
print(metrics.accuracy_score(Y_test, Y_pred))
