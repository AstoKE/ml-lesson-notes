# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('Iris.xls')
#pd.read_csv("veriler.csv")
#test

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişkenler

# print(x)
# print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0) 
#yüzde 33 i test için yüzde 67 si train için ayrıldı

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print(X_train)
print("************")
print(X_test)

# Buradan itibaren sınıflandırma algoritmaları başlar
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(X_train, y_train) 

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


# Confusion Matrix python

from sklearn.metrics import confusion_matrix

print("LR")
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("KNN")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# linear veri kümelerinde 2 tip veriyi ayırmaya yarayan algo
from sklearn.svm import SVC

svc = SVC(kernel='poly')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('SVC')
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Doğrusal olmayan (non-linear) veri kümesi için kernel trick
# Çok kullanılıyor


# Naive Bayes 

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('GNB')
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DTC')
print(cm)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=12, criterion='entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

# Sınıflandırma olasılıklarını verir
y_proba = rfc.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
print('RFC')
print(cm)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,0], pos_label='e')
print(fpr)
print(tpr)


