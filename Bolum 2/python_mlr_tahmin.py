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
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
print(c)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)


ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)



#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)



#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# boy sütununu al
boy = s2.iloc[:, 3:4].values
print(boy)

# bağımsız değişkenler: boy hariç
sol = s2.iloc[:, :3]  # boydan önceki kolonlar
sag = s2.iloc[:, 4:]  # boydan sonraki kolonlar

veri = pd.concat([sol, sag], axis=1)

# train-test ayırma
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train, y_train)

y_pred = r2.predict(x_test)


import statsmodels.api as sm

X = np.append(arr = np.ones((22, 1)).astype(int), values = veri, axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

# burdan gelen sonuç sonrası en yüksek P değerine sahip olanı elememiz gerekiyor.

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

# görünüğü üzere en yüksek değer '4' ü sildik.


# Böylece backward elimination yöntemiyle uygun olmayan veriyi tahminden kaldırdık.































































