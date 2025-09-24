# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:56:16 2025

@author: Enes
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))



from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# 1) Ölçekleme
sc1 = StandardScaler()
X_scaled = sc1.fit_transform(X)            # X: (n_samples, 1)

sc2 = StandardScaler()
y_scaled = sc2.fit_transform(Y).ravel()    # Y: (n_samples, 1) -> ravel ile (n_samples,)

# 2) SVR modeli
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_scaled, y_scaled)

# 3) Grafik için düzgün eğri: X'i sıralayıp sık grid üret
X_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_grid_scaled = sc1.transform(X_grid)

y_grid_scaled_pred = svr_reg.predict(X_grid_scaled)
y_grid_pred = sc2.inverse_transform(y_grid_scaled_pred.reshape(-1, 1))

# 4) Orijinal uzayda görselleştir (daha anlamlı)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, y_grid_pred, color='blue')
plt.title('SVR (RBF) - Orijinal Ölçekte')
plt.xlabel('Seviye')
plt.ylabel('Maaş')
plt.show()

# 5) Tahmin örnekleri (doğru ölçek akışı)
x_new = np.array([[11]])
x_new_scaled = sc1.transform(x_new)
y_new_scaled_pred = svr_reg.predict(x_new_scaled)
y_new_pred = sc2.inverse_transform(y_new_scaled_pred.reshape(-1, 1))
print("SVR(11) tahmin:", y_new_pred.ravel()[0])

x_new = np.array([[6.6]])
x_new_scaled = sc1.transform(x_new)
y_new_scaled_pred = svr_reg.predict(x_new_scaled)
y_new_pred = sc2.inverse_transform(y_new_scaled_pred.reshape(-1, 1))
print("SVR(6.6) tahmin:", y_new_pred.ravel()[0])


#Decision Tree Regresyonu
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)
Z = X + 0.5
K = X - 0.4


plt.scatter(X, Y, color='red')
plt.plot(X,r_dt.predict(X), color='blue')

plt.plot(X,r_dt.predict(Z), color='green')
plt.plot(X,r_dt.predict(K), color='yellow')
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

# desicion tree algoritması veriyi aralıklar bölerek her aralıkta sabit bir tahmin verir.
# SVR: veriya göre optimum "marj" içinde düzgün bir eğri çizer.

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0) # n_estimators = kaç tane ağaç kullanılacak
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()


from sklearn.metrics import r2_score

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))

# Özet R2 Değerleri
print("----------------------------------")
print("Linear R2 değeri: ")
print(r2_score(Y, lin_reg.predict(X)))

print("Polynomial R2 değeri: ")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)))) # en iyi değerlerden biri

print("SVR R2 Değeri: ")
print(r2_score(y_scaled, svr_reg.predict(X_scaled)))

print("Decision Tree R2 Değeri")
print(r2_score(Y, r_dt.predict(X))) # sıkıntılı bi algoritma

print("Random Forest R2 Değeri: ")
print(r2_score(Y, rf_reg.predict(X))) # en iyi değerlerden biri