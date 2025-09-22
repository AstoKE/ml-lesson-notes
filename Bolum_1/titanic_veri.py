import numpy as np
import pandas as pd

veriler = pd.read_csv('Titanic-Dataset.csv')

# print(veriler)

name = veriler[['Name']]

# print(name)

name_age = veriler[['Name', 'Age']] # to select the columns or column

print(veriler.head(5))
print("*******************************************")
print(veriler.tail(3))
print("*******************************************")
print(veriler.info())
print("*******************************************")
print(veriler.dtypes)
print("*******************************************")


# Missing Data Control

print(veriler.isnull().sum()) #in this list we can see the missing datas of the data set 

# As a result of isnull func we can see that age has missing values.
# Now we are going to fill them with the average of other ages

veriler['Age'].fillna(veriler['Age'].mean(), inplace=True)
print(veriler.isnull().sum()) # to check are we doing right thing

veriler['Embarked'].fillna('S', inplace=True)
print(veriler.isnull().sum())

# we fill the missing age data with average
# now lets drop(delete) the 'Cabin' column bcs its empty

veriler.drop('Cabin', axis=1, inplace=True)
print(veriler.isnull().sum())

kadınlar = veriler[veriler['Sex']== 'female'] # only females

secim = veriler[(veriler['Age'] > 30) & (veriler['Pclass'] == 1)] # age bigger than 30 and first class passengers

print(kadınlar,"\n*****************************","\n", secim)

print(veriler.groupby('Sex')['Age'].mean()) # average age based on sex

print("*******************************************")

print(veriler.groupby(['Pclass', 'Survived'])['Fare'].mean()) # based on class and surviving to average ticket price

#*********************************
# NumPy Codes
#*********************************

age_array = np.array(veriler['Age'])  # make them array
fare_array = np.array(veriler['Fare'])


print("Average Age: ", np.mean(age_array))
print("Bilet ücreti standart sapması: ", np.std(fare_array))


#boolean indexing 

youngs = age_array[age_array < 18] # passenger below 18

#based on ages categorizing 

veriler['AgeGroup'] = pd.cut(veriler['Age'], bins=[0, 18, 40, 60, 100], labels=['Young', 'Adult', 'Mid-Age', 'Senior'])

print(veriler['AgeGroup'])

ort_yas = np.mean(veriler['Age'])
medyan_fare = np.median(veriler[veriler['Age'] > ort_yas]['Fare'])
print("Sonuç", medyan_fare)


#****************************************
# Mathplotlib
#****************************************

import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")
df['Age'].fillna(df['Age'].mean(), inplace=True)

#Histogram - Yaş dağılımı
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=30, color='purple', edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

#Bar Chart - Cinsiyete göre Hayatta Kalma Sayısı

survival_by_sex = df.groupby('Sex')['Survived'].sum()

plt.figure(figsize=(6, 4))
plt.bar(survival_by_sex.index, survival_by_sex.values, color=['lightblue', 'lightgreen'])
plt.title('Survival Passsengers by Gender')
plt.xlabel('Sex')
plt.ylabel('Survived Count')
plt.show()

#Pie Chart - Yolcu Limanı Dağılımı

df['Embarked'].fillna('S', inplace=True)
embarked_count = df['Embarked'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(embarked_count, labels=embarked_count.index, autopct='%1.1f%%',
        startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Passenger Distribution by Embarkation Port')
plt.axis('equal') #Daire şeklinge göstermek içn
plt.show()


#Scatter Plot - Yaş ve Bilet Ücreti

plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Fare'], alpha=0.5, c='darkblue')
plt.title('Fare vs Age')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.show()