#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:55:50 2019

@author: emre
"""
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
pca = PCA(n_components=100)

training_data = pd.read_csv('hw08_training_data.csv')
test_data = pd.read_csv('hw08_test_data.csv')
training_labels = pd.read_csv('hw08_training_label.csv')

easy_sample = training_data.iloc[0:1000,:]

easy_sample.drop(training_data.columns[
        [0,1,2,16,17,18,19,73,74,75,76,77,78,79,80,81,82,83,84,85,91,127,129,130,
         137,138,139,146,147,148,154,155,156,160,162,179,180,186
         ,187,189,190,191,208,209,223,224,225]
        ],axis=1,inplace=True)

test_data.drop(training_data.columns[
        [0,1,2,16,17,18,19,73,74,75,76,77,78,79,80,81,82,83,84,85,91,127,129,130,
         137,138,139,146,147,148,154,155,156,160,162,179,180,186
         ,187,189,190,191,208,209,223,224,225]
        ],axis=1,inplace=True)

training_data.drop(training_data.columns[
        [0,1,2,16,17,18,19,73,74,75,76,77,78,79,80,81,82,83,84,85,91,127,129,130,
         137,138,139,146,147,148,154,155,156,160,162,179,180,186
         ,187,189,190,191,208,209,223,224,225]
        ],axis=1,inplace=True)



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_values_test = training_data.select_dtypes(include=numerics)
numeric_values_test = numeric_values_test.fillna(numeric_values_test.mean())

numeric_values = training_data.select_dtypes(include=numerics)
numeric_values=numeric_values.fillna(numeric_values.mean())

easy_numeric_values = easy_sample.select_dtypes(include=numerics)
easy_numeric_values=numeric_values.fillna(easy_numeric_values.mean())
easy_pca_compressed = pd.DataFrame(pca.fit_transform(easy_numeric_values))

ohe = OneHotEncoder()
non_numeric_values = training_data.select_dtypes(exclude=numerics)
non_numeric_values['VAR131'].fillna('no data',inplace=True)

from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
non_numeric_values=pd.DataFrame(ohe.fit_transform(non_numeric_values).toarray())
numeric_values = pd.DataFrame(ss.fit_transform(numeric_values))
x_training = pd.concat([non_numeric_values,numeric_values],axis=1)

target_1_label = training_labels.iloc[:,1].fillna(0)
target_2_label = training_labels.iloc[:,2].fillna(0)
target_3_label = training_labels.iloc[:,3].fillna(0)
target_4_label = training_labels.iloc[:,4].fillna(0)
target_5_label = training_labels.iloc[:,5].fillna(0)
target_6_label = training_labels.iloc[:,6].fillna(0)

target = target_1_label + target_2_label + target_3_label + target_4_label + target_5_label + target_6_label

#Splitting data into training and test data
#from sklearn.model_selection import train_test_split
#x_training, x_test,y_training,y_test = train_test_split(all_values,target,test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_training,y_training)
y_prediction = knn.predict(x_test)
cm = confusion_matrix(y_test,y_prediction)
print('KNN')
print(cm)
mae = mean_absolute_error(y_test, y_prediction)
rms = sqrt(mean_squared_error(y_test, y_prediction))
score_funtion = knn.score(x_test, y_test)
acc_score = accuracy_score(y_test, y_prediction)
print('Mean Absolute Error is : ' , mae)
print('Root Mean Square Error is : ' , rms)
print('Score of Function is : ', score_funtion )
print('Accuracy score:')
print(acc_score)




