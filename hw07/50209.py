#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:36:09 2019

@author: Emre Uludağ
"""
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
pca = PCA(n_components=100)

target1_training = pd.read_csv('hw07_target1_training_data.csv')
target1_training.drop(target1_training.columns[0],axis=1,inplace=True)
target1_test = pd.read_csv('hw07_target1_test_data.csv')
target1_training_label = pd.read_csv('hw07_target1_training_label.csv')
target1_training_label.drop(target1_training_label.columns[0],axis=1,inplace=True)

target2_training = pd.read_csv('hw07_target2_training_data.csv')
target2_training.drop(target2_training.columns[0],axis=1,inplace=True)
target2_test = pd.read_csv('hw07_target2_test_data.csv')
target2_training_label = pd.read_csv('hw07_target2_training_label.csv')
target2_training_label.drop(target2_training_label.columns[0],axis=1,inplace=True)

target3_training = pd.read_csv('hw07_target3_training_data.csv')
target3_training.drop(target3_training.columns[0],axis=1,inplace=True)
target3_test = pd.read_csv('hw07_target3_test_data.csv')
target3_training_label = pd.read_csv('hw07_target3_training_label.csv')
target3_training_label.drop(target3_training_label.columns[0],axis=1,inplace=True)

#Filling nan data with mean values.
target1_training=target1_training.fillna(target1_training.mean())
target2_training=target2_training.fillna(target2_training.mean())
target3_training=target3_training.fillna(target3_training.mean())

#Filling nan data with mean values in test set
target1_test=target1_test.fillna(target1_test.mean())
target2_test=target2_test.fillna(target2_test.mean())
target3_test=target3_test.fillna(target3_test.mean())

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#training 1 data preprocessing
non_numeric_values_1 = target1_training.select_dtypes(exclude=numerics)
numeric_values_1 = target1_training.select_dtypes(include=numerics)
non_numeric_values_1.fillna('none',inplace=True)

#test 1 data preprocessing
non_numeric_values_test_1 = target1_test.select_dtypes(exclude=numerics)
numeric_values_test_1 = target1_test.select_dtypes(include=numerics)
non_numeric_values_test_1.fillna('none',inplace=True)

#training 2 data preprocessing
non_numeric_values_2 = target2_training.select_dtypes(exclude=numerics)
numeric_values_2 = target2_training.select_dtypes(include=numerics)
non_numeric_values_2.fillna('none',inplace=True)

#test 2 data preprocessing
non_numeric_values_test_2 = target2_test.select_dtypes(exclude=numerics)
numeric_values_test_2 = target2_test.select_dtypes(include=numerics)
non_numeric_values_test_2.fillna('none',inplace=True)

#training 3 data preprocessing
non_numeric_values_3 = target3_training.select_dtypes(exclude=numerics)
numeric_values_3 = target3_training.select_dtypes(include=numerics)
non_numeric_values_3.fillna('none',inplace=True)

#test 3 data preprocessing
non_numeric_values_test_3 = target3_test.select_dtypes(exclude=numerics)
numeric_values_test_3 = target3_test.select_dtypes(include=numerics)
non_numeric_values_test_3.fillna('none',inplace=True)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

non_numeric_values_1=pd.DataFrame(ohe.fit_transform(non_numeric_values_1).toarray())
non_numeric_values_2=pd.DataFrame(ohe.fit_transform(non_numeric_values_2).toarray())
non_numeric_values_3=pd.DataFrame(ohe.fit_transform(non_numeric_values_3).toarray())

non_numeric_values_test_1=pd.DataFrame(ohe.fit_transform(non_numeric_values_test_1).toarray())
non_numeric_values_test_2=pd.DataFrame(ohe.fit_transform(non_numeric_values_test_2).toarray())
non_numeric_values_test_3=pd.DataFrame(ohe.fit_transform(non_numeric_values_test_3).toarray())


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

#training target 1
numeric_values_1 = pd.DataFrame(ss.fit_transform(numeric_values_1))
numeric_values_1 = pd.DataFrame(pca.fit_transform(numeric_values_1))
x_training_1 = pd.concat([numeric_values_1,non_numeric_values_1],axis=1,ignore_index = True)

#test target 1
numeric_values_test_1 = pd.DataFrame(ss.fit_transform(numeric_values_test_1))
numeric_values_test_1 = pd.DataFrame(pca.fit_transform(numeric_values_test_1))
x_test_1 = pd.concat([numeric_values_test_1,non_numeric_values_test_1],axis=1,ignore_index = True)

#training target 2
numeric_values_2 = pd.DataFrame(ss.fit_transform(numeric_values_2))
numeric_values_2 = pd.DataFrame(pca.fit_transform(numeric_values_2))
x_training_2 = pd.concat([numeric_values_2,non_numeric_values_2],axis=1,ignore_index = True)

#test target 2
numeric_values_test_2 = pd.DataFrame(ss.fit_transform(numeric_values_test_2))
numeric_values_test_2 = pd.DataFrame(pca.fit_transform(numeric_values_test_2))
x_test_2 = pd.concat([numeric_values_test_2,non_numeric_values_test_2],axis=1,ignore_index = True)

#training target 3
numeric_values_3 = pd.DataFrame(ss.fit_transform(numeric_values_3))
numeric_values_3 = pd.DataFrame(pca.fit_transform(numeric_values_3))
x_training_3 = pd.concat([numeric_values_3,non_numeric_values_3],axis=1,ignore_index = True)

#test target 3
numeric_values_test_3 = pd.DataFrame(ss.fit_transform(numeric_values_test_3))
numeric_values_test_3 = pd.DataFrame(pca.fit_transform(numeric_values_test_3))
x_test_3 = pd.concat([numeric_values_test_3,non_numeric_values_test_3],axis=1,ignore_index = True)


#Splitting data into training and test data
from sklearn.model_selection import train_test_split
x_training_1_splited, x_test_1_splitted, y_training_1, y_test_1 = train_test_split(x_training_1,target1_training_label,test_size=0.2, random_state=1)
x_training_2_splited, x_test_2_splitted, y_training_2, y_test_2 = train_test_split(x_training_2,target2_training_label,test_size=0.2, random_state=1)
x_training_3_splited, x_test_3_splitted, y_training_3, y_test_3 = train_test_split(x_training_3,target3_training_label,test_size=0.2, random_state=1)
#Above part is for dividing the data into training and test sets in order to train my model and benchmark it afterwards.



from xgboost import XGBClassifier
classifier = XGBClassifier()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

from sklearn.neighbors import KNeighborsClassifier


#classifier.fit(x_training_1_splited, y_training_1)
#y_predict_1 = classifier.predict(x_test_1_splitted)

knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(x_training_1_splited,y_training_1)
y_predict_1 = knn.predict(x_test_1_splitted)

#dtc.fit(x_training_1_splited,y_training_1)
#y_predict_1 = dtc.predict(x_test_1_splitted)

cm = confusion_matrix(y_test_1,y_predict_1)
acc_score = accuracy_score(y_test_1, y_predict_1)
fpr,tpr,treshhold = roc_curve(y_test_1, y_predict_1)

print('For target group 1:')
print('Training results in the form of confusion matrix:')
print(cm)
print('Accuracy:')
print(acc_score)
print ('Auroc: ')
print(auc(fpr,tpr))

dtc.fit(x_training_2_splited,y_training_2)
y_predict_2 = dtc.predict(x_test_2_splitted)

#knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
#knn.fit(x_training_2_splited,y_training_2)
#y_predict_2 = knn.predict(x_test_2_splitted)

#classifier.fit(x_training_2_splited, y_training_2)
#y_predict_2 = classifier.predict(x_test_2_splitted)

cm = confusion_matrix(y_test_2,y_predict_2)
acc_score = accuracy_score(y_test_2, y_predict_2)
fpr,tpr,treshhold = roc_curve(y_test_2, y_predict_2)

print('For target group 2:')
print('Training results in the form of confusion matrix:')
print(cm)
print('Accuracy:')
print(acc_score)
print ('Auroc: ')
print(auc(fpr,tpr))

#classifier.fit(x_training_3_splited, y_training_3)
#y_predict_3 = classifier.predict(x_test_3_splitted)

dtc.fit(x_training_3_splited,y_training_3)
y_predict_3 = dtc.predict(x_test_3_splitted)

#knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
#knn.fit(x_training_3_splited,y_training_3)
#y_predict_3 = knn.predict(x_test_3_splitted)

cm = confusion_matrix(y_test_3,y_predict_3)
acc_score = accuracy_score(y_test_3, y_predict_3)
fpr,tpr,treshhold = roc_curve(y_test_3, y_predict_3)

print('For target group 3:')
print('Training results in the form of confusion matrix:')
print(cm)
print('Accuracy:')
print(acc_score)
print ('Auroc: ')
print(auc(fpr,tpr))

#SOLUTION PART, ABOVE WAS ONLY PREPARATION

#So I am going to train the model with uncommented methodologies as they gave the best results

#TARGET 1
knn.fit(x_training_1,target1_training_label)
y_predict_real_1=knn.predict(x_test_1)
np.savetxt("hw07_target1_test_predictions.csv",y_predict_real_1 , delimiter=",", fmt='%d')

#TARGET 2
dtc.fit(x_training_2,target2_training_label)
y_predict_real_2=dtc.predict(x_test_1)
np.savetxt("hw07_target2_test_predictions.csv",y_predict_real_2 , delimiter=",", fmt='%d')

#TARGET 3
dtc.fit(x_training_3,target3_training_label)
y_predict_real_3=dtc.predict(x_test_3)
np.savetxt("hw07_target3_test_predictions.csv",y_predict_real_3 , delimiter=",", fmt='%d')

