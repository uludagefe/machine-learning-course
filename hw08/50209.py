#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:55:50 2019

@author: Emre Uludağ
"""
import numpy as np
import pandas as pd
from math import sqrt,isnan
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
training_data = pd.read_csv('hw08_training_data.csv')
test_data = pd.read_csv('hw08_test_data.csv')
training_labels = pd.read_csv('hw08_training_label.csv')

#========================================================================
#========================================================================
#========================================================================
#========================================================================
#==============getting rid of unnecessary parts of the data==============
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

training_labels.drop(training_labels.columns[0],axis=1,inplace=True)

elements = []
for index, row in training_labels.iterrows():
    for element in row:
        if(not isnan(element) and element == 1):
            elements.append(index)                
            print(index)
            break
training_data = training_data.iloc[elements,:]
training_labels = training_labels.iloc[elements,:]
#==============getting rid of unnecessary parts of the data==============
#========================================================================
#========================================================================
#========================================================================
#========================================================================

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_values = training_data.select_dtypes(include=numerics)
numeric_values=numeric_values.fillna(numeric_values.mean())

numeric_values_test = test_data.select_dtypes(include=numerics)
numeric_values_test = numeric_values_test.fillna(numeric_values_test.mean())

ohe = OneHotEncoder()
ss= StandardScaler()
pca = PCA(n_components=100)

non_numeric_values = training_data.select_dtypes(exclude=numerics)
non_numeric_values['VAR131'].fillna('no data',inplace=True)
non_numeric_values=pd.DataFrame(ohe.fit_transform(non_numeric_values).toarray())

non_numeric_values_test = test_data.select_dtypes(exclude=numerics)
non_numeric_values_test['VAR131'].fillna('no data',inplace=True)
non_numeric_values_test=pd.DataFrame(ohe.fit_transform(non_numeric_values_test).toarray())

numeric_values = pd.DataFrame(ss.fit_transform(numeric_values))
x_training = pd.concat([non_numeric_values,numeric_values],axis=1)
x_training = pd.DataFrame(pca.fit_transform(x_training))

numeric_values_test = pd.DataFrame(ss.fit_transform(numeric_values_test))
x_for_test = pd.concat([non_numeric_values_test,numeric_values_test],axis=1)
x_for_test = pd.DataFrame(pca.fit_transform(x_for_test))

def get_target(row):
    for c in training_labels.columns:
         if row[c]== 1 and c == 'TARGET_1':
             return 0
         elif row[c]== 1 and c == 'TARGET_2':
             return 1
         elif row[c]== 1 and c == 'TARGET_3':
             return 2
         elif row[c]== 1 and c == 'TARGET_4':
             return 3
         elif row[c]== 1 and c == 'TARGET_5':
             return 4
         elif row[c]== 1 and c == 'TARGET_6':
             return 5

def encode_targets(row):
         if row[0]== 0 :
             return pd.Series([1,0,0,0,0,0])
         elif row[0] == 1:
             return pd.Series([0,1,0,0,0,0])
         elif row[0] == 2:
             return pd.Series([0,0,1,0,0,0])
         elif row[0] == 3:
             return pd.Series([0,0,0,1,0,0])
         elif row[0] == 4:
             return pd.Series([0,0,0,0,1,0])
         elif row[0] == 5:
             return pd.Series([0,0,0,0,0,1])

training_labels.fillna(0,inplace=True)
training_labels = training_labels.apply(get_target, axis=1)

#Splitting data into training and test data
from sklearn.model_selection import train_test_split
x_training, x_test,y_training,y_test = train_test_split(x_training,training_labels,test_size=0.2, random_state=0)


from xgboost import DMatrix,train
xg_train = DMatrix(x_training, label=y_training)
xg_test = DMatrix(x_test, label=y_test)
xg_test_real = DMatrix(x_for_test)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 12
bst = train(param, xg_train, num_round, watchlist)
# get prediction
prediction_training = pd.DataFrame(bst.predict(xg_test))
prediction_test = pd.DataFrame(bst.predict(xg_test_real))

cm = confusion_matrix(y_test,prediction_training)
acc_score = accuracy_score(y_test, prediction_training)
#fpr,tpr,treshhold = roc_curve(y_test, pred)
print('Training results in the form of confusion matrix:')
print(cm)
print('Accuracy:')
print(acc_score)
error_rate = np.sum(prediction_training != y_test) / y_test.shape[0]
print('Test error using softmax = {}'.format(error_rate))

acc_score = accuracy_score(y_test, prediction_training)

print('Accuracy score:')
print(acc_score)


#==========Part for writing prediction data to a file==========

#Transform data to encoded format according to class names 'TARGET_1,TARGET_2,...'
prediction_test = prediction_test.apply(encode_targets, axis=1)
#Write test data predictions to csv file
np.savetxt("hw08_test_predictions.csv",prediction_test.apply(encode_targets, axis=1) , delimiter=",", fmt='%d')
