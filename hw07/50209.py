#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:36:09 2019

@author: Emre Uludağ
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

target1_training = pd.read_csv('hw07_target1_training_data.csv')
target1_test = pd.read_csv('hw07_target1_test_data.csv')
target1_training_label = pd.read_csv('hw07_target1_training_label.csv')

target2_training = pd.read_csv('hw07_target2_training_data.csv')
target2_test = pd.read_csv('hw07_target2_test_data.csv')
target2_training_label = pd.read_csv('hw07_target2_training_data.csv')

target3_training = pd.read_csv('hw07_target3_training_data.csv')
target3_test = pd.read_csv('hw07_target3_test_data.csv')
target3_training_label = pd.read_csv('hw07_target3_training_label.csv')

sample_test = target1_training.iloc[0:1000,:]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def take_rmse(x):
    size = len(x)
    inner_sum = 0
    mean_of_data = np.mean(x)
    for i in x:
        inner_sum += np.square(i-mean_of_data)
    return np.sqrt(inner_sum/size)

def count_nan(x):
    count = 0
    for i in x:
        if(str(i) == 'nan'):
            count+=1
    return count


def do_label_encoding(x):
    le.fit(x)
    x = le.transform(x)
    print(x)





# Below lines will be run when the preprocessing starts
# =============================================================================
# 
#    
#    
# print(sample_test.iloc[:,45])
# do_label_encoding(sample_test.iloc[:,45])
# 
# for i in range(len(sample_test.columns)):
#     print(count_nan(sample_test.iloc[:,i])/len(sample_test.iloc[:,i]))
# 
# for i in range(len(sample_test.columns)):
#     if(count_nan(sample_test.iloc[:,i])/len(sample_test.iloc[:,i])>0.05):
#         print(i)
#         
# =============================================================================
        

