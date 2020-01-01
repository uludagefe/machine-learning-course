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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

training_data = pd.read_csv('hw08_training_data.csv')
test_data = pd.read_csv('hw08_test_data.csv')
training_labels = pd.read_csv('hw08_training_label.csv')

easy_sample = training_data.iloc[1:1000,:]

easy_sample.drop(training_data.columns[
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
numeric_values = easy_sample.select_dtypes(include=numerics)
numeric_values=numeric_values.fillna(numeric_values.mean())

non_numeric_values = easy_sample.select_dtypes(exclude=numerics)
le = LabelEncoder()

for i in non_numeric_values.columns:
    