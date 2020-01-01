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

sample_test = target1_training.iloc[0:100,0:100]

