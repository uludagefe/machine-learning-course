#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:23:49 2019

@author: burakegeli
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

### Data Preprocessing

###Loding Data
data = pd.read_csv('hw08_training_data.csv')
label_data = pd.read_csv('hw08_training_label.csv')
test_data = pd.read_csv('hw08_test_data.csv')

## First elimination: Elemination of meaningless values
first_group = data.iloc[:,3:16]
second_group = data.iloc[:,21:73]
third_group = data.iloc[:,86:91]
fourth_group = data.iloc[:,92:127]
fifth_group_final = data.iloc[:,128:129]
sixth_group = data.iloc[:,131:137]
seventh_group_final = data.iloc[:,140:146]
eighth_group = data.iloc[:,149:155]
nineth_group = data.iloc[:,157:179]
tenth_group_final = data.iloc[:,181:186]
eleventh_group_final = data.iloc[:,188:189]
twelveth_group = data.iloc[:,192:208]
thirteenth_group = data.iloc[:,210:224]

imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)

## Second elimination : Replacing all the nan and categorical values

#First Group
numbers_1 = first_group.iloc[:,1:11].values
imputer = imputer.fit(numbers_1)
numbers_1 = imputer.transform(numbers_1)
numbers_2 = first_group.iloc[:,12].values
cat_val_1 = first_group.iloc[:,0:1].values
cat_val_2 = first_group.iloc[:,11:12].values
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features = 'all')
cat_val_1 = le.fit_transform(cat_val_1)
cat_val_1 = pd.DataFrame(cat_val_1)
cat_val_1 = ohe.fit_transform(cat_val_1).toarray()
cat_val_2 = le.fit_transform(cat_val_2)
cat_val_2 = pd.DataFrame(cat_val_2)
cat_val_2 = ohe.fit_transform(cat_val_2).toarray()
cat_val_1 = pd.DataFrame(cat_val_1)
cat_val_2 = pd.DataFrame(cat_val_2)
numbers_1 = pd.DataFrame(numbers_1)
numbers_2 = pd.DataFrame(numbers_2)
step_1 = pd.concat([cat_val_1, numbers_1], axis = 1)
step_2 = pd.concat([step_1,cat_val_2], axis = 1)
first_group_final = pd.concat([step_2, numbers_2], axis = 1)
#End of the first group

#Second Group
imputer = imputer.fit(second_group)
second_group_final = imputer.transform(second_group)
second_group_final = pd.DataFrame(second_group_final)
#End of the second group

#Third Group
imputer = imputer.fit(third_group)
third_group_final = imputer.transform(third_group)
third_group_final = pd.DataFrame(third_group_final)
#End of the third group

#Fourth Group

sub_group_1 =  fourth_group.iloc[:,0:2].values
sub_group_1 = pd.DataFrame(sub_group_1)

sub_group_2 = fourth_group.iloc[:,2:3].values
sub_group_2 = le.fit_transform(sub_group_2)
sub_group_2 = pd.DataFrame(sub_group_2)
sub_group_2 = ohe.fit_transform(sub_group_2).toarray()
sub_group_2 = pd.DataFrame(sub_group_2)

sub_group_3 = fourth_group.iloc[:,3:9]
imputer = imputer.fit(sub_group_3)
sub_group_3 = imputer.transform(sub_group_3)
sub_group_3 = pd.DataFrame(sub_group_3)

sub_group_4 = fourth_group.iloc[:,9:10]
sub_group_4 = le.fit_transform(sub_group_4)
sub_group_4 = pd.DataFrame(sub_group_4)
sub_group_4 = ohe.fit_transform(sub_group_4).toarray()
sub_group_4 = pd.DataFrame(sub_group_4)

sub_group_5 = fourth_group.iloc[:,10:19]

sub_group_6 = fourth_group.iloc[:,19:20]
sub_group_6 = le.fit_transform(sub_group_6)
sub_group_6 = pd.DataFrame(sub_group_6)
sub_group_6 = ohe.fit_transform(sub_group_6).toarray()
sub_group_6 = pd.DataFrame(sub_group_6)

sub_group_7 = fourth_group.iloc[:,20:34]
imputer = imputer.fit(sub_group_7)
sub_group_7 = imputer.transform(sub_group_7)
sub_group_7 = pd.DataFrame(sub_group_7)

step_4_1 = pd.concat([sub_group_1, sub_group_2], axis = 1)
step_4_2 = pd.concat([step_4_1,sub_group_3], axis = 1)
step_4_3 = pd.concat([step_4_2, sub_group_4], axis = 1)
step_4_4 = pd.concat([step_4_3,sub_group_5], axis = 1)
step_4_5 = pd.concat([step_4_4,sub_group_6], axis = 1)

fourth_group_final = pd.concat([step_4_5,sub_group_7], axis = 1)
#End of the fourth group

#Sixth Group
#Neglected since categorical values contain nan values
#sub_gro_1 = sixth_group.iloc[:,0:1]
#sub_gro_1 = le.fit_transform(sub_gro_1)
#sub_gro_1 = pd.DataFrame(sub_gro_1)
#sub_gro_1 = ohe.fit_transform(sub_gro_1).toarray()
#sub_gro_1 = pd.DataFrame(sub_gro_1)

sub_gro_2 = sixth_group.iloc[:,1:2].values
sub_gro_2 = le.fit_transform(sub_gro_2)
sub_gro_2 = pd.DataFrame(sub_gro_2)
sub_gro_2 = ohe.fit_transform(sub_gro_2).toarray()
sub_gro_2 = pd.DataFrame(sub_gro_2)
sub_gro_3 = sixth_group.iloc[:,2:]
sixth_group_final = pd.concat([sub_gro_2,sub_gro_3], axis = 1)
#End of the sixth group

#Eighth Group
sub_gr_1 = eighth_group.iloc[:,0:1].values
sub_gr_1 = le.fit_transform(sub_gr_1)
sub_gr_1 = pd.DataFrame(sub_gr_1)
sub_gr_1 = ohe.fit_transform(sub_gr_1).toarray()
sub_gr_1 = pd.DataFrame(sub_gr_1)

sub_gr_2 = eighth_group.iloc[:,1:2].values
sub_gr_2 = le.fit_transform(sub_gr_2)
sub_gr_2 = pd.DataFrame(sub_gr_2)
sub_gr_2 = ohe.fit_transform(sub_gr_2).toarray()
sub_gr_2 = pd.DataFrame(sub_gr_2)

sub_gr_3 = eighth_group.iloc[:,2:4]

sub_gr_4 = eighth_group.iloc[:,4:5]
sub_gr_4 = le.fit_transform(sub_gr_4)
sub_gr_4 = pd.DataFrame(sub_gr_4)
sub_gr_4 = ohe.fit_transform(sub_gr_4).toarray()
sub_gr_4 = pd.DataFrame(sub_gr_4)

sub_gr_5 = eighth_group.iloc[:,5:6]
sub_gr_5 = le.fit_transform(sub_gr_5)
sub_gr_5 = pd.DataFrame(sub_gr_5)
sub_gr_5 = ohe.fit_transform(sub_gr_5).toarray()
sub_gr_5 = pd.DataFrame(sub_gr_5)

step_8_1 = pd.concat([sub_gr_1, sub_gr_2], axis = 1)
step_8_2 = pd.concat([step_8_1,sub_gr_3], axis = 1)
step_8_3 = pd.concat([step_8_2, sub_gr_4], axis = 1)
eighth_group_final = pd.concat([step_8_3,sub_gr_5], axis = 1)
eighth_group_final = eighth_group_final
#End of the eighth group


#Nineth Group
sub_gro_1 = nineth_group.iloc[:,0:2]

sub_gro_2 = nineth_group.iloc[:,2:3]
sub_gro_2 = le.fit_transform(sub_gro_2)
sub_gro_2 = pd.DataFrame(sub_gro_2)
sub_gro_2 = ohe.fit_transform(sub_gro_2).toarray()
sub_gro_2 = pd.DataFrame(sub_gro_2)

sub_gro_3 = nineth_group.iloc[:,3:4]
sub_gro_3 = le.fit_transform(sub_gro_3)
sub_gro_3 = pd.DataFrame(sub_gro_3)
sub_gro_3 = ohe.fit_transform(sub_gro_3).toarray()
sub_gro_3 = pd.DataFrame(sub_gro_3)

sub_gro_4 = nineth_group.iloc[:,4:5]

sub_gro_5 = nineth_group.iloc[:,6:]
imputer = imputer.fit(sub_gro_5)
sub_gro_5 = imputer.transform(sub_gro_5)
sub_gro_5 = pd.DataFrame(sub_gro_5)

step_9_1 = pd.concat([sub_gro_1, sub_gro_2], axis = 1)
step_9_2 = pd.concat([step_9_1,sub_gro_3], axis = 1)
step_9_3 = pd.concat([step_9_2, sub_gro_4], axis = 1)
nineth_group_final = pd.concat([step_9_3,sub_gro_5], axis = 1)
nineth_group_final = nineth_group_final
#End of the nineth group


#Twelveth Group
imputer = imputer.fit(twelveth_group)
twelveth_group_final = imputer.transform(twelveth_group)
twelveth_group_final = pd.DataFrame(twelveth_group_final)
#End of the twelveth group


#Thirteenth Group

sub_g_1 = thirteenth_group.iloc[:,0:12]
imputer = imputer.fit(sub_g_1)
sub_g_1 = imputer.transform(sub_g_1)
sub_g_1 = pd.DataFrame(sub_g_1)

sub_g_2 = thirteenth_group.iloc[:,12:13]
sub_g_2 = le.fit_transform(sub_g_2)
sub_g_2 = pd.DataFrame(sub_g_2)
sub_g_2 = ohe.fit_transform(sub_g_2).toarray()
sub_g_2 = pd.DataFrame(sub_g_2)

sub_g_3 = thirteenth_group.iloc[:,13:14]
sub_g_3 = le.fit_transform(sub_g_3)
sub_g_3 = pd.DataFrame(sub_g_3)
sub_g_3 = ohe.fit_transform(sub_g_3).toarray()
sub_g_3 = pd.DataFrame(sub_g_3)

step_13_1 = pd.concat([sub_g_1, sub_g_2], axis = 1)
thirteenth_group_final = pd.concat([step_13_1,sub_g_3], axis = 1)
#End of the twelveth group



#Creating final test data

final_1 = pd.concat([first_group_final, second_group_final], axis = 1)
final_2 = pd.concat([final_1,third_group_final], axis = 1)
final_3 = pd.concat([final_2, fourth_group_final], axis = 1)
final_4 = pd.concat([final_3, fifth_group_final], axis = 1)
final_5 = pd.concat([final_4,sixth_group_final], axis = 1)
final_6 = pd.concat([final_5, seventh_group_final], axis = 1)
final_7 = pd.concat([final_6, eighth_group_final], axis = 1)
final_8 = pd.concat([final_7,nineth_group_final], axis = 1)
final_9 = pd.concat([final_8, tenth_group_final], axis = 1)
final_10 = pd.concat([final_9, eleventh_group_final], axis = 1)
final_11 = pd.concat([final_10,twelveth_group_final], axis = 1)
test_data = pd.concat([final_11, thirteenth_group_final], axis = 1)

#Prepeating Label Data

label_data_prep = label_data.iloc[:,1:]
imputer = imputer.fit(label_data_prep)
label_data_prep = imputer.transform(label_data_prep)
label_data_prep = pd.DataFrame(label_data_prep)

x = test_data.values.astype(int)
y1 = label_data_prep.iloc[:,0].values.astype(int)
y2 = label_data_prep.iloc[:,1:2].values
y3 = label_data_prep.iloc[:,2:3].values
y4 = label_data_prep.iloc[:,3:4].values
y5 = label_data_prep.iloc[:,4:5].values
y6 = label_data_prep.iloc[:,5:6].values

#Train - Test Split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y1,test_size=0.2, random_state=0)




sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)




# 2. KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)

y_pred_1 = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred_1)
print('KNN')
print(cm)

mae = mean_absolute_error(y_test, y_pred_1)

rms = sqrt(mean_squared_error(y_test, y_pred_1))

score_funtion = knn.score(x_test, y_test)
print('Mean Absolute Error is : ' , mae)

print('Root Mean Square Error is : ' , rms)

print('Score of Function is : ', score_funtion )

# 3. SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred_1 = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred_1)
print('SVC')
print(cm)

mae = mean_absolute_error(y_test, y_pred_1)

rms = sqrt(mean_squared_error(y_test, y_pred_1))

score_funtion = svc.score(x_test, y_test)
print('Mean Absolute Error is : ' , mae)

print('Root Mean Square Error is : ' , rms)

print('Score of Function is : ', score_funtion )

# 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_1 = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred_1)
print('GNB')
print(cm)

mae = mean_absolute_error(y_test, y_pred_1)

rms = sqrt(mean_squared_error(y_test, y_pred_1))

score_funtion = gnb.score(x_test, y_test)
print('Mean Absolute Error is : ' , mae)

print('Root Mean Square Error is : ' , rms)

print('Score of Function is : ', score_funtion )

# 5. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred_1 = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred_1)
print('DTC')
print(cm)

mae = mean_absolute_error(y_test, y_pred_1)

rms = sqrt(mean_squared_error(y_test, y_pred_1))

score_funtion = dtc.score(x_test, y_test)
print('Mean Absolute Error is : ' , mae)

print('Root Mean Square Error is : ' , rms)

print('Score of Function is : ', score_funtion )

# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred_1 = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred_1)
print('RFC')
print(cm)

mae = mean_absolute_error(y_test, y_pred_1)

rms = sqrt(mean_squared_error(y_test, y_pred_1))

score_funtion = rfc.score(x_test, y_test)
print('Mean Absolute Error is : ' , mae)

print('Root Mean Square Error is : ' , rms)

print('Score of Function is : ', score_funtion )
#Making Predictions for Target_1
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

y_pred_1 = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred_1)
print('XGBOOST')
print(cm)

mae = mean_absolute_error(y_test, y_pred_1)

rms = sqrt(mean_squared_error(y_test, y_pred_1))

score_funtion = classifier.score(x_test, y_test)
print('Mean Absolute Error is : ' , mae)

print('Root Mean Square Error is : ' , rms)

print('Score of Function is : ', score_funtion )




np.savetxt("test_predictions.csv",y_pred_test_data , delimiter=",", fmt='%d')

#NaN detector
counter = 0
for x in nineth_group['VAR163']:
    if x == 'NaN':
        counter = counter + 1
print(counter)
