
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import *
from sklearn.exceptions import *
from sklearn.naive_bayes import *
from sklearn.model_selection import *

images_data_set = pd.read_csv('hw01_images.csv',header =None)
label_data_set = pd.read_csv('hw01_labels.csv', header =None)

#No warning is wanted because of dimensionality issues
warnings.filterwarnings(action='ignore',category=DataConversionWarning)

#As we want 200 images for training data and 400 images for test data
split_rate = 200 / 400

x_axis_training, x_axis_test,y_axis_training,y_axis_test = train_test_split(images_data_set,label_data_set, test_size=split_rate, random_state= 0)

trainingMeanSupplementary = pd.concat([x_axis_training,y_axis_training],axis=1)
count = 0

femaleStatXaxis = trainingMeanSupplementary[trainingMeanSupplementary.iloc[:,-1] == 1]
maleStatXaxis = trainingMeanSupplementary[trainingMeanSupplementary.iloc[:,-1] == 2]

maleStatXaxis = maleStatXaxis.iloc[:, :-1]
femaleStatXaxis = femaleStatXaxis.iloc[:, :-1]

#x stands for data
#y stands for labels

#I am fitting our training data into gaussian naive bayes
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(x_axis_training, y_axis_training)

#Then calculating the prediction value for gender
y_hat_training = gaussian_naive_bayes.predict(x_axis_training)
y_hat_test = gaussian_naive_bayes.predict(x_axis_test)

#Calculation of statistical values
female_label = 1
male_label = 2
male_count = 0
female_count = 0
for i in y_axis_training.values:
    if i == male_label:
        male_count+=1
    elif i == female_label:
        female_count+=1

female_prior_prob = female_count /(male_count + female_count)
male_prior_prob = male_count /(male_count + female_count)

print("\nThe male mean values of each pixel in the image train data set are :\n")
print(np.mean(maleStatXaxis, axis = 0))

print("\nThe female mean values of each pixel in the image train data set are :\n")
print(np.mean(femaleStatXaxis, axis = 0))


print("\nThe male standard deviation values of each pixel in the image train data set are :\n")
print(np.std(maleStatXaxis, axis = 0))

print("\nThe female standard deviation values of each pixel in the image train data set are :\n")
print(np.std(femaleStatXaxis, axis = 0))

print("\nPrior probability for male is : "+str(male_prior_prob))
print("Prior probability for female is : "+str(female_prior_prob)+"\n")


print("\nConfusion matrix for training set :")
print("y-train / y-hat")
confusion_matrix_training = confusion_matrix(y_axis_training,y_hat_training)
print(confusion_matrix_training)

print("\nConfusion matrix for test set :")
print("y-test / y-hat")

confusion_matrix_test = confusion_matrix(y_axis_test,y_hat_test)
print(confusion_matrix_test)
