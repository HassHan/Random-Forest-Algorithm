#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################################################################
# Import libraries 
#############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#############################################################################
# Read in csv file and assign the labels 
#############################################################################
# Array of strings to hold label names
labels =["age", "workclass", "fnlwgt", "education", "education-num", "maritalstatus",
               "occupation", "relationship", "race", "sex", "capitalgain",
               "capitalloss", "hoursperweek", "nativecountry", "class"] 
# Read in the csv into a dataframe
data = pd.read_csv('adult.csv', header=None, names=labels)


#############################################################################
# Performing initial evaluation of the dataset
#############################################################################
# Get the first 5 data entries
print(data.head(5))
# Get the last 5 data entries 
print(data.tail(5))
# Get all column names 
print(data.columns)
# Get the "age" and "workclass" columns and their values
print (data[['age', 'workclass']])
# Get the data entries that have an age above 40
print (data['age'] > 40)
# Get an overall description of the dataset
print(data.describe())
print(data.cov())
print(data.corr())
print(data.shape)
print(data.corr())

#############################################################################
# Visualise data to see trend in the education type and capital gain
#############################################################################
var = data.groupby('education').capitalgain.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('education')
ax1.set_ylabel('capitalgain')
ax1.set_title("Sum of Capital Gain per Education Type")
var.plot(kind='line')
plt.show()

#############################################################################
# Clean data
#############################################################################
# As most values that are empty are represented with a '?' 
# we can replace all '?' values and replace them as NaN
data = data.replace('[?]', np.nan, regex=True)
# Verify '?' have been replaced with NaN
print(data.tail(30))
# Print the number of null values in the dataset
print(data.isnull().count())
# Drop NaN values
data.dropna(inplace = True)
# Verify NaN values are dropped
print(data.head(30))

#############################################################################
# Save data
#############################################################################
data.to_csv('adult1.csv' , index = False)  

#############################################################################
# Checking if NaN values were actually dropped 
#############################################################################
# The excel file had:
# 32561 rows before the clean
# 30163 rows after the clean