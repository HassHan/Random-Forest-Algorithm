# -*- coding: utf-8 -*-
"""
@author: Hassan 
"""
##############################################################################################
#Importing all the libaries that will be used for the Random Forest Classification algorithm.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
##############################################################################################
#Reading the preprocessed csv datasset file. File was originally adult.csv, but reads adult1.csv as this is the preprocessed dataset file.
Adultdata = pd.read_csv('adult1.csv')
##############################################################################################
#Displays the columns along with the information for them.
Adultdata.head()
##############################################################################################
#Preprocessing - Converts 'sex' column features that are Strings (object) into an Integer.
Adultdata.replace(' Female', 0, inplace = True)
Adultdata.replace(' Male', 1, inplace = True)
Adultdata.replace(' White', 2, inplace = True)
Adultdata.replace(' Black', 3, inplace = True)
Adultdata.replace(' Amer-Indian-Eskimo', 4, inplace = True)
Adultdata.replace(' Asian-Pac-Islander', 5, inplace = True)
Adultdata.replace(' Other', 6, inplace = True)
##############################################################################################
#The features of the dataset that will be used for the model and including other features that are linked i.e. hours per week and capital loss.
adult_Features = ["age", 
            "sex", 
            "race", 
            "fnlwgt", 
            "capitalgain", 
            "hoursperweek"]
##############################################################################################
#Collects the dataset features specified.
collect = Adultdata[adult_Features]
#Age feature for y axis.
y = Adultdata.age
#Sex feature for x axis.
x = Adultdata.sex
###############################################################################################
#A for loop to get the accuracy score 10 times.
for i in range (10):
    #Using the train and test methods for x,y with the dataset and establishing a test size and random state.
    x_train, x_test, y_Train, y_Test = train_test_split(collect,y, test_size = .25, random_state = 5)
    #Amount of estimations for the algorithm
    RFclass = RandomForestClassifier(n_estimators = 25)
    #Using x and y train methods.
    RFclass.fit(x_train,y_Train)
    #y predictions will predict x test
    y_Predictions = RFclass.predict(x_test)
    #Confusion matrix shows an array in the variable explorer based from the dataset.
    confusion_Matrix = sklearn.metrics.confusion_matrix(y_Test,y_Predictions) 
    #Prints the array for the confusion matrix
    print(confusion_matrix(y_Test, y_Predictions))
###############################################################################################
#Prints out the accuracy score for the algorithm.
    accuracy_score = print ('Accuracy Score: {}%'.format(sklearn.metrics.accuracy_score(y_Test, y_Predictions) * 100))
#Shows the plot graph.
plt.show()
##Saves the plot graph in same directory as the dataset and the size of graphs being defined when running on Spyder.
sns_plot = sns.pairplot(collect, hue='sex', size=2)
sns_plot.savefig("Random Forest Visualisation.png")