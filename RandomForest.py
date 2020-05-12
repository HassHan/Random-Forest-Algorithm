"""
Created on Wed Dec 20 09:32:15 2017

@author: Hassan Hanif
"""

from pandas import Series, DataFrame
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import seaborn as sns
#from sklearn.tree import export_graphviz

 
data = pd.read_csv('adult1.csv')
data.head()

sns.pairplot(data = data)

feature_cols = ["age", "capitalgain",]
X = data[feature_cols]
y = data.age
#adult1.contains("workclass").astype(int).head()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train,y_train)

predictions = classifier.predict(X_test)
conf_matrix = sklearn.metrics.confusion_matrix(y_test,predictions)
conf_matrix

print(sklearn.metrics.accuracy_score(y_test,predictions)* 100) 

#for tree_in_forest in n_estimators:
#    if (i_tree <1):
#                export_graphviz(tree_in_forest,
#                feature_names=X.columns,
#                filled=True,
#                rounded=True)
#
#os.system('dot -Tpng tree.dot -o tree.png')