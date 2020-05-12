#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn import datasets
#iris = datasets.load_iris()
#
#df = pd.DataFrame(iris.data, columns=iris.feature_names)
## sklearn provides the iris species as integer values since this is required for classification
## here we're just adding a column with the species names to the dataframe for visualisation
#df['species'] = np.array([iris.target_names[i] for i in iris.target])
#sns.pairplot(df, hue='species')
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target, test_size=0.5, stratify=iris.target, random_state=123456)
#
#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
#rf.fit(X_train, y_train)
#
#from sklearn.metrics import accuracy_score
#predicted = rf.predict(X_test)
#accuracy = accuracy_score(y_test, predicted)
#print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
#print(f'Mean accuracy score: {accuracy:.3}')
#
#from sklearn.metrics import confusion_matrix
#cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
#sns.heatmap(cm, annot=True)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Load the packages for modeling
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

# Load the datasets
train = pd.read_csv("adult1.csv")
test = pd.read_csv("adult1.csv")
# Seperate out predictors and target from the training data set
# Remove the ID field from the test dataset and save it.
# Drop the ID field from the training set
train_y = train['TARGET']
train_x = train
train_x.drop(['ID', 'TARGET'], axis=1, inplace=True)
test_id = test['ID']
del test['ID']


# Fixing the outliers in column 'var3'
train_x['var3'].replace(-999999,0, inplace=True)
test['var3'].replace(-999999,0, inplace=True)

# Remove all the columns which have constant values. 
# These columns have zero std deviation.
rm_col=[] 
for col in train_x.columns:
    if train_x[col].std()==0:
        rm_col.append(col)

train_x.drop(rm_col, axis=1, inplace=True)
test.drop(rm_col, axis=1, inplace=True)

# Remove the duplicate columns. 
# Here we have columns with different name but exactly same values for each rows
# We will compare each columns with all other columns
dups_col = []
for ii in range(len(train_x.columns)-1):
    for jj in range(ii+1,len(train_x.columns)):
        col1=train_x.columns[ii]
        col2=train_x.columns[jj]
        # take the columns as arrays adn then compare the values.
        if np.array_equal(train_x[col1].values, train_x[col2].values) and not col2 in dups_col:
            dups_col.append(col2)

train_x.drop(dups_col, axis=1, inplace=True)
test.drop(dups_col, axis=1, inplace=True)


# Define a classifier
rf_clf = RandomForestClassifier(max_depth=15,n_estimators=70, min_samples_leaf=50,
                                  min_samples_split=100, random_state=10)

# Train the model
rf_clf.fit(train_x,train_y)

# Plot the top 40 important features
imp_feat_rf = pd.Series(rf_clf.feature_importances_, index=train_x.columns).sort_values(ascending=False)
imp_feat_rf[:40].plot(kind='bar', title='Feature Importance with Random Forest', figsize=(12,8))
plt.ylabel('Feature Importance values')
plt.subplots_adjust(bottom=0.25)
plt.savefig('FeatImportance.png')
plt.show()

# Save indexes of the important features in descending order of their importance
indices = np.argsort(rf_clf.feature_importances_)[::-1]

# list the names of the names of top 40 selected features adn remove the unicode
select_feat =[str(s) for s in train_x.columns[indices][:50]]

# Make the subsets with 40 features only
train_x_sub = train_x[select_feat]
test_sub = test[select_feat]

#### We will use GridSearch package with cross validation to find best estimators from a list of parameters
# Define a new Random Forest Classifier
select_rf_clf = RandomForestClassifier(random_state=10)

param_grid = {
    
            'n_estimators': [50, 80, 100],
            'max_depth': [5,10, 15]
}

# we will 10-fold cross-validation
grid_clf = GridSearchCV(select_rf_clf,param_grid,cv=10)
grid_clf.fit(train_x_sub,train_y)

# Take the best model
best_rf_clf = grid_clf.best_estimator_

# Make prediction with test data
predicted_proba = best_rf_clf.predict_proba(test_sub)

submission = pd.DataFrame({'ID':test_id,'TARGET':predicted_proba[:,1]})
submission.to_csv('submission.csv', index=False)