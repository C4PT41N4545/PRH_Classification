import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.model_selection import GridSearchCV
os.chdir('..')
os.chdir('5_Preprocessed')
os.chdir('Process_Data')
path = os.getcwd()
train = pd.read_csv(path+'\\Medical_Bill_PRH_2017-2021_clean_train_scaled.csv')
test = pd.read_csv(path+'\\Medical_Bill_PRH_2017-2021_clean_test_scaled.csv')

X_train = train.drop(columns=['PRH'])
y_train = train['PRH']
X_test = test.drop(columns=['PRH'])
y_test = test['PRH']

def hyperparameters(X_train, y_train, nfolds):
    #create a dictionary of all values we want to test
    param_grid = {'n_neighbors': np.arange(3, 15)}
    # decision tree model
    dtree_model= KNeighborsClassifier()
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X_train, y_train)
    print(dtree_gscv.best_params_)
    print(dtree_gscv.best_score_)
    return dtree_gscv.best_params_

parameters = hyperparameters(X_train, y_train, nfolds = 5)
print(parameters)
nbrs = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'])

nbrs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)
print("----------------------")
print("Accuracy = %f " % nbrs.score(X_test, y_test))
print("----------------------")
print(classification_report(y_test, y_pred))
