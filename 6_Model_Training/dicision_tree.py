import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os
from sklearn import metrics
import matplotlib.pyplot as plt

os.chdir('..')
os.chdir('4_Split')
os.chdir('Process_Data')
path = os.getcwd()
train = pd.read_csv(path+'\\Medical_Bill_PRH_2017-2021_clean_train.csv')
test = pd.read_csv(path+'\\Medical_Bill_PRH_2017-2021_clean_test.csv')

X_train = train.drop(columns=['ACCIDENT_ISSUE_CODE',
                              'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID', 'PRH'])
y_train = train['PRH']

X_test = test.drop(columns=['ACCIDENT_ISSUE_CODE',
                            'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID', 'PRH'])
y_test = test['PRH']



def hyperparameters(X_train, y_train, nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
    # decision tree model
    dtree_model= tree.DecisionTreeClassifier()
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X_train, y_train)
    print(dtree_gscv.best_params_)
    print(dtree_gscv.best_score_)
    return dtree_gscv.best_params_


def classification(X_train, y_train, X_test, y_test, parameter):
    clf = tree.DecisionTreeClassifier(criterion = parameter['criterion'], max_depth = parameter['max_depth'])
    clf_n = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("----------------------")
    print("Accuracy = %f " % clf.score(X_test, y_test))
    print("----------------------")
    print(classification_report(y_test, y_pred))
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

parameter = hyperparameters(X_train, y_train, nfolds=5)
#parameter = {"criterion" : 'entropy', "max_depth":9}
classification(X_train, y_train, X_test, y_test, parameter)
