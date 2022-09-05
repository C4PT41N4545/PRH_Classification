import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
import os

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


clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=None)


def classification(X_train, y_train, X_test, y_test):
    clf_n = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("----------------------")
    print("Accuracy = %f " % clf.score(X_test, y_test))
    print("----------------------")
    print(classification_report(y_test, y_pred))


classification(X_train, y_train, X_test, y_test)
