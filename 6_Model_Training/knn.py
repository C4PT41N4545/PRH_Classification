import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

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

nbrs = KNeighborsClassifier(n_neighbors=5)

nbrs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)
print("----------------------")
print("Accuracy = %f " % nbrs.score(X_test, y_test))
print("----------------------")
print(classification_report(y_test, y_pred))
