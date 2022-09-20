import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
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
    model= KNeighborsClassifier()
    #use gridsearch to test all values
    gscv = GridSearchCV(model, param_grid, cv=nfolds)
    #fit model to data
    gscv.fit(X_train, y_train)
    print(gscv.best_params_)
    print(gscv.best_score_)
    return gscv.best_params_

#parameters = hyperparameters(X_train, y_train, nfolds = 5)
parameters = {"n_neighbors":13}
nbrs = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'])

nbrs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)
print("----------------------")
print("Accuracy = %f " % nbrs.score(X_test, y_test))
print("----------------------")
print(classification_report(y_test, y_pred))
y_pred_proba = nbrs.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
def roc_plot(fpr, tpr, auc):
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
roc_plot(fpr,tpr,auc)
cf_matrix = confusion_matrix(y_test, y_pred)
def plot_confusion(cf_matrix):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
    cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
    zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()
plot_confusion(cf_matrix)
