import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import os
import pickle
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
def hyperparameters(X_train, y_train, nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15), 'class_weight': [None, 'balanced']}
    # decision tree model
    dtree_model= RandomForestClassifier()
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X_train, y_train)

    return dtree_gscv.best_params_, dtree_gscv.best_score_

def plot_confusion(cf_matrix, title):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
    cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
    zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title(title);
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
    plt.savefig(f"Training_Results/RandomForest/{title}.png")
    plt.show()
    
    

def train_DT(X_train, y_train, X_test, y_test, hyper=True, title="Base"):
    if hyper:
        parameter, best_score = hyperparameters(X_train, y_train, 5)
        print(f"RandomForest {title} GridSearch parameters : {parameter}")
        print(f"RandomForest {title} GridSearch best score : {best_score}")
        clf = RandomForestClassifier(criterion=parameter['criterion'], max_depth=parameter['max_depth'] ,class_weight=parameter['class_weight'])
        
    else:
        clf = RandomForestClassifier()
    clf = clf.fit(X_train, y_train)
    os.makedirs('Saved_Model/RandomForest', exist_ok=True)
    filename = f'Saved_Model/RandomForest/RandomForest_{title}_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf

def report(y_true,y_pred,title):
    print(f"RandomForest {title} Classification Report")
    print(classification_report(y_true, y_pred))

def roc_plot(fpr, tpr, auc, title):
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.title(f"RandomForest_{title}_ROC&AUC_Testset")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig(f"Training_Results/RandomForest/RandomForest_{title}_ROC&AUC_Testset.png")
    plt.show()     

def roc_all(base, under, over):
    plt.plot(base['fpr'],base['tpr'],label="Base AUC="+str(base['auc']))
    plt.plot(under['fpr'],under['tpr'],label="Under AUC="+str(under['auc']))
    plt.plot(over['fpr'],over['tpr'],label="Over AUC="+str(over['auc']))
    plt.title("roc RandomForest")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig(f"Training_Results/RandomForest/roc_RandomForest.png")
    plt.show() 

def PrecisionRecallAll(base, under, over):
    plt.plot(base['recall'],base['precision'],label="Base")
    plt.plot(under['recall'],under['precision'],label="Under")
    plt.plot(over['recall'],over['precision'],label="Over")
    plt.title("PrecisionRecallAll RandomForest True label")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc=3)
    plt.savefig(f"Training_Results/RandomForest/PrecisionRecallAll RandomForest True label.png")
    plt.show()

def PrecisionRecall(y_test, test_pred_proba, title):
    precision_true, recall_true, thresholds_true = precision_recall_curve(y_test, test_pred_proba, pos_label=True)
    precision_false, recall_false, thresholds_false = precision_recall_curve(y_test, test_pred_proba, pos_label=False)
    plt.plot(recall_false,precision_false,label="False")
    plt.plot(recall_true,precision_true,label="True")
    plt.title(f"PrecisionRecall {title} RandomForest")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc=3)
    plt.savefig(f"Training_Results/RandomForest/PrecisionRecall {title} RandomForest True label.png")
    plt.show()

def main(load_model):
    os.chdir('..')
    os.chdir('4_Split')
    os.chdir('Process_Data')
    path = os.getcwd()
    train = pd.read_csv(path+'\\Medical_Bill_PRH_2017-2021_clean_train.csv')
    test = pd.read_csv(path+'\\Medical_Bill_PRH_2017-2021_clean_test.csv')
    os.chdir('..')
    os.chdir('..')
    os.chdir('6_Model_Training')
    os.makedirs('Training_Results/RandomForest', exist_ok=True)
    X_train = train.drop(columns=['ACCIDENT_ISSUE_CODE', 'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID','PAYMENT_AMOUNT', 'PRH'])
    y_train = train['PRH']
    X_test = test.drop(columns=['ACCIDENT_ISSUE_CODE', 'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID','PAYMENT_AMOUNT', 'PRH'])
    y_test = test['PRH']
    os.makedirs('Saved_Model/RandomForest', exist_ok=True)
    os.makedirs('Training_Results/RandomForest', exist_ok=True)
    #base
    if load_model:
        print("Use Load Model Base...")
        DT = pickle.load(open("Saved_Model/RandomForest/RandomForest_Base_model.sav", 'rb'))
    else:
        print("Training RandomForest Base ...")
        DT = train_DT(X_train, y_train, X_test, y_test, hyper=False)
    train_predict = DT.predict(X_train)
    test_predict = DT.predict(X_test)
    train_pred_proba = DT.predict_proba(X_train)[::,1]
    test_pred_proba = DT.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, test_pred_proba)
    auc = metrics.roc_auc_score(y_test, test_pred_proba)
    train_cf_matrix = confusion_matrix(y_train, train_predict)
    test_cf_matrix = confusion_matrix(y_test, test_predict)
    report(y_train,train_predict, "Base Trainset")
    report(y_test, test_predict, "Base Testset")
    plot_confusion(train_cf_matrix, f"RandomForest_Base_ConfusionMatrix_Trainset")
    plot_confusion(test_cf_matrix, f"RandomForest_Base_ConfusionMatrix_Testset")
    roc_plot(fpr,tpr,auc,title="Base")
    precision, recall, thresholds = precision_recall_curve(y_test, test_pred_proba, pos_label=True)
    PrecisionRecall(y_test, test_pred_proba, "Base")
    print("-"*50)
    
    #under
    train_under = pd.concat([train[train.PRH == True].sample(24000),train[train.PRH == False]],axis=0)
    X_train_under = train_under.drop(columns=['ACCIDENT_ISSUE_CODE', 'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID','PAYMENT_AMOUNT', 'PRH'])
    y_train_under = train_under['PRH']
    if load_model:
        print("Use Load Model Under...")
        DT_under = pickle.load(open("Saved_Model/RandomForest/RandomForest_Under_model.sav", 'rb'))
    else:
        print("Training RandomForest Under ...")
        DT_under = train_DT(X_train_under, y_train_under, X_test, y_test, title="Under")
    train_predict_under = DT_under.predict(X_train_under)
    test_predict_under = DT_under.predict(X_test)
    train_pred_proba_under = DT_under.predict_proba(X_train_under)[::,1]
    test_pred_proba_under = DT_under.predict_proba(X_test)[::,1]
    fpr_under, tpr_under, _ = metrics.roc_curve(y_test, test_pred_proba_under)
    auc_under = metrics.roc_auc_score(y_test, test_pred_proba_under)
    train_cf_matrix_under = confusion_matrix(y_train_under, train_predict_under)
    test_cf_matrix_under = confusion_matrix(y_test, test_predict_under)
    report(y_train_under,train_predict_under, "Under Trainset")
    report(y_test, test_predict_under, "Under Testset")
    plot_confusion(train_cf_matrix_under, f"RandomForest_Under_ConfusionMatrix_Trainset")
    plot_confusion(test_cf_matrix_under, f"RandomForest_Under_ConfusionMatrix_Testset")
    roc_plot(fpr_under,tpr_under,auc_under,title="Under")
    precision_under, recall_under, thresholds_under = precision_recall_curve(y_test, test_pred_proba_under, pos_label=True)
    PrecisionRecall(y_test, test_pred_proba_under, "Under")
    print("-"*50)
    
    #over
    train_over = pd.concat([train[train.PRH == False].sample(470000, replace=True),train[train.PRH == True]],axis=0)
    X_train_over = train_over.drop(columns=['ACCIDENT_ISSUE_CODE', 'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID','PAYMENT_AMOUNT', 'PRH'])
    y_train_over = train_over['PRH']
    if load_model:
        print("Use Load Model over...")
        DT_over = pickle.load(open("Saved_Model/RandomForest/RandomForest_Over_model.sav", 'rb'))
    else:
        print("Training RandomForest over ...")
        DT_over = train_DT(X_train_over, y_train_over, X_test, y_test, title="Over", hyper=False)
    train_predict_over = DT_over.predict(X_train_over)
    test_predict_over = DT_over.predict(X_test)
    train_pred_proba_over = DT_over.predict_proba(X_train_over)[::,1]
    test_pred_proba_over = DT_over.predict_proba(X_test)[::,1]
    fpr_over, tpr_over, _ = metrics.roc_curve(y_test, test_pred_proba_over)
    auc_over = metrics.roc_auc_score(y_test, test_pred_proba_over)
    train_cf_matrix_over = confusion_matrix(y_train_over, train_predict_over)
    test_cf_matrix_over = confusion_matrix(y_test, test_predict_over)
    report(y_train_over,train_predict_over, "over Trainset")
    report(y_test, test_predict_over, "over Testset")
    plot_confusion(train_cf_matrix_over, f"RandomForest_over_ConfusionMatrix_Trainset")
    plot_confusion(test_cf_matrix_over, f"RandomForest_over_ConfusionMatrix_Testset")
    roc_plot(fpr_over,tpr_over,auc_over,title="over")
    precision_over, recall_over, thresholds_over = precision_recall_curve(y_test, test_pred_proba_over, pos_label=True)
    PrecisionRecall(y_test, test_pred_proba_over, "Over")
    print("-"*50)


    base = {'fpr':fpr, 'tpr':tpr, 'auc':auc, 'precision':precision, 'recall':recall}
    under = {'fpr':fpr_under, 'tpr':tpr_under, 'auc':auc_under, 'precision':precision_under, 'recall':recall_under}
    over = {'fpr':fpr_over, 'tpr':tpr_over, 'auc':auc_over, 'precision':precision_over, 'recall':recall_over}
    roc_all(base, under, over)
    PrecisionRecallAll(base,under,over)
if __name__ == '__main__':
    main(load_model=False)
    print("Training RandomForest Model Complete !")