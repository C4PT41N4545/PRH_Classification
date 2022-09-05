import pandas as pd
import numpy  as np
import os
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

def preprocess(train, test):
    
    train = train.drop(columns=['ACCIDENT_ISSUE_CODE', 'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID'])
    test = test.drop(columns=['ACCIDENT_ISSUE_CODE', 'INFORM_DATE', 'HOSPITAL_BILLING_INVOICE_ID'])
    
    df_col = train.columns
    
    scaler = MinMaxScaler()
    
    scaler.fit(train)
    
    train_scaled = pd.DataFrame(scaler.transform(train), columns=df_col)
    
    test_scaled = pd.DataFrame(scaler.transform(test), columns=df_col)
    

    return train_scaled, test_scaled, scaler

def main():
    
    os.chdir('..')
    os.chdir('4_Split')
    os.chdir('Process_Data')
    path = os.getcwd()
    train = pd.read_csv(path + '\\Medical_Bill_PRH_2017-2021_clean_train.csv')
    test = pd.read_csv(path + '\\Medical_Bill_PRH_2017-2021_clean_test.csv')
    
    train_scaled, test_scaled, scaler = preprocess(train, test)
    os.chdir('..')
    os.chdir('..')
    os.chdir('5_Preprocessed')
    os.makedirs('Process_Data', exist_ok=True) 
    dump(scaler, open('scaler.pkl', 'wb'))
    train_scaled.to_csv("Process_Data/Medical_Bill_PRH_2017-2021_clean_train_scaled.csv", index=False)
    test_scaled.to_csv("Process_Data/Medical_Bill_PRH_2017-2021_clean_test_scaled.csv", index=False)
    
if __name__ == '__main__':
    main()
    print('Preprocessing Complete !')
    