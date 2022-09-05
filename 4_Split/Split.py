import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def Split(df):
    df['100000'] = df['PAYMENT_AMOUNT']
    
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    
    return X_train, X_test

def main():
    os.chdir('..')
    os.chdir('3_Clean')
    os.chdir('Process_Data')
    path = os.getcwd()
    df = pd.read_csv(path + '\\Medical_Bill_PRH_2017-2021_clean.csv')
    
    X_train, X_test = Split(df)
    
    os.chdir('..')
    os.chdir('..')
    os.chdir('4_Split')
    os.makedirs('Process_Data', exist_ok=True) 
    
    X_train.to_csv("Process_Data/Medical_Bill_PRH_2017-2021_clean_train.csv", index=False)
    X_test.to_csv("Process_Data/Medical_Bill_PRH_2017-2021_clean_test.csv", index=False)

if __name__ == '__main__':
    
    main()
    
    print("Process Split Complete!")
    
    
    