import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
def clean(df):
    
    df = df[df.INFORM_DATE > '2016-12-31']
    
    df = df.fillna(0)
    
    df = df[((df.SEX == 'M') | (df.SEX == 'F'))]
    
    df = pd.get_dummies(df, columns=['SEX'])
    
    return df

def plot(df):
    df.PRH.value_counts().plot(kind='bar',title='PRH Value Counts')
    plt.show()
    df.ACCIDENT_CAUSE.value_counts().plot(kind='bar',title='ACCIDENT CAUSE Value Counts')
    plt.show()

def main(df):
    
    df = clean(df)
    
    os.chdir('..')
    os.chdir('..')
    os.chdir('3_Clean')
    os.makedirs('Process_Data', exist_ok=True) 
    
    df.to_csv("Process_Data/Medical_Bill_PRH_2017-2021_clean.csv", index=False)
    
    plot(df)

if __name__ == '__main__':
    os.chdir('..')
    os.chdir('2_Join')
    os.chdir('Process_Data')
    path = os.getcwd()
    df = pd.read_csv(path+'\\Medical_Bill_PRH_2017-2021.csv')
    main(df)
    print("Process Cleansing Successfuly!")
