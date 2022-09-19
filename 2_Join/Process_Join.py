import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def join(df1, df3_1, df3_2):
    
    df3 = pd.merge(df3_1,df3_2,on='HOSPITAL_BILLING_INVOICE_ID')
    
    df = pd.merge(df1,df3,on='ACCIDENT_ISSUE_CODE')
    
    return df

def join_PRH(df, df_prh):
    df['ACCIDENT_ISSUE_CODE'] = df['ACCIDENT_ISSUE_CODE'].astype("string")
    df_prh['accident_id'] = df_prh['accident_id'].astype("string")
    df_prh.accident_id = df_prh.accident_id.str[:11]
    print(df_prh.accident_id.nunique())
    prh = df.ACCIDENT_ISSUE_CODE.isin(df_prh.accident_id)
        
    df['PRH'] = prh
    
    return df

def plot(df):
    
    test = pd.DataFrame()
    test['year'] = pd.DatetimeIndex(df['INFORM_DATE']).year
    test['prh'] = df.PRH
    test['id'] = df.ACCIDENT_ISSUE_CODE
    test = test.groupby(['year', 'prh']).count().reset_index()
    fig, (ax1) = plt.subplots(1, figsize=(10,10))
    fig.suptitle('prh count on year')
    ax1 = sns.barplot(x="year", y="id", hue="prh", data=test)
    plt.show()

def main():
    os.chdir('..')
    os.chdir('1_Raw')
    os.chdir('Process_Data')
    path = os.getcwd()
    df3_1 = pd.read_csv(path + '\\3.1  ข้อมูลใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล_processed.csv')
    df3_2 = pd.read_csv(path + '\\3.2  ข้อมูลรายละเอียดใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล_processed.csv')
    df1 = pd.read_csv(path + '\\1.ข้อมูลการรับแจ้งประสบอันตราย_processed.csv')
    df_prh = pd.read_csv(path + '\\PRH5011_accident_id.csv',low_memory=False)
    
    print('Processing Join')
    
    df = join(df1,df3_1,df3_2)
    
    print('Processing Join PRH')
    
    result = join_PRH(df, df_prh)
    
    plot(result)
    
    os.chdir('..')
    os.chdir('..')
    os.chdir('2_Join')
    os.makedirs('Process_Data', exist_ok=True) 
    
    result.to_csv("Process_Data/Medical_Bill_PRH_2017-2021.csv", index=False)

if __name__ == '__main__':
    
    main()
    
    print('Process Join Sucessfully!')
    