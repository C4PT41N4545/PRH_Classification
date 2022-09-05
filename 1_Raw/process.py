import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')

def file1_process():
    #load Data
    df1 = pd.read_csv("Raw_Data/งานวินิจฉัย/1.ข้อมูลการรับแจ้งประสบอันตราย.csv",sep=',',engine='python',quoting=csv.QUOTE_ALL,on_bad_lines = 'skip')
    #filter deleted data
    df1 = df1[df1.IS_DELETED == 'N']
    #list of columns
    df1_col = ['ACCIDENT_ISSUE_CODE', 'ACCIDENT_CAUSE', 'ACCIDENT_INJURY', 'INFORM_DATE',
          'BUSINESS_GROUP_CODE', 'ACCIDENT_ORGAN_SUB_GROUP', 'SEVERITY_CODE']
    #select columns
    df1 = df1[df1_col]
    #save to csv
    df1.to_csv("Process_Data/1.ข้อมูลการรับแจ้งประสบอันตราย_processed.csv",index=False)

def file2_process():
    # load Data
    df2 = pd.read_csv('Raw_Data/งานวินิจฉัย/2.1 ข้อมูลการวินิจฉัย.csv',low_memory=False)
    # Select Columns
    df2_col = ['ACCIDENT_ISSUE_CODE','INVESTIGATE_STATUS', 'APPROVAL_STATUS', 'IS_DELETED']
    # Select Approve Data
    df2 = df2[((df2.APPROVAL_STATUS == 'A') & (df2.INVESTIGATE_STATUS == 1) & (df2.IS_DELETED == "N"))]
    #select Columns
    df2 = df2[df2_col]
    #Drop Duplicate
    df2.drop_duplicates(subset=['ACCIDENT_ISSUE_CODE'])
    #save to csv
    df2.to_csv("Process_Data/2.1 ข้อมูลการวินิจฉัย_processed.csv", index=False)
    
def file3_1_process():
    # Load Data
    df3_1 = pd.read_csv("Raw_Data/งานวินิจฉัย/3.1  ข้อมูลใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล.csv", low_memory=False)
    #select columns
    df3_1_col = ['HOSPITAL_BILLING_INVOICE_ID', 'ACCIDENT_ISSUE_CODE', 'HOSPITAL_WCF_CODE', 'SEX', 'LENGTH_OF_STAY',
          'PAYMENT_AMOUNT',]
    # select non delete money
    df3_1 = df3_1[df3_1.IS_DELETED == 'N']
    #format date
    df3_1.CREATED_DATE = pd.to_datetime(df3_1.CREATED_DATE.str[:10])
    #select columns
    df3_1 = df3_1[df3_1_col]
    # save to csv
    df3_1.to_csv('Process_Data/3.1  ข้อมูลใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล_processed.csv',index=False)
    
def file3_2_process():
    #load Data
    df3_2 = pd.read_csv("Raw_Data/งานวินิจฉัย/3.2  ข้อมูลรายละเอียดใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล.csv", low_memory=False)
    #select Columns
    df3_2_col = ['HOSPITAL_BILLING_INVOICE_ID', 'HOSPITAL_BILLING_INVOICE_ITEM_ID', 'MEDICAL_FEE_ITEMS_CODE','PRICE']
    #select Columns
    df3_2 = df3_2[df3_2_col]
    # change columns value to column
    df3_2 = df3_2.pivot_table(values='PRICE', index='HOSPITAL_BILLING_INVOICE_ID', columns='MEDICAL_FEE_ITEMS_CODE', aggfunc='first')
    #reset index and drop column
    df3_2 = df3_2.reset_index().drop(columns=[0])
    # save to csv
    df3_2.to_csv('Process_Data/3.2  ข้อมูลรายละเอียดใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล_processed.csv',index=False)

def PRH5011_Process():
    #set file path
    mypath = "Raw_Data/PRH5011/"
    # get list of file name
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # select file name
    #onlyfiles = ["PRH5011_2560.xlsx", "PRH5011_2561.xlsx", "PRH5011_2562.xlsx", "PRH5011_2563.xlsx", "PRH5011_2564.xlsx"]
    accident_id = []
    #loop through file
    for file in onlyfiles:
        path = mypath+str(file)
        # read excel file
        df = pd.read_excel(path,header=4, engine="openpyxl")
        #drop null value
        df = df.dropna(subset=["เลขที่ประสบอันตราย"])
        #get Accident code
        temp = df['เลขที่ประสบอันตราย']
        # format Accident code
        temp = [s.replace('/' , '') for s in temp]
        print(file, str(len(temp)))
        # append to list
        accident_id += temp
    print("total length: {}".format(len(accident_id)))
    #create DataFrame
    accident_id_df = pd.DataFrame(accident_id, columns=["accident_id"])
    #save to csv
    accident_id_df.to_csv("Process_Data/PRH5011_accident_id.csv", index=False)
    
def main():
    print("Processing 1.ข้อมูลการรับแจ้งประสบอันตราย.csv")
    file1_process()
    print("Processing 2.1 ข้อมูลการวินิจฉัย.csv")
    file2_process()
    print("Processing 3.1  ข้อมูลใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล.csv")
    file3_1_process()
    print("Processing 3.2  ข้อมูลรายละเอียดใบแจ้งหนี้-ใบเสร็จค่ารักษาพยาบาล")
    file3_2_process()
    print("Processing PRH5011")
    PRH5011_Process()
    print("Raw Data Processed!")
    
if __name__ == "__main__":
    os.makedirs('Process_Data', exist_ok=True) 
    main()