import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path
import sys

sys.path.insert(1, '.')

from database import database
from datetime import datetime

db = database.Database('bank_database')

def loan_train_du():
    df = db.df_query('SELECT * FROM loan_train')
   # stats(df)
   # loan_train_distribution(df)
   # loan_train_correlation(df)
   # loan_amount_status(df)
    client_age_on_loan()

def loan_train_distribution(df):  
    sns.histplot(df['granted_date'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_date.jpg')
    plt.clf()
    df['granted_date'] = np.log(df['granted_date']) # log transformation
    sns.histplot(df['granted_date'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_date_log.jpg')
    plt.clf()

    sns.histplot(df['amount'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_amount.jpg')
    plt.clf()
    df['amount'] = np.log(df['amount']) # log transformation
    sns.histplot(df['amount'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_amount_log.jpg')
    plt.clf()

    sns.histplot(df['duration'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_duration.jpg')
    plt.clf()
    df['duration'] = np.log(df['duration']) # log transformation
    sns.histplot(df['duration'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_duration_log.jpg')
    plt.clf()

    sns.histplot(df['payments'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_payments_log.jpg')
    plt.clf()
    df['payments'] = np.log(df['payments']) # log transformation
    sns.histplot(df['payments'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_payments_log.jpg')
    plt.clf()

    sns.countplot(x ='loan_status', data = df)
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_status.jpg')
    plt.clf()

def loan_test_du():
    df = db.df_query('SELECT * FROM loan_test')
    stats(df)
    loan_test_distribution(df)

def loan_test_distribution(df):
    sns.histplot(df['granted_date'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_date.jpg')
    plt.clf()
    df['granted_date'] = np.log(df['granted_date']) # log transformation
    sns.histplot(df['granted_date'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_date_log.jpg')
    plt.clf()

    sns.histplot(df['amount'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_amount.jpg')
    plt.clf()
    df['amount'] = np.log(df['amount']) # log transformation
    sns.histplot(df['amount'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_amount_log.jpg')
    plt.clf()

    sns.histplot(df['duration'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_duration.jpg')
    plt.clf()
    df['duration'] = np.log(df['duration']) # log transformation
    sns.histplot(df['duration'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_duration_log.jpg')
    plt.clf()

    sns.histplot(df['payments'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_payments.jpg')
    plt.clf()
    df['payments'] = np.log(df['payments']) # log transformation
    sns.histplot(df['payments'])
    plt.savefig('data_understanding/plots/distribution/loan/loan_test_payments_log.jpg')
    plt.clf()

def loan_train_correlation(df):
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(
            df.corr(), 
            cmap = sns.diverging_palette(220, 10, as_cmap = True),
            square=True, 
            cbar=False,
            ax=ax,
            annot=True, 
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize':12 })
    plt.savefig('data_understanding/plots/correlation/loan/loan_train_correlation.jpg')
    plt.clf()

def loan_amount_status(df):
    # Amount  
    df_good = df.loc[(df['loan_status'] == 1) ]
    df_bad = df.loc[(df['loan_status'] == -1) ]

    print("Number of status 1: ",df_good.shape[0])
    print("Number of status -1: ",df_bad.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    df_good.amount.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6)
    df_bad.amount.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6)
    ax1.set_title('Loan Amount')
    ax2.set_title('Loan Amount')
    ax1.legend()
    ax2.legend()
    plt.savefig('data_understanding/plots/distribution/loan/loan_train_amount_status.jpg')
    plt.clf()

def client_age_on_loan():
    df_loan = db.df_query('SELECT * FROM loan_train')
    df_account = db.df_query('SELECT * FROM account')
    df_disp = db.df_query('SELECT * FROM disposition')
    df_client = db.df_query('SELECT * FROM client')

    
    # relacionar disp with account and client
    merged_df = pd.merge(df_disp, df_client, how="inner",on="client_id")
    merged_df = pd.merge(merged_df, df_account, how="inner",on="account_id")
    merged_df = pd.merge(merged_df, df_loan, how="inner",on="account_id")

    for index, row in merged_df.iterrows(): 
        #birthDate
        birth_str = str(merged_df['birth_number'][index])
        if int(birth_str[2:4]) < 50:
           merged_df['birth_number'][index] =datetime(int("19"+birth_str[0:2]),int(birth_str[2:4]),int(birth_str[4:6]))
        else: 
            merged_df["birth_number"][index] = datetime(int("19"+birth_str[0:2]), int(birth_str[2:4]) - 50, int(birth_str[4:6]))
        #loan_date
        merged_df["granted_date"][index] = datetime(int("19"+str(merged_df["granted_date"][index])[0:2]),int(str(merged_df["granted_date"][index])[2:4]),int(str(merged_df["granted_date"][index])[4:6]))

    merged_df['client_age_on_loan'] = (merged_df['granted_date'] - merged_df['birth_number']).dt.days / 365
    

    sns.kdeplot( merged_df['client_age_on_loan'], shade=True)
    plt.show()
    plt.savefig('data_understanding/plots/correlation/loan/client_age_on_loan.jpg')
    plt.clf()

if __name__ == '__main__':
    Path("data_understanding/plots/distribution/loan").mkdir(parents=True, exist_ok=True)
    Path("data_understanding/plots/correlation/loan").mkdir(parents=True, exist_ok=True)
    #print("### LOAN TRAIN ###")
    loan_train_du()
    #print()
    #print("### LOAN TEST ###")
    #loan_test_du()
    
