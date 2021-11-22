import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path
import sys
sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')
from datetime import datetime

def trans_test_distribution():
    trans_test_df = pd.read_csv(get_files_folder()/'trans_test.csv', delimiter=";", low_memory=False)
    stats(trans_test_df)

    # Transaction Test Dataset
    sns.histplot(trans_test_df['date'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_date.jpg')
    plt.clf()

    trans_test_df['date'] = np.log(trans_test_df['date']) # log transformation
    sns.histplot(trans_test_df['date'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_date_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='type', data = trans_test_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_type.jpg')
    plt.clf()

    print()
    sns.countplot(x ='operation', data = trans_test_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_operation.jpg')
    plt.clf()


    sns.histplot(trans_test_df['amount'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_amount.jpg')
    plt.clf()

    trans_test_df['amount'] = np.log(trans_test_df['amount']) # log transformation
    sns.histplot(trans_test_df['amount'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_amount_log.jpg')
    plt.clf()

    sns.histplot(trans_test_df['balance'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_balance.jpg')
    plt.clf()

    trans_test_df['balance'] = np.log(trans_test_df['balance']) # log transformation
    sns.histplot(trans_test_df['balance'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_balance_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='k_symbol', data = trans_test_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_k_symbol.jpg')
    plt.clf()

    print()
    sns.countplot(x ='bank', data = trans_test_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_bank.jpg')
    plt.clf()

    sns.histplot(trans_test_df['account'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_account.jpg')
    plt.clf()

    trans_test_df['account'] = np.log(trans_test_df['account']) # log transformation
    sns.histplot(trans_test_df['account'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_test_account_log.jpg')
    plt.clf()

def trans_train_distribution():
    trans_train_df = pd.read_csv(get_files_folder()/'trans_train.csv', delimiter=";", low_memory=False)
    stats(trans_train_df)

    # Transaction Train Dataset
    sns.histplot(trans_train_df['date'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_date.jpg')
    plt.clf()

    trans_train_df['date'] = np.log(trans_train_df['date']) # log transformation
    sns.histplot(trans_train_df['date'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_date_log.jpg')
    plt.clf()


    print()
    sns.countplot(x ='type', data = trans_train_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_type.jpg')
    plt.clf()


    print()
    sns.countplot(x ='operation', data = trans_train_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_type_log.jpg')
    plt.clf()


    sns.histplot(trans_train_df['amount'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_amount.jpg')
    plt.clf()


    trans_train_df['amount'] = np.log(trans_train_df['amount']) # log transformation
    sns.histplot(trans_train_df['amount'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_amount_log.jpg')
    plt.clf()


    sns.histplot(trans_train_df['balance'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_balance.jpg')
    plt.clf()


    trans_train_df['balance'] = np.log(trans_train_df['balance']) # log transformation
    sns.histplot(trans_train_df['balance'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_balance_log.jpg')
    plt.clf()


    print()
    sns.countplot(x ='k_symbol', data = trans_train_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_k_symbol.jpg')
    plt.clf()


    print()
    sns.countplot(x ='bank', data = trans_train_df)
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_bank.jpg')
    plt.clf()


    sns.histplot(trans_train_df['account'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_account.jpg')
    plt.clf()


    trans_train_df['account'] = np.log(trans_train_df['account']) # log transformation
    sns.histplot(trans_train_df['account'])
    plt.savefig('data_understanding/plots/distribution/trans/trans_train_account_log.jpg')
    plt.clf()

def transactions_correlation_matrix(): 
    df = db.df_query('SELECT * FROM trans_train')

    df["trans_id"] = pd.to_numeric(df["trans_id"], errors='coerce')
    df["account_id"] = pd.to_numeric(df["account_id"], errors='coerce')
    df["trans_date"] = pd.to_numeric(df["trans_date"], errors='coerce')
    df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
    df["balance"] = pd.to_numeric(df["balance"], errors='coerce')
    df["account"] = pd.to_numeric(df["account"], errors='coerce')

    df["trans_type"] = df['trans_type'].astype('category').cat.codes
    df["operation"] = df['operation'].astype('category').cat.codes
    df["k_symbol"] = df['k_symbol'].astype('category').cat.codes
    df["bank"] = df['bank'].astype('category').cat.codes

    #for column in df:
    #    print(type(df[column][0]))

    corrMatrix = df.corr()
    
    plt.figure(figsize=(20,15))
    ax=plt.subplot(111)
    sns.heatmap(corrMatrix,ax=ax,annot=True)
    plt.savefig('data_understanding/plots/distribution/trans/correlation.jpg')
    plt.clf()

def transactions_loan_date():
    df_trans = db.df_query('SELECT * FROM trans_train')
    df_loan = db.df_query('SELECT * FROM loan_train')
    df_account = db.df_query('SELECT * FROM account')
    
    merged_df = pd.merge(df_loan, df_account, how="inner",on="account_id")
    merged_df = pd.merge(merged_df, df_trans, how="inner",on="account_id")

    for index, row in merged_df.iterrows():
        merged_df["granted_date"][index] = datetime(int("19"+str(merged_df["granted_date"][index])[0:2]),int(str(merged_df["granted_date"][index])[2:4]),int(str(merged_df["granted_date"][index])[4:6]))
        merged_df["trans_date"][index] = datetime(int("19"+str(merged_df["trans_date"][index])[0:2]),int(str(merged_df["trans_date"][index])[2:4]),int(str(merged_df["trans_date"][index])[4:6]))

    x = merged_df["granted_date"]
    y = merged_df["trans_date"]

    plt.xlabel('Loan Granted Date')
    plt.ylabel('Transaction Date')

    plt.scatter(x, y)
    plt.show()

def last_balance_loan():
    df = db.df_query('SELECT * FROM trans_train')
    df_loan = db.df_query('SELECT account_id,amount FROM loan_train')
    
    # transação mais recente feita
    df_trans = db.df_query('SELECT account_id, MAX(trans_date) AS trans_date FROM trans_train GROUP BY account_id')
   
    # visualização melhor das datas
    #for index, row in df_trans.iterrows():
    #    df_trans["trans_date"][index] = datetime(int("19"+str(df_trans["trans_date"][index])[0:2]),int(str(df_trans["trans_date"][index])[2:4]),int(str(df_trans["trans_date"][index])[4:6]))

    df_trans2 = pd.merge(df_trans, df, how="inner",on=['account_id','trans_date'])
    df_merged =pd.merge(df_trans2, df_loan, how="inner",on=['account_id'])

    print(df_merged)

    x = df_merged["balance"]
    y = df_merged["amount_y"]

    plt.xlabel('Account balance')
    plt.ylabel('Loan quantity')

    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    Path("data_understanding/plots/distribution/trans").mkdir(parents=True, exist_ok=True)
    #trans_test_distribution()
    #trans_train_distribution()
    #transactions_loan_date()
    last_balance_loan()
    