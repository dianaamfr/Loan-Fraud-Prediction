import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

def trans_test_du():
    trans_test_df = pd.read_csv(get_files_folder()/'trans_test.csv', delimiter=";", low_memory=False)
    stats(trans_test_df)

    # Transaction Test Dataset
    sns.histplot(trans_test_df['date'])
    plt.savefig('plots/distribution/trans/trans_test_date.jpg')
    plt.clf()

    trans_test_df['date'] = np.log(trans_test_df['date']) # log transformation
    sns.histplot(trans_test_df['date'])
    plt.savefig('plots/distribution/trans/trans_test_date_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='type', data = trans_test_df)
    plt.savefig('plots/distribution/trans/trans_test_type.jpg')
    plt.clf()

    print()
    sns.countplot(x ='operation', data = trans_test_df)
    plt.savefig('plots/distribution/trans/trans_test_operation.jpg')
    plt.clf()


    sns.histplot(trans_test_df['amount'])
    plt.savefig('plots/distribution/trans/trans_test_amount.jpg')
    plt.clf()

    trans_test_df['amount'] = np.log(trans_test_df['amount']) # log transformation
    sns.histplot(trans_test_df['amount'])
    plt.savefig('plots/distribution/trans/trans_test_amount_log.jpg')
    plt.clf()

    sns.histplot(trans_test_df['balance'])
    plt.savefig('plots/distribution/trans/trans_test_balance.jpg')
    plt.clf()

    trans_test_df['balance'] = np.log(trans_test_df['balance']) # log transformation
    sns.histplot(trans_test_df['balance'])
    plt.savefig('plots/distribution/trans/trans_test_balance_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='k_symbol', data = trans_test_df)
    plt.savefig('plots/distribution/trans/trans_test_k_symbol.jpg')
    plt.clf()

    print()
    sns.countplot(x ='bank', data = trans_test_df)
    plt.savefig('plots/distribution/trans/trans_test_bank.jpg')
    plt.clf()

    sns.histplot(trans_test_df['account'])
    plt.savefig('plots/distribution/trans/trans_test_account.jpg')
    plt.clf()

    trans_test_df['account'] = np.log(trans_test_df['account']) # log transformation
    sns.histplot(trans_test_df['account'])
    plt.savefig('plots/distribution/trans/trans_test_account_log.jpg')
    plt.clf()

def trans_train_du():
    trans_train_df = pd.read_csv(get_files_folder()/'trans_train.csv', delimiter=";", low_memory=False)
    stats(trans_train_df)

    # Transaction Train Dataset
    sns.histplot(trans_train_df['date'])
    plt.savefig('plots/distribution/trans/trans_train_date.jpg')
    plt.clf()

    trans_train_df['date'] = np.log(trans_train_df['date']) # log transformation
    sns.histplot(trans_train_df['date'])
    plt.savefig('plots/distribution/trans/trans_train_date_log.jpg')
    plt.clf()


    print()
    sns.countplot(x ='type', data = trans_train_df)
    plt.savefig('plots/distribution/trans/trans_train_type.jpg')
    plt.clf()


    print()
    sns.countplot(x ='operation', data = trans_train_df)
    plt.savefig('plots/distribution/trans/trans_train_type_log.jpg')
    plt.clf()


    sns.histplot(trans_train_df['amount'])
    plt.savefig('plots/distribution/trans/trans_train_amount.jpg')
    plt.clf()


    trans_train_df['amount'] = np.log(trans_train_df['amount']) # log transformation
    sns.histplot(trans_train_df['amount'])
    plt.savefig('plots/distribution/trans/trans_train_amount_log.jpg')
    plt.clf()


    sns.histplot(trans_train_df['balance'])
    plt.savefig('plots/distribution/trans/trans_train_balance.jpg')
    plt.clf()


    trans_train_df['balance'] = np.log(trans_train_df['balance']) # log transformation
    sns.histplot(trans_train_df['balance'])
    plt.savefig('plots/distribution/trans/trans_train_balance_log.jpg')
    plt.clf()


    print()
    sns.countplot(x ='k_symbol', data = trans_train_df)
    plt.savefig('plots/distribution/trans/trans_train_k_symbol.jpg')
    plt.clf()


    print()
    sns.countplot(x ='bank', data = trans_train_df)
    plt.savefig('plots/distribution/trans/trans_train_bank.jpg')
    plt.clf()


    sns.histplot(trans_train_df['account'])
    plt.savefig('plots/distribution/trans/trans_train_account.jpg')
    plt.clf()


    trans_train_df['account'] = np.log(trans_train_df['account']) # log transformation
    sns.histplot(trans_train_df['account'])
    plt.savefig('plots/distribution/trans/trans_train_account_log.jpg')
    plt.clf()


    
if __name__ == '__main__':
    Path("plots/distribution/trans").mkdir(parents=True, exist_ok=True)
    trans_test_du()
    trans_train_du()