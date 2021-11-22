import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path
import sys
from matplotlib.ticker import PercentFormatter

sys.path.insert(1, '.')

from database import database

db = database.Database('bank_database')

create_plots_folders('trans')

def trans_test_du():
    trans_test_df = db.df_query('SELECT * FROM trans_test')
    stats(trans_test_df)

    # Transaction Test Dataset
    sns.histplot(trans_test_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_date.jpg')
    plt.clf()

    trans_test_df['trans_date'] = np.log(trans_test_df['trans_date']) # log transformation
    sns.histplot(trans_test_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_date_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='trans_type', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_type.jpg')
    plt.clf()

    print()
    sns.countplot(x ='operation', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_operation.jpg')
    plt.clf()

    sns.histplot(trans_test_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_amount.jpg')
    plt.clf()

    trans_test_df['amount'] = np.log(trans_test_df['amount']) # log transformation
    sns.histplot(trans_test_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_amount_log.jpg')
    plt.clf()

    sns.histplot(trans_test_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_balance.jpg')
    plt.clf()

    trans_test_df['balance'] = np.log(trans_test_df['balance']) # log transformation
    sns.histplot(trans_test_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_balance_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='k_symbol', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_k_symbol.jpg')
    plt.clf()

    print()
    sns.countplot(x ='bank', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_bank.jpg')
    plt.clf()

    sns.histplot(trans_test_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_account.jpg')
    plt.clf()

    trans_test_df['account'] = np.log(trans_test_df['account']) # log transformation
    sns.histplot(trans_test_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_account_log.jpg')
    plt.clf()

def trans_train_du():
    trans_train_df = db.df_query('SELECT * FROM trans_train')
    stats(trans_train_df)

    # Transaction Train Dataset
    sns.histplot(trans_train_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_date.jpg')
    plt.clf()

    trans_train_df['trans_date'] = np.log(trans_train_df['trans_date']) # log transformation
    sns.histplot(trans_train_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_date_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='trans_type', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_type.jpg')
    plt.clf()

    print()
    sns.countplot(x ='operation', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_type_log.jpg')
    plt.clf()

    sns.histplot(trans_train_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_amount.jpg')
    plt.clf()

    trans_train_df['amount'] = np.log(trans_train_df['amount']) # log transformation
    sns.histplot(trans_train_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_amount_log.jpg')
    plt.clf()

    sns.histplot(trans_train_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_balance.jpg')
    plt.clf()

    trans_train_df['balance'] = np.log(trans_train_df['balance']) # log transformation
    sns.histplot(trans_train_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_balance_log.jpg')
    plt.clf()

    print()
    sns.countplot(x ='k_symbol', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_k_symbol.jpg')
    plt.clf()

    print()
    sns.countplot(x ='bank', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_bank.jpg')
    plt.clf()

    sns.histplot(trans_train_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_account.jpg')
    plt.clf()

    trans_train_df['account'] = np.log(trans_train_df['account']) # log transformation
    sns.histplot(trans_train_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_account_log.jpg')
    plt.clf()
    
def num_trans_status():
    df_loans = db.df_query('SELECT * FROM loan_train')
    df_trans = db.df_query('SELECT account_id, COUNT(*) AS num_trans FROM trans_train WHERE account_id IN (SELECT DISTINCT account_id FROM loan_train) GROUP BY account_id ORDER BY account_id')

    df = pd.merge(df_loans, df_trans, how="inner",on="account_id")

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.num_trans.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.num_trans)) / len(df_good.num_trans))
   
    df_bad.num_trans.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.num_trans)) / len(df_bad.num_trans))

    ax1.set_ylim([0,0.16])
    ax2.set_ylim([0,0.16])

    ax1.set_title('Number of transactions of the accounts')
    ax2.set_title('Number of transactions of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1)) 
    ax2.yaxis.set_major_formatter(PercentFormatter(1)) 

    plt.savefig(get_correlation_folder('trans')/'num_trans_status.jpg')
    plt.clf()

def avg_amount_status():
    df_loans = db.df_query('SELECT * FROM loan_train')
    df_trans = db.df_query('SELECT account_id, AVG(amount) AS avg_amount FROM trans_train WHERE account_id IN (SELECT DISTINCT account_id FROM loan_train) GROUP BY account_id ORDER BY account_id')

    df = pd.merge(df_loans, df_trans, how="inner",on="account_id")

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.avg_amount.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.avg_amount)) / len(df_good.avg_amount))
   
    df_bad.avg_amount.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.avg_amount)) / len(df_bad.avg_amount))

    ax1.set_xlim([0,25000])
    ax2.set_xlim([0,25000])
    ax1.set_ylim([0,0.14])
    ax2.set_ylim([0,0.14])

    ax1.set_title('Average transaction amount of the accounts')
    ax2.set_title('Average transaction amount of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('trans')/'avg_amount_status.jpg')
    plt.clf()

def avg_balance_status():
    df_loans = db.df_query('SELECT * FROM loan_train')
    df_trans = db.df_query('SELECT account_id, AVG(balance) AS avg_balance FROM trans_train WHERE account_id IN (SELECT DISTINCT account_id FROM loan_train) GROUP BY account_id ORDER BY account_id')

    df = pd.merge(df_loans, df_trans, how="inner",on="account_id")

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.avg_balance.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.avg_balance)) / len(df_good.avg_balance))
   
    df_bad.avg_balance.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.avg_balance)) / len(df_bad.avg_balance))

    ax1.set_xlim([10000,80000])
    ax2.set_xlim([10000,80000])
    ax1.set_ylim([0,0.18])
    ax2.set_ylim([0,0.18])

    ax1.set_title('Average transaction balance of the accounts')
    ax2.set_title('Average transaction balance of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('trans')/'avg_balance_status.jpg')
    plt.clf()

if __name__ == '__main__':
    Path("plots/distribution/trans").mkdir(parents=True, exist_ok=True)
    trans_test_du()
    trans_train_du()
    num_trans_status()
    avg_amount_status()
    avg_balance_status()