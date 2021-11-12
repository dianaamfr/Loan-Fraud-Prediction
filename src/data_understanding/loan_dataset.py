import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

def loan_train_du():
    df = pd.read_csv(get_files_folder()/'loan_train.csv', delimiter=";", low_memory=False)
    # df = db.df_query('SELECT * FROM loan_train') # TODO - get database
    stats(df)
    loan_train_distribution(df)
    loan_train_correlation(df)
    loan_amount_status(df)


def loan_train_distribution(df):  
    sns.histplot(df['date'])
    plt.savefig('plots/distribution/loan/loan_train_date.jpg')
    plt.clf()
    df['date'] = np.log(df['date']) # log transformation
    sns.histplot(df['date'])
    plt.savefig('plots/distribution/loan/loan_train_date_log.jpg')
    plt.clf()

    sns.histplot(df['amount'])
    plt.savefig('plots/distribution/loan/loan_train_amount.jpg')
    plt.clf()
    df['amount'] = np.log(df['amount']) # log transformation
    sns.histplot(df['amount'])
    plt.savefig('plots/distribution/loan/loan_train_amount_log.jpg')
    plt.clf()

    sns.histplot(df['duration'])
    plt.savefig('plots/distribution/loan/loan_train_duration.jpg')
    plt.clf()
    df['duration'] = np.log(df['duration']) # log transformation
    sns.histplot(df['duration'])
    plt.savefig('plots/distribution/loan/loan_train_duration_log.jpg')
    plt.clf()

    sns.histplot(df['payments'])
    plt.savefig('plots/distribution/loan/loan_train_payments_log.jpg')
    plt.clf()
    df['payments'] = np.log(df['payments']) # log transformation
    sns.histplot(df['payments'])
    plt.savefig('plots/distribution/loan/loan_train_payments_log.jpg')
    plt.clf()

    sns.countplot(x ='status', data = df)
    plt.savefig('plots/distribution/loan/loan_train_status.jpg')
    plt.clf()

def loan_test_du():
    df = pd.read_csv(get_files_folder()/'loan_test.csv', delimiter=";", low_memory=False)
    stats(df)
    loan_test_distribution(df)

def loan_test_distribution(df):
    sns.histplot(df['date'])
    plt.savefig('plots/distribution/loan/loan_test_date.jpg')
    plt.clf()
    df['date'] = np.log(df['date']) # log transformation
    sns.histplot(df['date'])
    plt.savefig('plots/distribution/loan/loan_test_date_log.jpg')
    plt.clf()

    sns.histplot(df['amount'])
    plt.savefig('plots/distribution/loan/loan_test_amount.jpg')
    plt.clf()
    df['amount'] = np.log(df['amount']) # log transformation
    sns.histplot(df['amount'])
    plt.savefig('plots/distribution/loan/loan_test_amount_log.jpg')
    plt.clf()

    sns.histplot(df['duration'])
    plt.savefig('plots/distribution/loan/loan_test_duration.jpg')
    plt.clf()
    df['duration'] = np.log(df['duration']) # log transformation
    sns.histplot(df['duration'])
    plt.savefig('plots/distribution/loan/loan_test_duration_log.jpg')
    plt.clf()

    sns.histplot(df['payments'])
    plt.savefig('plots/distribution/loan/loan_test_payments.jpg')
    plt.clf()
    df['payments'] = np.log(df['payments']) # log transformation
    sns.histplot(df['payments'])
    plt.savefig('plots/distribution/loan/loan_test_payments_log.jpg')
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
    plt.savefig('plots/correlation/loan/loan_train_correlation.jpg')
    plt.clf()

def loan_amount_status(df):
    # Amount  
    df_good = df.loc[(df['status'] == 1) ]
    df_bad = df.loc[(df['status'] == -1) ]

    print("Number of status 1: ",df_good.shape[0])
    print("Number of status -1: ",df_bad.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    df_good.amount.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6)
    df_bad.amount.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6)
    ax1.set_title('Loan Amount')
    ax2.set_title('Loan Amount')
    ax1.legend()
    ax2.legend()
    plt.savefig('plots/distribution/loan/loan_train_amount_status.jpg')
    plt.clf()

if __name__ == '__main__':
    Path("plots/distribution/loan").mkdir(parents=True, exist_ok=True)
    Path("plots/correlation/loan").mkdir(parents=True, exist_ok=True)
    print("### LOAN TRAIN ###")
    loan_train_du()
    print()
    print("### LOAN TEST ###")
    loan_test_du()
    



