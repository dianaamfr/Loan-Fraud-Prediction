import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import sys
from matplotlib.ticker import PercentFormatter
from datetime import datetime

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def loan_train_du():
    df = db.df_query('SELECT * FROM loan_train')
    stats(df)
    loan_train_distribution(df.copy())
    loan_train_correlation(df.copy())
    loan_amount_status(df.copy())
    loan_duration_status(df.copy())
    loan_payments_status(df.copy())
    client_age_on_loan()
    client_gender_status()

def loan_train_distribution(df):  
    sns.histplot(df['granted_date'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_date.jpg')
    plt.clf()
    df['granted_date'] = np.log(df['granted_date']) # log transformation
    sns.histplot(df['granted_date'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_date_log.jpg')
    plt.clf()

    sns.histplot(df['amount'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_amount.jpg')
    plt.clf()
    df['amount'] = np.log(df['amount']) # log transformation
    sns.histplot(df['amount'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_amount_log.jpg')
    plt.clf()

    sns.histplot(df['duration'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_duration.jpg')
    plt.clf()
    df['duration'] = np.log(df['duration']) # log transformation
    sns.histplot(df['duration'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_duration_log.jpg')
    plt.clf()

    sns.histplot(df['payments'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_payments_log.jpg')
    plt.clf()
    df['payments'] = np.log(df['payments']) # log transformation
    sns.histplot(df['payments'])
    plt.savefig(get_distribution_folder('loan')/'loan_train_payments_log.jpg')
    plt.clf()

    sns.countplot(x ='loan_status', data = df)
    plt.savefig(get_distribution_folder('loan')/'loan_train_status.jpg')
    plt.clf()

def loan_test_du():
    df = db.df_query('SELECT * FROM loan_test')
    stats(df)
    loan_test_distribution(df)

def loan_test_distribution(df):
    sns.histplot(df['granted_date'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_date.jpg')
    plt.clf()
    df['granted_date'] = np.log(df['granted_date']) # log transformation
    sns.histplot(df['granted_date'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_date_log.jpg')
    plt.clf()

    sns.histplot(df['amount'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_amount.jpg')
    plt.clf()
    df['amount'] = np.log(df['amount']) # log transformation
    sns.histplot(df['amount'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_amount_log.jpg')
    plt.clf()

    sns.histplot(df['duration'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_duration.jpg')
    plt.clf()
    df['duration'] = np.log(df['duration']) # log transformation
    sns.histplot(df['duration'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_duration_log.jpg')
    plt.clf()

    sns.histplot(df['payments'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_payments.jpg')
    plt.clf()
    df['payments'] = np.log(df['payments']) # log transformation
    sns.histplot(df['payments'])
    plt.savefig(get_distribution_folder('loan')/'loan_test_payments_log.jpg')
    plt.clf()

def loan_train_correlation(df):
    
    # Correlation Matrix
    _, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
            df.corr(), 
            cmap = sns.diverging_palette(220, 10, as_cmap = True),
            square=True, 
            cbar=False,
            ax=ax,
            annot=True, 
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize':12 })
    plt.savefig(get_correlation_folder('loan')/'loan_train_correlation.jpg')
    plt.clf()

    # Loan Amount, Duration and Payments
    sns.relplot(data=df, y="amount", x="payments", hue="loan_status", sizes=(40, 400), alpha=.8,height=6)
    plt.savefig(get_correlation_folder('loan')/'amount_payments.jpg')
    plt.clf()

    sns.relplot(data=df, y="amount", x="duration", hue="loan_status", sizes=(40, 400), alpha=.8,height=6)
    plt.savefig(get_correlation_folder('loan')/'amount_duration.jpg')
    plt.clf()

    # Amount/Duration = Payments
    df["amount/duration"] = df["amount"] / df["duration"]
    sns.scatterplot(data=df, y="payments", x="amount/duration", hue="loan_status")
    plt.savefig(get_correlation_folder('loan')/'amount_payments_duration.jpg')
    plt.clf()

    sns.histplot(data=df, x="amount", hue="loan_status", alpha=0.6)
    plt.savefig(get_correlation_folder('loan')/'amount_status.jpg')
    plt.clf()


def loan_amount_status(df):
    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.amount.hist(bins=20, ax=ax1, label='status 1', color='#00cfccff', alpha=1.0,
     weights=np.ones(len(df_good.amount)) / len(df_good.amount))

    df_bad.amount.hist(bins=20, ax=ax2, label='status -1', color='#ff9973ff', alpha=1.0,
     weights=np.ones(len(df_bad.amount)) / len(df_bad.amount))

    ax1.set_xlim([0,550000])
    ax2.set_xlim([0,550000])

    ax1.set_ylim([0,0.16])
    ax2.set_ylim([0,0.16])

    ax1.set_title('Good Loan Amount')
    ax2.set_title('Bad Loan Amount')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1)) 
    ax2.yaxis.set_major_formatter(PercentFormatter(1)) 

    plt.savefig(get_distribution_folder('loan')/'loan_train_amount_status.jpg')
    plt.clf()

def loan_duration_status(df):
    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.duration.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6,
     weights=np.ones(len(df_good.duration)) / len(df_good.duration))

    df_bad.duration.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.duration)) / len(df_bad.duration))

    ax1.set_ylim([0,0.25])
    ax2.set_ylim([0,0.25])

    ax1.set_title('Good Loan Duration')
    ax2.set_title('Bad Loan Duration')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_distribution_folder('loan')/'loan_train_duration_status.jpg')
    plt.clf()

def loan_payments_status(df):
    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.payments.hist(bins=20, ax=ax1, label='status 1', color='#00cfccff', alpha=1.0,
     weights=np.ones(len(df_good.payments)) / len(df_good.payments))

    df_bad.payments.hist(bins=20, ax=ax2, label='status -1', color='#ff9973ff', alpha=1.0,
     weights=np.ones(len(df_bad.payments)) / len(df_bad.payments))

    ax1.set_xlim([0,10000])
    ax2.set_xlim([0,10000])

    ax1.set_ylim([0,0.11])
    ax2.set_ylim([0,0.11])

    ax1.set_title('Good Loan Payments')
    ax2.set_title('Bad Loan Payments')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_distribution_folder('loan')/'loan_train_payments_status.jpg')
    plt.clf()

def client_age_on_loan():
    df_loan = db.df_query('SELECT * FROM loan_train')
    df_account = db.df_query('SELECT * FROM account')
    df_disp = db.df_query('SELECT * FROM disposition WHERE disp_type = "OWNER"')
    df_client = db.df_query('SELECT * FROM client')

    
    # relate disposition with account and client
    merged_df = pd.merge(df_disp, df_client, how="inner",on="client_id")
    merged_df = pd.merge(merged_df, df_account, how="inner",on="account_id")
    merged_df = pd.merge(merged_df, df_loan, how="inner",on="account_id")

    for index, _ in merged_df.iterrows():
        #birth_date
        birth_str = str(merged_df['birth_number'][index])
        if int(birth_str[2:4]) < 50:
           merged_df['birth_number'][index] =datetime(int("19"+birth_str[0:2]),int(birth_str[2:4]),int(birth_str[4:6]))
        else: 
            merged_df["birth_number"][index] = datetime(int("19"+birth_str[0:2]), int(birth_str[2:4]) - 50, int(birth_str[4:6]))
        #loan_date
        merged_df["granted_date"][index] = datetime(int("19"+str(merged_df["granted_date"][index])[0:2]),int(str(merged_df["granted_date"][index])[2:4]),int(str(merged_df["granted_date"][index])[4:6]))

    merged_df['owner_age_on_loan'] = (merged_df['granted_date'] - merged_df['birth_number']).dt.days / 365
    

    sns.kdeplot( merged_df['owner_age_on_loan'], shade=True)
    plt.savefig(get_correlation_folder('loan')/'owner_age_on_loan.jpg')
    plt.clf()


def split_birth(birth_number):
    birth_str = str(birth_number)

    # Male
    if int(birth_str[2:4]) < 50:
        return 'M', int('19' + birth_str)

    # Female
    year = '19' + birth_str[0:2]
    month = str(int(birth_str[2:4]) - 50).zfill(2)
    day = birth_str[4:]
    return 'F', int(year + month + day)

def client_gender_status():
    df_loan = db.df_query('SELECT * FROM loan_train')
    df_account = db.df_query('SELECT * FROM account')
    df_disp = db.df_query('SELECT * FROM disposition')
    df_client = db.df_query('SELECT * FROM client')

    # relate disposition with account and client
    merged_df = pd.merge(df_disp, df_client, how="inner",on="client_id")
    merged_df = pd.merge(merged_df, df_account, how="inner",on="account_id")
    merged_df = pd.merge(merged_df, df_loan, how="inner",on="account_id")

    merged_df['gender'], merged_df['birth_date'] = zip(*merged_df['birth_number'].map(split_birth))

    sns.countplot(x='gender', data=merged_df,  hue="loan_status")
    plt.savefig(get_correlation_folder('loan')/'client_gender_on_loan.jpg')
    plt.clf()

if __name__ == '__main__':
    create_plots_folders('loan')
    print("### LOAN TRAIN ###")
    loan_train_du()
    print()
    print("### LOAN TEST ###")
    loan_test_du()
