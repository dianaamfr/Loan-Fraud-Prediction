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

def account_du():
    df = db.df_query('SELECT * FROM account')
    stats(df)
    account_distribution(df)
    
def account_distribution(df):
    # Account Dataset
    print()
    sns.countplot(x ='frequency', data = df)
    plt.savefig('data_understanding/plots/distribution/account/frequency.jpg')
    plt.clf()

    sns.histplot(df['creation_date'])
    plt.savefig('data_understanding/plots/distribution/account/date.jpg')
    plt.clf()

    df['creation_date'] = np.log(df['creation_date']) # log transformation
    sns.histplot(df['creation_date'])
    plt.savefig('data_understanding/plots/distribution/account/date_log.jpg')
    plt.clf()

def district_id_status():
    df = db.df_query('SELECT * FROM loan_train JOIN account USING(account_id)')

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.district_id.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6,
     weights=np.ones(len(df_good.district_id)) / len(df_good.district_id))

    df_bad.district_id.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.district_id)) / len(df_bad.district_id))

    ax1.set_ylim([0,0.17])
    ax2.set_ylim([0,0.17])

    ax1.set_title('Good Loan District ID')
    ax2.set_title('Bad Loan District ID')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('account')/'district_id_status.jpg')
    plt.clf()

# Days between account creation and loan issuance
def days_between_account_loan():
    df = db.df_query('SELECT * FROM loan_train JOIN account USING(account_id)')

    df['granted_date'] = df['granted_date'].apply(lambda x: int('19'+str(x)))
    df['granted_date'] = pd.to_datetime(df['granted_date'], format='%Y%m%d', errors='coerce')
    df['creation_date'] = df['creation_date'].apply(lambda x: int('19'+str(x)))
    df['creation_date'] = pd.to_datetime(df['creation_date'], format='%Y%m%d', errors='coerce')

    df['days_between_statistics'] = df['granted_date'] - df['creation_date']

    df_good = df.loc[(df['loan_status'] == 1) ]
    df_bad = df.loc[(df['loan_status'] == -1) ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    df_good.days_between_statistics.dt.days.hist(bins=20, ax=ax1, label='good', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.days_between_statistics.dt.days)) / len(df_good.days_between_statistics.dt.days))
   
    df_bad.days_between_statistics.dt.days.hist(bins=20, ax=ax2, label='bad', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.days_between_statistics.dt.days)) / len(df_bad.days_between_statistics.dt.days))

    ax1.set_ylim([0,0.15])
    ax2.set_ylim([0,0.15])

    ax1.set_title('Days Between Account Creation and Loan Issuance')
    ax2.set_title('Days Between Account Creation and Loan Issuance')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1)) 
    ax2.yaxis.set_major_formatter(PercentFormatter(1)) 

    plt.savefig(get_correlation_folder('account')/'loan_account_dates.jpg')
    plt.clf()

if __name__ == '__main__':
    Path("data_understanding/plots/distribution/account").mkdir(parents=True, exist_ok=True)
    print("### ACCOUNT ###")
    create_plots_folders('account') # TODO: move to exploration.py file
    account_du()
    district_id_status()
    days_between_account_loan()
