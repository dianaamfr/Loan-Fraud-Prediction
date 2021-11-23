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

def card_train_du():
    df = db.df_query('SELECT * FROM card_train')
    stats(df)
    card_distribution(df)

def card_distribution(df):
    sns.countplot(x ='card_type', data=df)
    plt.savefig(get_distribution_folder('card')/'card_train_type.jpg')
    plt.clf()
    
    sns.histplot(df['issued'])
    plt.savefig(get_distribution_folder('card')/'card_train_issued.jpg')
    plt.clf()

    df['issued'] = np.log(df['issued']) # log transformation
    sns.histplot(df['issued'])
    plt.savefig(get_distribution_folder('card')/'card_train_issued_log.jpg')
    plt.clf()

def card_test_du():
    df = db.df_query('SELECT * FROM card_test')
    stats(df)

    sns.countplot(x ='card_type', data = df)
    plt.savefig(get_distribution_folder('card')/'card_test_type.jpg')
    plt.clf()

    sns.histplot(df['issued'])
    plt.savefig(get_distribution_folder('card')/'card_test_issued.jpg')
    plt.clf()

    df['issued'] = np.log(df['issued']) # log transformation
    sns.histplot(df['issued'])
    plt.savefig(get_distribution_folder('card')/'card_test_issued_log.jpg')
    plt.clf()

def card_type_status():
    df = db.df_query('SELECT account_id, card_id, card_type, loan_status FROM loan_train JOIN disposition USING(account_id) LEFT JOIN card_train USING(disp_id)')

    df.loc[df['card_type'].isna(), 'card_type'] = 'no_card'

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    x_axis = np.arange(df['card_type'].nunique())

    fig, ax = plt.subplots(figsize=(16, 6)) # 7, 6

    print("X_AXIS")
    print(x_axis)
    print("GOOOOOOD")
    print(len(df_good))
    print("BAAAAAAD")
    print(len(df_bad))
    print("GOOOOOOD VALUESSSSS")
    print(df_good['card_type'].value_counts())
    print("BAAAAAAD VALUESSSSS")
    print(df_bad['card_type'].value_counts())
    # print(df_bad.loc[df_bad['loan_status'] == -1])

    plt.bar(x_axis - 0.2, df_good['card_type'].value_counts()/len(df_good), 0.4, label = 'status 1', color='green', alpha=0.6)
    plt.bar(x_axis + 0.2, df_bad['card_type'].value_counts()/len(df_bad), 0.4, label = 'status -1', color='red', alpha=0.6)

    # apagar:
    # plt.bar(x_axis - 0.2, df_good['card_type'].value_counts(), 0.4, label = 'status 1', color='green', alpha=0.6)
    # plt.bar(x_axis + 0.2, df_bad['card_type'].value_counts(), 0.4, label = 'status -1', color='red', alpha=0.6)

    plt.xticks(x_axis, df['card_type'].unique())
    plt.xlabel("type", labelpad=10)
    plt.ylabel("count", labelpad=10)
    plt.title("Card Type Count")
    plt.legend()
     
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('card')/'card_type_status.jpg')
    plt.clf()

if __name__ == '__main__':
    create_plots_folders('card')
    card_train_du()
    card_test_du()
    card_type_status()