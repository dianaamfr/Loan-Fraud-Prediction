import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
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

    df_good_series = df_good['card_type'].value_counts()
    df_bad_series = df_bad['card_type'].value_counts()
    
    df_good_count = df_good_series.to_frame('total')
    df_bad_count = df_bad_series.to_frame('total')
    aux_bad = pd.DataFrame([[0], [0], [0]], columns = ['Total'], index=['classic', 'gold', 'junior'])
    df_bad_count = df_bad_count.append(aux_bad)

    x_axis = np.arange(df['card_type'].nunique())

    _, ax = plt.subplots(figsize=(16, 6))

    x_ticks = [str(col) for col in df['card_type'].value_counts().index]

    plt.bar(x_axis - 0.2, df_good_count['total']/len(df_good['loan_status']), 0.4, label = 'status 1', color='#00cfccff', alpha=1.0)
    plt.bar(x_axis + 0.2, df_bad_count['total']/len(df_bad['loan_status']), 0.4, label = 'status -1', color='#ff9973ff', alpha=1.0)

    plt.xticks(x_axis, x_ticks)
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