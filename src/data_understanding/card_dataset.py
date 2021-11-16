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

if __name__ == '__main__':
    create_plots_folders('card')
    card_train_du()
    card_test_du()