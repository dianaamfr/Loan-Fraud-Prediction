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

if __name__ == '__main__':
    Path("data_understanding/plots/distribution/account").mkdir(parents=True, exist_ok=True)
    print("### ACCOUNT ###")
    account_du()