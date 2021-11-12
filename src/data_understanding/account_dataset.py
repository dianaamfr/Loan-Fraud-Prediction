import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

def account_du():
    df = pd.read_csv(get_files_folder()/'account.csv', delimiter=";", low_memory=False)
    stats(df)
    account_distribution(df)
    
def account_distribution(df):
    
    # Account Dataset
    print()
    sns.countplot(x ='frequency', data = df)
    plt.savefig('plots/distribution/account/frequency.jpg')
    plt.clf()

    sns.histplot(df['date'])
    plt.savefig('plots/distribution/account/date.jpg')
    plt.clf()

    df['date'] = np.log(df['date']) # log transformation
    sns.histplot(df['date'])
    plt.savefig('plots/distribution/account/date_log.jpg')
    plt.clf()


if __name__ == '__main__':
    Path("plots/distribution/account").mkdir(parents=True, exist_ok=True)
    #Path("plots/correlation/account").mkdir(parents=True, exist_ok=True)
    print("### ACCOUNT ###")
    account_du()