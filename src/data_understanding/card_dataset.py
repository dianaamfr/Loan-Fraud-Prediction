import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

def card_train_du():
    card_train_df = pd.read_csv(get_files_folder()/'card_train.csv', delimiter=";", low_memory=False)
    stats(card_train_df)

    # Card Train Dataset
    print()
    sns.countplot(x ='type', data = card_train_df)
    plt.savefig('data_understanding/plots/distribution/card/card_train_type.jpg')
    plt.clf()
    
    sns.histplot(card_train_df['issued'])
    plt.savefig('data_understanding/plots/distribution/card/card_train_issued.jpg')
    plt.clf()

    card_train_df['issued'] = np.log(card_train_df['issued']) # log transformation
    sns.histplot(card_train_df['issued'])
    plt.savefig('data_understanding/plots/distribution/card/card_train_issued_log.jpg')
    plt.clf()


def card_test_du():
    card_test_df = pd.read_csv(get_files_folder()/'card_test.csv', delimiter=";", low_memory=False)
    stats(card_test_df)

    # Card Test Dataset
    print()
    sns.countplot(x ='type', data = card_test_df)
    plt.savefig('data_understanding/plots/distribution/card/card_test_type.jpg')
    plt.clf()

    sns.histplot(card_test_df['issued'])
    plt.savefig('data_understanding/plots/distribution/card/card_test_issued.jpg')
    plt.clf()

    card_test_df['issued'] = np.log(card_test_df['issued']) # log transformation
    sns.histplot(card_test_df['issued'])
    plt.savefig('data_understanding/plots/distribution/card/card_test_issued_log.jpg')
    plt.clf()

if __name__ == '__main__':
    Path("data_understanding/plots/distribution/card").mkdir(parents=True, exist_ok=True)
    card_train_du()
    card_test_du()