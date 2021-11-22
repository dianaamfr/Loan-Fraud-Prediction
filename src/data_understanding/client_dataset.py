
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

def client_du():
    df = pd.read_csv(get_files_folder()/'client.csv', delimiter=";", low_memory=False)
    stats(df)
    client_distribution(df)

def client_distribution(df):

    # Client Dataset
    sns.histplot(df['birth_number'])
    plt.savefig('plots/distribution/client/birth_number.jpg')
    plt.clf()

    df['birth_number'] = np.log(df['birth_number']) # log transformation
    sns.histplot(df['birth_number'])
    plt.savefig('plots/distribution/client/birth_number_log.jpg')
    plt.clf()
   
if __name__ == '__main__':
    Path("plots/distribution/client").mkdir(parents=True, exist_ok=True)
    client_du()