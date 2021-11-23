import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def client_du():
    df =  db.df_query('SELECT * FROM client')
    stats(df)
    client_distribution(df)

def client_distribution(df):

    # Client Dataset
    sns.histplot(df['birth_number'])
    plt.savefig(get_distribution_folder('client')/'birth_number.jpg')
    plt.clf()

    df['birth_number'] = np.log(df['birth_number']) # log transformation
    sns.histplot(df['birth_number'])
    plt.savefig(get_distribution_folder('client')/'birth_number_log.jpg')
    plt.clf()
   
if __name__ == '__main__':
    create_plots_folders('client')
    client_du()