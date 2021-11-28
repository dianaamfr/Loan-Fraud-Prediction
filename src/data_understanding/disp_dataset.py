import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import sys

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def disp_du():
    df = db.df_query('SELECT * FROM disposition')
    stats(df)
    disp_distribution(df)

def disp_distribution(df):
    # Disposition Dataset
    print()
    sns.countplot(x ='disp_type', data = df)
    plt.title('Disposition Type')
    plt.xlabel('type')
    plt.savefig(get_distribution_folder('disp')/'type.jpg')
    plt.clf()

if __name__ == '__main__':
    create_plots_folders('disp')
    disp_du()