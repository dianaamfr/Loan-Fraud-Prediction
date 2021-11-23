import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

# TODO - change to use DB
def disp_du():
    df = pd.read_csv(get_files_folder()/'disp.csv', delimiter=";", low_memory=False)
    stats(df)
    disp_distribution(df)

def disp_distribution(df):
    # Disposition Dataset
    sns.countplot(x ='type', data = df)
    plt.savefig(get_distribution_folder('disp')/'type.jpg')
    plt.clf()

if __name__ == '__main__':
    create_plots_folders('disp')
    disp_du()