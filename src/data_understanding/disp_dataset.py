import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

def disp_du():
    df = pd.read_csv(get_files_folder()/'disp.csv', delimiter=";", low_memory=False)
    stats(df)
    disp_distribution(df)

def disp_distribution(df):
    # Disposition Dataset
    print()
    sns.countplot(x ='type', data = df)
    plt.savefig('plots/distribution/disp/type.jpg')
    plt.clf()

if __name__ == '__main__':
    Path("plots/distribution/disp").mkdir(parents=True, exist_ok=True)
    disp_du()