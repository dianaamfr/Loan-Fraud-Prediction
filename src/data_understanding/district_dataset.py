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

def district_du():
    df = db.df_query('SELECT * FROM district')
    stats(df)
    district_distribution(df)

def district_distribution(df):
    # Region
    print()
    sns.countplot(x ='region', data = df)
    plt.savefig('data_understanding/plots/distribution/district/region.jpg')
    plt.clf()

    # Inhabitants No
    sns.histplot(df['nr_inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_no.jpg')
    plt.clf()

    plt.boxplot(df['nr_inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_no_boxplot.jpg')
    plt.clf()

    # Municipalities
    sns.histplot(df['nr_municip_inhabitants_499'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_499.jpg')
    plt.clf()

    sns.histplot(df['nr_municip_inhabitants_500_1999'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_500_1999.jpg')
    plt.clf()

    sns.histplot(df['nr_municip_inhabitants_2000_9999'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_2000_9999.jpg')
    plt.clf()

    sns.histplot(df['nr_municip_inhabitants_10000'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_10000.jpg')
    plt.clf()

    # Cities
    sns.histplot(df['nr_cities'])
    plt.savefig('data_understanding/plots/distribution/district/cities.jpg')
    plt.clf()

    # Inhabitants Ratio
    sns.histplot(df['ratio_urban_inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/urban_inhabitants_ratio.jpg')
    plt.clf()

    sns.histplot(df['average_salary'])
    plt.savefig('data_understanding/plots/distribution/district/salary.jpg')
    plt.clf()

    df["unemployment_rate_95"] = pd.to_numeric(df["unemployment_rate_95"], errors='coerce')
    print("Na values in unemploymant_rate_95: ", df["unemployment_rate_95"].isna().sum())
    sns.histplot(df["unemployment_rate_95"])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_95.jpg')
    plt.clf()

    sns.histplot(df["unemployment_rate_96"])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_96.jpg')
    plt.clf()

    sns.histplot(df['nr_enterpreneurs_1000_inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/enterpreneurs.jpg')
    plt.clf()

    df["nr_commited_crimes_95"] = pd.to_numeric(df["nr_commited_crimes_95"], errors='coerce')
    print("Na values in nr_commited_crimes_95:", df["nr_commited_crimes_95"].isna().sum())
    sns.histplot(df["nr_commited_crimes_95"])
    plt.savefig('data_understanding/plots/distribution/district/crimes_95.jpg')
    plt.clf()

    sns.histplot(df["nr_commited_crimes_96"])
    plt.savefig('data_understanding/plots/distribution/district/crimes_96.jpg')
    plt.clf()


if __name__ == '__main__':
    Path("data_understanding/plots/distribution/district").mkdir(parents=True, exist_ok=True)
    district_du()