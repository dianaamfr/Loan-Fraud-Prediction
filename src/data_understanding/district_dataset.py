from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
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

    df["unemployment_rate_95"] = pd.to_numeric(df["unemployment_rate_95"], errors='coerce')
    df["nr_commited_crimes_95"] = pd.to_numeric(df["nr_commited_crimes_95"], errors='coerce')

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

    # Unemployment Rate
    df["unemployment_rate_95"] = pd.to_numeric(df["unemployment_rate_95"], errors='coerce')
    df.dropna(subset=['unemployment_rate_95'], inplace=True) # Remove NaN values
    sns.histplot(df["unemployment_rate_95"])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_95.jpg')
    plt.clf()

    qqplot(df['unemployment_rate_95'],norm,fit=True,line="45")
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_95_qqplot.jpg')
    plt.clf()

    sns.histplot(df["unemployment_rate_96"])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_96.jpg')
    plt.clf()

    sns.histplot(df['nr_enterpreneurs_1000_inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/enterpreneurs.jpg')
    plt.clf()

    # Commited Crimes
    df["nr_commited_crimes_95"] = pd.to_numeric(df["nr_commited_crimes_95"], errors='coerce')
    df.dropna(subset=['nr_commited_crimes_95'], inplace=True) # Remove NaN values
    sns.histplot(df["nr_commited_crimes_95"])
    plt.savefig('data_understanding/plots/distribution/district/crimes_95.jpg')
    plt.clf()

    plt.boxplot(df['nr_commited_crimes_95'])
    plt.savefig('data_understanding/plots/distribution/district/crimes_95_boxplot.jpg')
    plt.clf()

    sns.histplot(df["nr_commited_crimes_96"])
    plt.savefig('data_understanding/plots/distribution/district/crimes_96.jpg')
    plt.clf()

    plt.boxplot(df['nr_commited_crimes_96'])
    plt.savefig('data_understanding/plots/distribution/district/crimes_96_boxplot.jpg')
    plt.clf()

def district_correlation():
    df = db.df_query('SELECT * FROM district')
    df["nr_commited_crimes_95"] = pd.to_numeric(df["nr_commited_crimes_95"], errors='coerce')
    df["unemployment_rate_95"] = pd.to_numeric(df["unemployment_rate_95"], errors='coerce')

    corrMatrix = df.corr()
    plt.figure(figsize=(20,15))
    ax=plt.subplot(111)
    sns.heatmap(corrMatrix,ax=ax, annot=True)
    plt.savefig('data_understanding/plots/correlation/district/district.jpg')
    plt.clf()

def variable_relations():
    district_df = db.df_query('SELECT * FROM district')

    # Drop NaN values
    district_df["nr_commited_crimes_95"] = pd.to_numeric(district_df["nr_commited_crimes_95"], errors='coerce')
    district_df["unemployment_rate_95"] = pd.to_numeric(district_df["unemployment_rate_95"], errors='coerce')
    district_df.dropna(inplace=True) # Remove NaN values

    # Compare nr commited crimes 
    sns.relplot(data=district_df, y="nr_commited_crimes_95", x="nr_commited_crimes_96", sizes=(40, 400), alpha=.5,height=6)
    plt.savefig('data_understanding/plots/correlation/district/commited_crimes_95_96.jpg')
    plt.clf()

    # Compare nr inhabitants
    sns.relplot(data=district_df, y="unemployment_rate_95", x="unemployment_rate_96", sizes=(40, 400), alpha=.5,height=6)
    plt.savefig('data_understanding/plots/correlation/district/unemployment_95_96.jpg')
    plt.clf()

    # Inhabitants and salary
    sns.relplot(data=district_df, y="ratio_urban_inhabitants", x="nr_inhabitants", hue="average_salary", size="average_salary",sizes=(40, 400), alpha=.5,height=6)
    plt.savefig('data_understanding/plots/correlation/district/inhabitants_salary.jpg')
    plt.clf()

    # Inhabitants and commited crimes 96
    sns.relplot(data=district_df, y="nr_commited_crimes_96", x="nr_inhabitants", sizes=(40, 400), alpha=.5,height=6)
    plt.savefig('data_understanding/plots/correlation/district/inhabitants_crimes.jpg')
    plt.clf()

    g = sns.relplot(data=district_df, y="nr_commited_crimes_95", x="nr_commited_crimes_96", hue="average_salary", size="average_salary", sizes=(10, 400), alpha=0.5)
    g.set(xscale="log")
    g.set(yscale="log")
    plt.savefig('data_understanding/plots/correlation/district/crimes_salary.jpg')
    plt.clf()

def loan_relations():
    loan_df = db.df_query('SELECT * FROM loan_train')
    district_df = db.df_query('SELECT * FROM district')
    account_df = db.df_query('SELECT * FROM account')

    district_df["nr_commited_crimes_95"] = pd.to_numeric(district_df["nr_commited_crimes_95"], errors='coerce')
    district_df["unemployment_rate_95"] = pd.to_numeric(district_df["unemployment_rate_95"], errors='coerce')
    district_df.dropna(inplace=True) # Remove NaN values

    merged_df = pd.merge(loan_df, account_df, how="inner",on="account_id")
    merged_df = pd.merge(merged_df, district_df, how="inner",on="district_id")

    corrMatrix = merged_df.corr()
    plt.figure(figsize=(20,15))
    ax=plt.subplot(111)
    sns.heatmap(corrMatrix,ax=ax, annot=True)
    plt.savefig('data_understanding/plots/correlation/district/district_loan.jpg')
    plt.clf()

if __name__ == '__main__':
    Path("data_understanding/plots/distribution/district").mkdir(parents=True, exist_ok=True)
    Path("data_understanding/plots/correlation/district").mkdir(parents=True, exist_ok=True)
    district_du()
    district_correlation()
    variable_relations()
    loan_relations()