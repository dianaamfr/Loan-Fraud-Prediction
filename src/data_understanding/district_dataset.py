import sys
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def district_du():
    df = db.df_query('SELECT * FROM district')
    # Drop NaN values
    df["nr_commited_crimes_95"] = pd.to_numeric(df["nr_commited_crimes_95"], errors='coerce')
    df["unemployment_rate_95"] = pd.to_numeric(df["unemployment_rate_95"], errors='coerce')
    df.dropna(inplace=True) # Remove NaN values

    stats(df)
    district_distribution(df)
    district_correlation(df.copy())
    variable_relations(df.copy())
    loan_relations()

def district_distribution(df):
    # Region
    sns.countplot(x ='region', data = df)
    plt.savefig(get_distribution_folder('district')/'region.jpg')
    plt.clf()

    # Inhabitants No
    sns.histplot(df['nr_inhabitants'])
    plt.savefig(get_distribution_folder('district')/'inhabitants_no.jpg')
    plt.clf()

    plt.boxplot(df['nr_inhabitants'])
    plt.savefig(get_distribution_folder('district')/'inhabitants_no_boxplot.jpg')
    plt.clf()

    # Municipalities
    sns.histplot(df['nr_municip_inhabitants_499'])
    plt.savefig(get_distribution_folder('district')/'inhabitants_499.jpg')
    plt.clf()

    sns.histplot(df['nr_municip_inhabitants_500_1999'])
    plt.savefig(get_distribution_folder('district')/'inhabitants_500_1999.jpg')
    plt.clf()

    sns.histplot(df['nr_municip_inhabitants_2000_9999'])
    plt.savefig(get_distribution_folder('district')/'inhabitants_2000_9999.jpg')
    plt.clf()

    sns.histplot(df['nr_municip_inhabitants_10000'])
    plt.savefig(get_distribution_folder('district')/'inhabitants_10000.jpg')
    plt.clf()

    # Cities
    sns.histplot(df['nr_cities'])
    plt.savefig(get_distribution_folder('district')/'cities.jpg')
    plt.clf()

    # Inhabitants Ratio
    sns.histplot(df['ratio_urban_inhabitants'])
    plt.savefig(get_distribution_folder('district')/'urban_inhabitants_ratio.jpg')
    plt.clf()

    sns.histplot(df['average_salary'])
    plt.savefig(get_distribution_folder('district')/'salary.jpg')
    plt.clf()

    # Unemployment Rate
    sns.histplot(df["unemployment_rate_95"])
    plt.savefig(get_distribution_folder('district')/'unemploymant_rate_95.jpg')
    plt.clf()

    qqplot(df['unemployment_rate_95'],norm,fit=True,line="45")
    plt.savefig(get_distribution_folder('district')/'unemploymant_rate_95_qqplot.jpg')
    plt.clf()

    sns.histplot(df["unemployment_rate_96"])
    plt.savefig(get_distribution_folder('district')/'unemploymant_rate_96.jpg')
    plt.clf()

    sns.histplot(df['nr_enterpreneurs_1000_inhabitants'])
    plt.savefig(get_distribution_folder('district')/'enterpreneurs.jpg')
    plt.clf()

    # Commited Crimes
    sns.histplot(df["nr_commited_crimes_95"])
    plt.savefig(get_distribution_folder('district')/'crimes_95.jpg')
    plt.clf()

    plt.boxplot(df['nr_commited_crimes_95'])
    plt.savefig(get_distribution_folder('district')/'crimes_95_boxplot.jpg')
    plt.clf()

    sns.histplot(df["nr_commited_crimes_96"])
    plt.savefig(get_distribution_folder('district')/'crimes_96.jpg')
    plt.clf()

    plt.boxplot(df['nr_commited_crimes_96'])
    plt.savefig(get_distribution_folder('district')/'crimes_96_boxplot.jpg')
    plt.clf()

def district_correlation(df):
    corrMatrix = df.corr()
    plt.figure(figsize=(20,15))
    ax=plt.subplot(111)
    sns.heatmap(corrMatrix,ax=ax, annot=True)
    plt.savefig(get_correlation_folder('district')/'district.jpg')
    plt.clf()

def variable_relations(district_df):

    # Compare nr commited crimes 
    sns.relplot(data=district_df, y="nr_commited_crimes_95", x="nr_commited_crimes_96", sizes=(40, 400), alpha=.8,height=6)
    plt.savefig(get_correlation_folder('district')/'commited_crimes_95_96.jpg')
    plt.clf()

    # Compare nr inhabitants
    sns.relplot(data=district_df, y="unemployment_rate_95", x="unemployment_rate_96", sizes=(40, 400), alpha=.8,height=6)
    plt.savefig(get_correlation_folder('district')/'unemployment_95_96.jpg')
    plt.clf()

    # Inhabitants and salary
    sns.relplot(data=district_df, y="ratio_urban_inhabitants", x="nr_inhabitants", hue="average_salary", size="average_salary",sizes=(40, 400), alpha=.8,height=6)
    plt.savefig(get_correlation_folder('district')/'inhabitants_salary.jpg')
    plt.clf()

    # Inhabitants and average salary
    g = sns.relplot(data=district_df, y="average_salary", x="nr_inhabitants", sizes=(40, 400), alpha=.8, height=6)
    plt.savefig(get_correlation_folder('district')/'inhabitants_salary2.jpg')
    plt.clf()

    # Inhabitants and commited crimes 96
    sns.relplot(data=district_df, y="nr_commited_crimes_96", x="nr_inhabitants", sizes=(40, 400), alpha=.8,height=6)
    plt.savefig(get_correlation_folder('district')/'inhabitants_crimes_96.jpg')
    plt.clf()

    # Inhabitants and commited crimes 95
    sns.relplot(data=district_df, y="nr_commited_crimes_95", x="nr_inhabitants", sizes=(40, 400), alpha=.8,height=6)
    plt.savefig(get_correlation_folder('district')/'inhabitants_crimes_95.jpg')
    plt.clf()

    g = sns.relplot(data=district_df, y="nr_commited_crimes_95", x="nr_commited_crimes_96", hue="average_salary", size="average_salary", sizes=(10, 400), alpha=0.8)
    g.set(xscale="log")
    g.set(yscale="log")
    plt.savefig(get_correlation_folder('district')/'crimes_salary.jpg')
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
    plt.savefig(get_correlation_folder('district')/'district_loan.jpg')
    plt.clf()

if __name__ == '__main__':
    create_plots_folders('district')
    district_du()