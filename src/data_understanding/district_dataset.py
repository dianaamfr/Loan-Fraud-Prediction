import sys
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from matplotlib.ticker import PercentFormatter

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

    average_crimes()
    average_unemployment()
    same_district()

    loan_relations()

def district_distribution(df):
    # Region
    plt.figure(figsize=(16, 7))
    sns.countplot(x ='region', data = df)
    plt.tight_layout()
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


def average_crimes():
    df = db.df_query('SELECT nr_commited_crimes_95 AS crimes_95, nr_commited_crimes_96 AS crimes_96, loan_status '
                    'FROM account JOIN district USING(district_id) JOIN loan_train USING(account_id)')
    # Drop NaN values
    df["crimes_95"] = pd.to_numeric(df["crimes_95"], errors='coerce')
    df.dropna(inplace=True) # Remove NaN values

    df["avg_crimes"] = (df["crimes_95"] + df["crimes_96"]) / 2.0

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average crimes for the district of the account

    df_good.avg_crimes.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.avg_crimes)) / len(df_good.avg_crimes))
   
    df_bad.avg_crimes.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.avg_crimes)) / len(df_bad.avg_crimes))

    ax1.set_title('Average crimes on the districts of the accounts')
    ax2.set_title('Average crimes on the districts of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('district')/'avg_crimes_account_district_status.jpg')
    plt.clf()

def average_unemployment():
    df = db.df_query('SELECT unemployment_rate_95 AS unemployment_95, unemployment_rate_96 AS unemployment_96, loan_status '
                    'FROM account JOIN district USING(district_id) JOIN loan_train USING(account_id)')
    # Drop NaN values
    df["unemployment_95"] = pd.to_numeric(df["unemployment_95"], errors='coerce')
    df.dropna(inplace=True) # Remove NaN values

    df["avg_unemployment"] = (df["unemployment_95"] + df["unemployment_96"]) / 2.0

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average unemployment for the district of the account

    df_good.avg_unemployment.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.avg_unemployment)) / len(df_good.avg_unemployment))
   
    df_bad.avg_unemployment.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.avg_unemployment)) / len(df_bad.avg_unemployment))

    ax1.set_title('Average unemployment on the districts of the accounts')
    ax2.set_title('Average unemployment on the districts of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('district')/'avg_unemployment_account_district_status.jpg')
    plt.clf()

def same_district():
    df = db.df_query('SELECT loan_status, '\
        'CASE WHEN district_account.district_id = district_client.district_id THEN 1 ELSE 0 END AS same_district '\
        'FROM loan_train JOIN account USING(account_id) ' \
        'JOIN district AS district_account ON district_account.district_id =  account.district_id '\
        'JOIN disposition USING(account_id) JOIN client USING(client_id) '\
        'JOIN district AS district_client ON district_client.district_id = client.district_id '\
        'WHERE disposition.disp_type = "OWNER"')
    
    sns.countplot(x='same_district', data=df,  hue="loan_status", palette=['#ff9973ff', '#00cfccff'])
    plt.savefig(get_correlation_folder('district')/'same_district_status.jpg')
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