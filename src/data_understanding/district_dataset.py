import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path

def district_du():
    df = pd.read_csv(get_files_folder()/'district.csv', delimiter=";", low_memory=False)
    stats(df)
    district_distribution(df)

def district_distribution(df):
    # Region
    print()
    sns.countplot(x ='region', data = df)
    plt.savefig('data_understanding/plots/distribution/district/region.jpg')
    plt.clf()

    # Inhabitants No
    sns.histplot(df['no. of inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_no.jpg')
    plt.clf()

    plt.boxplot(df['no. of inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_no_boxplot.jpg')
    plt.clf()

    df['no. of inhabitants'] = np.log(df['no. of inhabitants']) # log transformation
    sns.histplot(df['no. of inhabitants'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_no_log.jpg')
    plt.clf()

    # Municipalities
    sns.histplot(df['no. of municipalities with inhabitants < 499 '])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_499.jpg')
    plt.clf()

    df['no. of municipalities with inhabitants < 499 '] = np.log(df['no. of municipalities with inhabitants < 499 ']) # log transformation
    sns.histplot(df['no. of municipalities with inhabitants < 499 '])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_499_log.jpg')
    plt.clf()

    sns.histplot(df['no. of municipalities with inhabitants 500-1999'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_500_1999.jpg')
    plt.clf()

    df['no. of municipalities with inhabitants 500-1999'] = np.log(df['no. of municipalities with inhabitants 500-1999']) # log transformation
    sns.histplot(df['no. of municipalities with inhabitants 500-1999'])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_500_1999_log.jpg')
    plt.clf()

    sns.histplot(df['no. of municipalities with inhabitants 2000-9999 '])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_2000_9999.jpg')
    plt.clf()

    df['no. of municipalities with inhabitants 2000-9999 '] = np.log(df['no. of municipalities with inhabitants 2000-9999 ']) # log transformation
    sns.histplot(df['no. of municipalities with inhabitants 2000-9999 '])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_2000_1999_log.jpg')
    plt.clf()


    sns.histplot(df['no. of municipalities with inhabitants >10000 '])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_10000.jpg')
    plt.clf()
    
    df['no. of municipalities with inhabitants >10000 '] = np.log(df['no. of municipalities with inhabitants >10000 ']) # log transformation
    sns.histplot(df['no. of municipalities with inhabitants >10000 '])
    plt.savefig('data_understanding/plots/distribution/district/inhabitants_10000_log.jpg')
    plt.clf()


    # Cities
    sns.histplot(df['no. of cities '])
    plt.savefig('data_understanding/plots/distribution/district/cities.jpg')
    plt.clf()
    df['no. of cities '] = np.log(df['no. of cities ']) # log transformation
    sns.histplot(df['no. of cities '])
    plt.savefig('data_understanding/plots/distribution/district/cities_log.jpg')
    plt.clf()

    # Inhabitants Ratio
    sns.histplot(df['ratio of urban inhabitants '])
    plt.savefig('data_understanding/plots/distribution/district/urban_inhabitants_ratio.jpg')
    plt.clf()
    df['ratio of urban inhabitants '] = np.log(df['ratio of urban inhabitants ']) # log transformation
    sns.histplot(df['ratio of urban inhabitants '])
    plt.savefig('data_understanding/plots/distribution/district/urban_inhabitants_ratio_log.jpg')
    plt.clf()


    sns.histplot(df['average salary '])
    plt.savefig('data_understanding/plots/distribution/district/salary.jpg')
    plt.clf()

    df['average salary '] = np.log(df['average salary ']) # log transformation
    sns.histplot(df['average salary '])
    plt.savefig('data_understanding/plots/distribution/district/salary_log.jpg')
    plt.clf()

    df["unemploymant rate '95 "] = pd.to_numeric(df["unemploymant rate '95 "], errors='coerce')
    print("Na values in unemploymant rate '95", df["unemploymant rate '95 "].isna().sum())
    #df["unemploymant rate '95 "].dropna()
    sns.histplot(df["unemploymant rate '95 "])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_95.jpg')
    plt.clf()
    df["unemploymant rate '95 "] = np.log(df["unemploymant rate '95 "]) # log transformation
    sns.histplot(df["unemploymant rate '95 "])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_95_log.jpg')
    plt.clf()

    sns.histplot(df["unemploymant rate '96 "])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_96.jpg')
    plt.clf()

    df["unemploymant rate '96 "] = np.log(df["unemploymant rate '96 "]) # log transformation
    sns.histplot(df["unemploymant rate '96 "])
    plt.savefig('data_understanding/plots/distribution/district/unemploymant_rate_96_log.jpg')
    plt.clf()

    sns.histplot(df['no. of enterpreneurs per 1000 inhabitants '])
    plt.savefig('data_understanding/plots/distribution/district/enterpreneurs.jpg')
    plt.clf()

    df['no. of enterpreneurs per 1000 inhabitants '] = np.log(df['no. of enterpreneurs per 1000 inhabitants ']) # log transformation
    sns.histplot(df['no. of enterpreneurs per 1000 inhabitants '])
    plt.savefig('data_understanding/plots/distribution/district/enterpreneurs_log.jpg')
    plt.clf()

    df["no. of commited crimes '95 "] = pd.to_numeric(df["no. of commited crimes '95 "], errors='coerce')
    print("Na values in no. of commited crimes '95 ", df["no. of commited crimes '95 "].isna().sum())
    #df["no. of commited crimes '95 "].dropna()
    sns.histplot(df["no. of commited crimes '95 "])
    plt.savefig('data_understanding/plots/distribution/district/crimes_95.jpg')
    plt.clf()

    df["no. of commited crimes '95 "] = np.log(df["no. of commited crimes '95 "]) # log transformation
    sns.histplot(df["no. of commited crimes '95 "])
    plt.savefig('data_understanding/plots/distribution/district/crimes_95_log.jpg')
    plt.clf()


    sns.histplot(df["no. of commited crimes '96 "])
    plt.savefig('data_understanding/plots/distribution/district/crimes_96.jpg')
    plt.clf()

    df["no. of commited crimes '96 "] = np.log(df["no. of commited crimes '96 "]) # log transformation
    sns.histplot(df["no. of commited crimes '96 "])
    plt.savefig('data_understanding/plots/distribution/district/crimes_96_log.jpg')
    plt.clf()


if __name__ == '__main__':
    Path("data_understanding/plots/distribution/district").mkdir(parents=True, exist_ok=True)
    district_du()