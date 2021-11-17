import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import sys

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def loan_account_du():
    df = db.df_query('SELECT * FROM loan_train JOIN account USING(account_id)')
    print(df.head())
    # DAYS BETWEEN ACCOUNT CREATION AND LOAN ISSUANCE 

    #Passar para datetime
    df['granted_date'] = pd.to_datetime(df['granted_date'], format='%Y%m%d', errors='coerce')
    df['creation_date'] = pd.to_datetime(df['creation_date'], format='%Y%m%d', errors='coerce')
    print(df.head())
    df['days_between_statistics'] = df['granted_date'] - df['creation_date']
    print(df.head())

    df_good = df.loc[(df['loan_status'] == 1) ]
    df_bad = df.loc[(df['loan_status'] == -1) ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    df_good.days_between_statistics.dt.days.hist(bins=20, ax=ax1, label='good', color='green', alpha=0.6)
    df_bad.days_between_statistics.dt.days.hist(bins=20, ax=ax2, label='bad', color='red', alpha=0.6)
    ax1.set_title('Days Between Account Creation and Loan Issuance')
    ax2.set_title('Days Between Account Creation and Loan Issuance')
    ax1.legend()
    ax2.legend()

    plt.savefig(get_correlation_folder('loan')/'loan_account_dates.jpg')
    plt.clf()
    



def loan_district_du():
    # PIE CHART PERCENTAGE OF GOOD LOANS FOR DISTRICT
    df_loantrain=pd.read_csv(get_files_folder()/'loan_train.csv', delimiter=";", low_memory=False)
    df_account=pd.read_csv(get_files_folder()/'loan_train.csv', delimiter=";", low_memory=False)
    df_district=pd.read_csv(get_files_folder()/'district.csv', delimiter=";", low_memory=False)

    labels = df_district['code '].values #districts
    percentages = [] 
    districts_sem_loans=[]
    for dist in labels:
        cnt_good=0
        cnt_total=0
        df_accounts = df_account.loc[(df_account['district_id'] == dist) ]
        accounts=df_accounts['account_id'].values
        for account in accounts:
            df_loans = df_loantrain.loc[(df_loantrain['account_id'] == account)]
            loans=df_loans['loan_id']
            for loan in loans:
                df_good = df_loantrain.loc[(df_loantrain['loan_id'] == loan)]
                cnt_total+=1
                if(df_good.iloc[0]['status']==1): 
                    cnt_good+=1
        if(cnt_total!=0):
            print('este distrito tem loans',dist)
            good_percentage=cnt_good/cnt_total
            percentages.append(good_percentage)
        else:
            print('este distrito não tem loans',dist)
            districts_sem_loans.append(dist)

    for d in districts_sem_loans:
        indexArr = np.argwhere(labels==d)
        labels=np.delete(labels,indexArr)

    names=[]
    for d in labels:
        district_names=df_district.loc[(df_district['code '] == d) ]
        names.append(district_names.iloc[0]['name '])

    fig1, ax1 = plt.subplots()
    ax1.pie(percentages,  labels=names, autopct='%1.1f%%',)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()




if __name__ == '__main__':
    loan_account_du()