import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import sys

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def loan_district_du():
    # PIE CHART PERCENTAGE OF GOOD LOANS FOR DISTRICT
    df_loantrain = db.df_query('SELECT * FROM loan_train')
    df_account = db.df_query('SELECT * FROM account')
    df_district = db.df_query('SELECT * FROM district')

    labels = df_district['district_id'].values #districts
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
                if(df_good.iloc[0]['loan_status']==1): 
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
        district_names=df_district.loc[(df_district['district_id'] == d) ]
        names.append(district_names.iloc[0]['district_name'])

    fig1, ax1 = plt.subplots()
    ax1.pie(percentages,  labels=names, autopct='%1.1f%%',)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

if __name__ == '__main__':
    loan_district_du()