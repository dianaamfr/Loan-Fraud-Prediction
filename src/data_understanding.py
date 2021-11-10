from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##### Loading Datasets ##### 
files_folder = Path("../ficheiros_competicao/")

dataframes = dict.fromkeys(['account', 'card_test', 'card_train', 'client', 'disp', 'district', 'loan_test', 'loan_train', 'trans_test', 'trans_train'])

for key in dataframes:
    df = pd.read_csv(files_folder/(key+'.csv'), delimiter=";", low_memory=False)
    dataframes[key] = df

##### Preprocessing Datasets #####

# Account Dataset
# print(dataframes['account'].head())
# print()
# print(dataframes['account'].describe())
# print()
# print(dataframes['account'].info())
# print()
# print(dataframes['account'].isnull().sum())
# print()
# print(dataframes['account'].nunique())

# Card Test Dataset
# print(dataframes['card_test'].head())
# print()
# print(dataframes['card_test'].describe())
# print()
# print(dataframes['card_test'].info())
# print()
# print(dataframes['card_test'].isnull().sum())
# print()
# print(dataframes['card_test'].nunique())

# Card Train Dataset
# print(dataframes['card_train'].head())
# print()
# print(dataframes['card_train'].describe())
# print()
# print(dataframes['card_train'].info())
# print()
# print(dataframes['card_train'].isnull().sum())
# print()
# print(dataframes['card_train'].nunique())

# Client Dataset
# print(dataframes['client'].head())
# print()
# print(dataframes['client'].describe())
# print()
# print(dataframes['client'].info())
# print()
# print(dataframes['client'].isnull().sum())
# print()
# print(dataframes['client'].nunique())

# Disposition Dataset
# print(dataframes['disp'].head())
# print()
# print(dataframes['disp'].describe())
# print()
# print(dataframes['disp'].info())
# print()
# print(dataframes['disp'].isnull().sum())
# print()
# print(dataframes['disp'].nunique())

# District Dataset
# print(dataframes['district'].head())
# print()
# print(dataframes['district'].describe())
# print()
# print(dataframes['district'].info())
# print()
# print(dataframes['district'].isnull().sum())
# print()
# print(dataframes['district'].nunique())

# Loan Test Dataset
# print(dataframes['loan_test'].head())
# print()
# print(dataframes['loan_test'].describe())
# print()
# print(dataframes['loan_test'].info())
# print()
# print(dataframes['loan_test'].isnull().sum())
# print()
# print(dataframes['loan_test'].nunique())

# Loan Train Dataset
# print(dataframes['loan_train'].head())
# print()
# print(dataframes['loan_train'].describe())
# print()
# print(dataframes['loan_train'].info())
# print()
# print(dataframes['loan_train'].isnull().sum())
# print()
# print(dataframes['loan_train'].nunique())

# Transaction Test Dataset
# print(dataframes['trans_test'].head())
# print()
# print(dataframes['trans_test'].describe())
# print()
# print(dataframes['trans_test'].info())
# print()
# print(dataframes['trans_test'].isnull().sum())
# print()
# print(dataframes['trans_test'].nunique())

# Transaction Train Dataset
# print(dataframes['trans_train'].head())
# print()
# print(dataframes['trans_train'].describe())
# print()
# print(dataframes['trans_train'].info())
# print()
# print(dataframes['trans_train'].isnull().sum())
# print()
# print(dataframes['trans_train'].nunique())

##### Exploratory Data Analysis #####

# Account Dataset
# print()
# sns.countplot(x ='frequency', data = dataframes['account'])
# plt.show()
# sns.distplot(dataframes['account']['date'])
# plt.show()
# dataframes['account']['date'] = np.log(dataframes['account']['date']) # log transformation
# sns.distplot(dataframes['account']['date'])
# plt.show()

# Card Test Dataset
# print()
# sns.countplot(x ='type', data = dataframes['card_test'])
# plt.show()
# sns.distplot(dataframes['card_test']['issued'])
# plt.show()
# dataframes['card_test']['issued'] = np.log(dataframes['card_test']['issued']) # log transformation
# sns.distplot(dataframes['card_test']['issued'])
# plt.show()

# Card Train Dataset
# print()
# sns.countplot(x ='type', data = dataframes['card_train'])
# plt.show()
# sns.distplot(dataframes['card_train']['issued'])
# plt.show()
# dataframes['card_train']['issued'] = np.log(dataframes['card_train']['issued']) # log transformation
# sns.distplot(dataframes['card_train']['issued'])
# plt.show()

# Client Dataset
# sns.distplot(dataframes['client']['birth_number'])
# plt.show()
# dataframes['client']['birth_number'] = np.log(dataframes['client']['birth_number']) # log transformation
# sns.distplot(dataframes['client']['birth_number'])
# plt.show()

# Disposition Dataset
# print()
# sns.countplot(x ='type', data = dataframes['disp'])
# plt.show()

# District Dataset
# print()
# sns.countplot(x ='region', data = dataframes['district'])
# plt.show()

# sns.distplot(dataframes['district']['no. of inhabitants'])
# plt.show()
# print("Max no. of inhabitants", np.max(dataframes['district']['no. of inhabitants']))
# print("Second max no. of inhabitants:", sorted_inhabitants[-2])
# print("Min no. of inhabitants:", np.min(dataframes['district']['no. of inhabitants']))
# plt.boxplot(dataframes['district']['no. of inhabitants'])
# plt.show()
# dataframes['district']['no. of inhabitants'] = np.log(dataframes['district']['no. of inhabitants']) # log transformation
# sns.distplot(dataframes['district']['no. of inhabitants'])
# plt.show()
# sorted_inhabitants = np.sort(dataframes['district']['no. of inhabitants'])

# sns.distplot(dataframes['district']['no. of municipalities with inhabitants < 499 '])
# plt.show()
# dataframes['district']['no. of municipalities with inhabitants < 499 '] = np.log(dataframes['district']['no. of municipalities with inhabitants < 499 ']) # log transformation
# sns.distplot(dataframes['district']['no. of municipalities with inhabitants < 499 '])
# plt.show()

# sns.distplot(dataframes['district']['no. of municipalities with inhabitants 500-1999'])
# plt.show()
# dataframes['district']['no. of municipalities with inhabitants 500-1999'] = np.log(dataframes['district']['no. of municipalities with inhabitants 500-1999']) # log transformation
# sns.distplot(dataframes['district']['no. of municipalities with inhabitants 500-1999'])
# plt.show()

# sns.distplot(dataframes['district']['no. of municipalities with inhabitants 2000-9999 '])
# plt.show()
# dataframes['district']['no. of municipalities with inhabitants 2000-9999 '] = np.log(dataframes['district']['no. of municipalities with inhabitants 2000-9999 ']) # log transformation
# sns.distplot(dataframes['district']['no. of municipalities with inhabitants 2000-9999 '])
# plt.show()

# sns.distplot(dataframes['district']['no. of municipalities with inhabitants >10000 '])
# plt.show()
# dataframes['district']['no. of municipalities with inhabitants >10000 '] = np.log(dataframes['district']['no. of municipalities with inhabitants >10000 ']) # log transformation
# sns.distplot(dataframes['district']['no. of municipalities with inhabitants >10000 '])
# plt.show()

# sns.distplot(dataframes['district']['no. of cities '])
# plt.show()
# dataframes['district']['no. of cities '] = np.log(dataframes['district']['no. of cities ']) # log transformation
# sns.distplot(dataframes['district']['no. of cities '])
# plt.show()

# sns.distplot(dataframes['district']['ratio of urban inhabitants '])
# plt.show()
# dataframes['district']['ratio of urban inhabitants '] = np.log(dataframes['district']['ratio of urban inhabitants ']) # log transformation
# sns.distplot(dataframes['district']['ratio of urban inhabitants '])
# plt.show()

# sns.distplot(dataframes['district']['average salary '])
# plt.show()
# dataframes['district']['average salary '] = np.log(dataframes['district']['average salary ']) # log transformation
# sns.distplot(dataframes['district']['average salary '])
# plt.show()


#dataframes['district']["unemploymant rate '95 "] = pd.to_numeric(dataframes['district']["unemploymant rate '95 "], errors='coerce')
#print("Na values in unemploymant rate '95", dataframes['district']["unemploymant rate '95 "].isna().sum())
#dataframes['district']["unemploymant rate '95 "].dropna()
# sns.distplot(dataframes['district']["unemploymant rate '95 "])
# plt.show()
# dataframes['district']["unemploymant rate '95 "] = np.log(dataframes['district']["unemploymant rate '95 "]) # log transformation
# sns.distplot(dataframes['district']["unemploymant rate '95 "])
# plt.show()

# sns.distplot(dataframes['district']["unemploymant rate '96 "])
# plt.show()
# dataframes['district']["unemploymant rate '96 "] = np.log(dataframes['district']["unemploymant rate '96 "]) # log transformation
# sns.distplot(dataframes['district']["unemploymant rate '96 "])
# plt.show()

# sns.distplot(dataframes['district']['no. of enterpreneurs per 1000 inhabitants '])
# plt.show()
# dataframes['district']['no. of enterpreneurs per 1000 inhabitants '] = np.log(dataframes['district']['no. of enterpreneurs per 1000 inhabitants ']) # log transformation
# sns.distplot(dataframes['district']['no. of enterpreneurs per 1000 inhabitants '])
# plt.show()

# dataframes['district']["no. of commited crimes '95 "] = pd.to_numeric(dataframes['district']["no. of commited crimes '95 "], errors='coerce')
# print("Na values in no. of commited crimes '95 ", dataframes['district']["no. of commited crimes '95 "].isna().sum())
# dataframes['district']["no. of commited crimes '95 "].dropna()
#sns.distplot(dataframes['district']["no. of commited crimes '95 "])
# plt.show()
# dataframes['district']["no. of commited crimes '95 "] = np.log(dataframes['district']["no. of commited crimes '95 "]) # log transformation
# sns.distplot(dataframes['district']["no. of commited crimes '95 "])
# plt.show()

# sns.distplot(dataframes['district']["no. of commited crimes '96 "])
# plt.show()
# dataframes['district']["no. of commited crimes '96 "] = np.log(dataframes['district']["no. of commited crimes '96 "]) # log transformation
# sns.distplot(dataframes['district']["no. of commited crimes '96 "])
# plt.show()

# Loan Test Dataset
# sns.distplot(dataframes['loan_test']['date'])
# plt.show()
# dataframes['loan_test']['date'] = np.log(dataframes['loan_test']['date']) # log transformation
# sns.distplot(dataframes['loan_test']['date'])
# plt.show()

# sns.distplot(dataframes['loan_test']['amount'])
# plt.show()
# dataframes['loan_test']['amount'] = np.log(dataframes['loan_test']['amount']) # log transformation
# sns.distplot(dataframes['loan_test']['amount'])
# plt.show()

# sns.distplot(dataframes['loan_test']['duration'])
# plt.show()
# dataframes['loan_test']['duration'] = np.log(dataframes['loan_test']['duration']) # log transformation
# sns.distplot(dataframes['loan_test']['duration'])
# plt.show()

# sns.distplot(dataframes['loan_test']['payments'])
# plt.show()
# dataframes['loan_test']['payments'] = np.log(dataframes['loan_test']['payments']) # log transformation
# sns.distplot(dataframes['loan_test']['payments'])
# plt.show()

# Loan Train Dataset
# sns.distplot(dataframes['loan_train']['date'])
# plt.show()
# dataframes['loan_train']['date'] = np.log(dataframes['loan_train']['date']) # log transformation
# sns.distplot(dataframes['loan_train']['date'])
# plt.show()

# sns.distplot(dataframes['loan_train']['amount'])
# plt.show()
# dataframes['loan_train']['amount'] = np.log(dataframes['loan_train']['amount']) # log transformation
# sns.distplot(dataframes['loan_train']['amount'])
# plt.show()

# sns.distplot(dataframes['loan_train']['duration'])
# plt.show()
# dataframes['loan_train']['duration'] = np.log(dataframes['loan_train']['duration']) # log transformation
# sns.distplot(dataframes['loan_train']['duration'])
# plt.show()

# sns.distplot(dataframes['loan_train']['payments'])
# plt.show()
# dataframes['loan_train']['payments'] = np.log(dataframes['loan_train']['payments']) # log transformation
# sns.distplot(dataframes['loan_train']['payments'])
# plt.show()

# print()
# sns.countplot(x ='status', data = dataframes['loan_train'])
# plt.show()

# Transaction Test Dataset
# sns.distplot(dataframes['trans_test']['date'])
# plt.show()
# dataframes['trans_test']['date'] = np.log(dataframes['trans_test']['date']) # log transformation
# sns.distplot(dataframes['trans_test']['date'])
# plt.show()

# print()
# sns.countplot(x ='type', data = dataframes['trans_test'])
# plt.show()

# print()
# sns.countplot(x ='operation', data = dataframes['trans_test'])
# plt.show()

# sns.distplot(dataframes['trans_test']['amount'])
# plt.show()
# dataframes['trans_test']['amount'] = np.log(dataframes['trans_test']['amount']) # log transformation
# sns.distplot(dataframes['trans_test']['amount'])
# plt.show()

# sns.distplot(dataframes['trans_test']['balance'])
# plt.show()
# dataframes['trans_test']['balance'] = np.log(dataframes['trans_test']['balance']) # log transformation
# sns.distplot(dataframes['trans_test']['balance'])
# plt.show()

# print()
# sns.countplot(x ='k_symbol', data = dataframes['trans_test'])
# plt.show()

# print()
# sns.countplot(x ='bank', data = dataframes['trans_test'])
# plt.show()

# sns.distplot(dataframes['trans_test']['account'])
# plt.show()
# dataframes['trans_test']['account'] = np.log(dataframes['trans_test']['account']) # log transformation
# sns.distplot(dataframes['trans_test']['account'])
# plt.show()

# Transaction Train Dataset
# sns.distplot(dataframes['trans_train']['date'])
# plt.show()
# dataframes['trans_train']['date'] = np.log(dataframes['trans_train']['date']) # log transformation
# sns.distplot(dataframes['trans_train']['date'])
# plt.show()

# print()
# sns.countplot(x ='type', data = dataframes['trans_train'])
# plt.show()

# print()
# sns.countplot(x ='operation', data = dataframes['trans_train'])
# plt.show()

# sns.distplot(dataframes['trans_train']['amount'])
# plt.show()
# dataframes['trans_train']['amount'] = np.log(dataframes['trans_train']['amount']) # log transformation
# sns.distplot(dataframes['trans_train']['amount'])
# plt.show()

# sns.distplot(dataframes['trans_train']['balance'])
# plt.show()
# dataframes['trans_train']['balance'] = np.log(dataframes['trans_train']['balance']) # log transformation
# sns.distplot(dataframes['trans_train']['balance'])
# plt.show()

# print()
# sns.countplot(x ='k_symbol', data = dataframes['trans_train'])
# plt.show()

# print()
# sns.countplot(x ='bank', data = dataframes['trans_train'])
# plt.show()

# sns.distplot(dataframes['trans_train']['account'])
# plt.show()
# dataframes['trans_train']['account'] = np.log(dataframes['trans_train']['account']) # log transformation
# sns.distplot(dataframes['trans_train']['account'])
# plt.show()

#####################################
########### CORRELATION #############
#####################################

# fig, ax = plt.subplots(figsize=(20, 15))
# sns.heatmap(
#         dataframes['loan_train'].corr(), 
#         cmap = sns.diverging_palette(220, 10, as_cmap = True),
#         square=True, 
#         cbar=False,
#         ax=ax,
#         annot=True, 
#         linewidths=0.1,vmax=1.0, linecolor='white',
#         annot_kws={'fontsize':12 })
# plt.show()


############ 282 good loans, 46 bad loans ###########

##########################
####### LOAN AMOUNT ######
##########################

df_good = dataframes['loan_train'].loc[(dataframes['loan_train']['status'] == 1) ]
df_bad = dataframes['loan_train'].loc[(dataframes['loan_train']['status'] == -1) ]

print("Number of status 1: ",df_good.shape[0])
print("Number of status -1: ",df_bad.shape[0])

# Amount
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# df_good.amount.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6)
# df_bad.amount.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6)
# ax1.set_title('Loan Amount')
# ax2.set_title('Loan Amount')
# ax1.legend()
# ax2.legend()
# plt.show()

###################################################
# DAYS BETWEEN ACCOUNT CREATION AND LOAN ISSUANCE #
###################################################

#Passar para datetime
# dataframes['loan_train']['date'] = pd.to_datetime(dataframes['loan_train']['date'], format='%Y-%m-%d')
# dataframes['account']['date'] = pd.to_datetime(dataframes['loan_train']['date'], format='%Y-%m-%d')

# dataframes['loan_train']['days_between_statistics'] = dataframes['loan_train']['date'] - dataframes['account']['date']

# df_good = dataframes['loan_train'].loc[(dataframes['loan_train']['status'] == 1) ]
# df_bad = dataframes['loan_train'].loc[(dataframes['loan_train']['status'] == -1) ]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# df_good.days_between_statistics.dt.days.hist(bins=20, ax=ax1, label='good', color='green', alpha=0.6)
# df_bad.days_between_statistics.dt.days.hist(bins=20, ax=ax2, label='bad', color='red', alpha=0.6)
# ax1.set_title('Days Between Account Creation and Loan Issuance')
# ax2.set_title('Days Between Account Creation and Loan Issuance')
# ax1.legend()
# ax2.legend()
# plt.show()


######################################################
## PIE CHART PERCENTAGE OF GOOD LOANS FOR DISTRICT ##
######################################################


labels = dataframes['district']['code '].values #districts
percentages = [] 
districts_sem_loans=[]
for dist in labels:
    cnt_good=0
    cnt_total=0
    df_accounts = dataframes['account'].loc[(dataframes['account']['district_id'] == dist) ]
    accounts=df_accounts['account_id'].values
    for account in accounts:
        df_loans = dataframes['loan_train'].loc[(dataframes['loan_train']['account_id'] == account)]
        loans=df_loans['loan_id']
        for loan in loans:
            df_good = dataframes['loan_train'].loc[(dataframes['loan_train']['loan_id'] == loan)]
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
    district_names=dataframes['district'].loc[(dataframes['district']['code '] == d) ]
    names.append(district_names.iloc[0]['name '])

fig1, ax1 = plt.subplots()
ax1.pie(percentages,  labels=names, autopct='%1.1f%%',)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



######################################################
## SCATTER ##
######################################################
