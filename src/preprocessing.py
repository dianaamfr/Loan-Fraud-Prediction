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
print(dataframes['trans_train'].head())
print()
print(dataframes['trans_train'].describe())
print()
print(dataframes['trans_train'].info())
print()
print(dataframes['trans_train'].isnull().sum())
print()
print(dataframes['trans_train'].nunique())

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
# dataframes['district']['no. of inhabitants'] = np.log(dataframes['district']['no. of inhabitants']) # log transformation
# sns.distplot(dataframes['district']['no. of inhabitants'])
# plt.show()

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

# TODO
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

# TODO
# sns.distplot(dataframes['district']["no. of commited crimes '95 "])
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
sns.distplot(dataframes['trans_train']['date'])
plt.show()
dataframes['trans_train']['date'] = np.log(dataframes['trans_train']['date']) # log transformation
sns.distplot(dataframes['trans_train']['date'])
plt.show()

print()
sns.countplot(x ='type', data = dataframes['trans_train'])
plt.show()

print()
sns.countplot(x ='operation', data = dataframes['trans_train'])
plt.show()

sns.distplot(dataframes['trans_train']['amount'])
plt.show()
# dataframes['trans_train']['amount'] = np.log(dataframes['trans_train']['amount']) # log transformation
# sns.distplot(dataframes['trans_train']['amount'])
# plt.show()

sns.distplot(dataframes['trans_train']['balance'])
plt.show()
dataframes['trans_train']['balance'] = np.log(dataframes['trans_train']['balance']) # log transformation
sns.distplot(dataframes['trans_train']['balance'])
plt.show()

print()
sns.countplot(x ='k_symbol', data = dataframes['trans_train'])
plt.show()

print()
sns.countplot(x ='bank', data = dataframes['trans_train'])
plt.show()

sns.distplot(dataframes['trans_train']['account'])
plt.show()
# dataframes['trans_train']['account'] = np.log(dataframes['trans_train']['account']) # log transformation
# sns.distplot(dataframes['trans_train']['account'])
# plt.show()