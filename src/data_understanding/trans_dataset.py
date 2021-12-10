import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import sys
from datetime import datetime
from matplotlib.ticker import PercentFormatter

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def trans_test_du():
    trans_test_df = db.df_query('SELECT * FROM trans_test')
    stats(trans_test_df)

    # Transaction Test Dataset
    sns.histplot(trans_test_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_date.jpg')
    plt.clf()

    trans_test_df['trans_date'] = np.log(trans_test_df['trans_date']) # log transformation
    sns.histplot(trans_test_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_date_log.jpg')
    plt.clf()

    sns.countplot(x ='trans_type', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_type.jpg')
    plt.clf()

    sns.countplot(x ='operation', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_operation.jpg')
    plt.clf()

    sns.histplot(trans_test_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_amount.jpg')
    plt.clf()

    trans_test_df['amount'] = np.log(trans_test_df['amount']) # log transformation
    sns.histplot(trans_test_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_amount_log.jpg')
    plt.clf()

    sns.histplot(trans_test_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_balance.jpg')
    plt.clf()

    trans_test_df['balance'] = np.log(trans_test_df['balance']) # log transformation
    sns.histplot(trans_test_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_balance_log.jpg')
    plt.clf()

    sns.countplot(x ='k_symbol', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_k_symbol.jpg')
    plt.clf()

    sns.countplot(x ='bank', data = trans_test_df)
    plt.savefig(get_distribution_folder('trans')/'trans_test_bank.jpg')
    plt.clf()

    sns.histplot(trans_test_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_account.jpg')
    plt.clf()

    trans_test_df['account'] = np.log(trans_test_df['account']) # log transformation
    sns.histplot(trans_test_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_test_account_log.jpg')
    plt.clf()

def trans_train_du():
    trans_train_df = db.df_query('SELECT * FROM trans_train')
    stats(trans_train_df)

    # Transaction Train Dataset
    sns.histplot(trans_train_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_date.jpg')
    plt.clf()

    trans_train_df['trans_date'] = np.log(trans_train_df['trans_date']) # log transformation
    sns.histplot(trans_train_df['trans_date'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_date_log.jpg')
    plt.clf()

    sns.countplot(x ='trans_type', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_type.jpg')
    plt.clf()

    sns.countplot(x ='operation', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_type_log.jpg')
    plt.clf()

    sns.histplot(trans_train_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_amount.jpg')
    plt.clf()

    trans_train_df['amount'] = np.log(trans_train_df['amount']) # log transformation
    sns.histplot(trans_train_df['amount'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_amount_log.jpg')
    plt.clf()

    sns.histplot(trans_train_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_balance.jpg')
    plt.clf()

    trans_train_df['balance'] = np.log(trans_train_df['balance']) # log transformation
    sns.histplot(trans_train_df['balance'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_balance_log.jpg')
    plt.clf()

    sns.countplot(x ='k_symbol', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_k_symbol.jpg')
    plt.clf()
    
    sns.countplot(x ='bank', data = trans_train_df)
    plt.savefig(get_distribution_folder('trans')/'trans_train_bank.jpg')
    plt.clf()

    sns.histplot(trans_train_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_account.jpg')
    plt.clf()

    trans_train_df['account'] = np.log(trans_train_df['account']) # log transformation
    sns.histplot(trans_train_df['account'])
    plt.savefig(get_distribution_folder('trans')/'trans_train_account_log.jpg')
    plt.clf()

 
def num_trans_status():
    df_loans = db.df_query('SELECT * FROM loan_train')
    df_trans = db.df_query('SELECT account_id, COUNT(*) AS num_trans FROM trans_train WHERE account_id IN (SELECT DISTINCT account_id FROM loan_train) GROUP BY account_id ORDER BY account_id')

    df = pd.merge(df_loans, df_trans, how="inner",on="account_id")

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.num_trans.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.num_trans)) / len(df_good.num_trans))
   
    df_bad.num_trans.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.num_trans)) / len(df_bad.num_trans))

    ax1.set_ylim([0,0.16])
    ax2.set_ylim([0,0.16])

    ax1.set_title('Number of transactions of the accounts')
    ax2.set_title('Number of transactions of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1)) 
    ax2.yaxis.set_major_formatter(PercentFormatter(1)) 

    plt.savefig(get_correlation_folder('trans')/'num_trans_status.jpg')
    plt.clf()

def avg_amount_status():
    df_loans = db.df_query('SELECT * FROM loan_train')
    df_trans = db.df_query('SELECT account_id, AVG(amount) AS avg_amount FROM trans_train WHERE account_id IN (SELECT DISTINCT account_id FROM loan_train) GROUP BY account_id ORDER BY account_id')

    df = pd.merge(df_loans, df_trans, how="inner",on="account_id")

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.avg_amount.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.avg_amount)) / len(df_good.avg_amount))
   
    df_bad.avg_amount.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.avg_amount)) / len(df_bad.avg_amount))

    ax1.set_xlim([0,25000])
    ax2.set_xlim([0,25000])
    ax1.set_ylim([0,0.14])
    ax2.set_ylim([0,0.14])

    ax1.set_title('Average transaction amount of the accounts')
    ax2.set_title('Average transaction amount of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('trans')/'avg_amount_status.jpg')
    plt.clf()

def avg_balance_status():
    df_loans = db.df_query('SELECT * FROM loan_train')
    df_trans = db.df_query('SELECT account_id, AVG(balance) AS avg_balance FROM trans_train WHERE account_id IN (SELECT DISTINCT account_id FROM loan_train) GROUP BY account_id ORDER BY account_id')

    df = pd.merge(df_loans, df_trans, how="inner",on="account_id")

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_good.avg_balance.hist(bins=20, ax=ax1, label='status 1', color='green', alpha=0.6, 
     weights=np.ones(len(df_good.avg_balance)) / len(df_good.avg_balance))
   
    df_bad.avg_balance.hist(bins=20, ax=ax2, label='status -1', color='red', alpha=0.6,
     weights=np.ones(len(df_bad.avg_balance)) / len(df_bad.avg_balance))

    ax1.set_xlim([10000,80000])
    ax2.set_xlim([10000,80000])
    ax1.set_ylim([0,0.18])
    ax2.set_ylim([0,0.18])

    ax1.set_title('Average transaction balance of the accounts')
    ax2.set_title('Average transaction balance of the accounts')
    ax1.legend()
    ax2.legend()

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(get_correlation_folder('trans')/'avg_balance_status.jpg')
    plt.clf()


def transactions_correlation_matrix(): 
    df = db.df_query('SELECT * FROM trans_train')

    df["trans_id"] = pd.to_numeric(df["trans_id"], errors='coerce')
    df["account_id"] = pd.to_numeric(df["account_id"], errors='coerce')
    df["trans_date"] = pd.to_numeric(df["trans_date"], errors='coerce')
    df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
    df["balance"] = pd.to_numeric(df["balance"], errors='coerce')
    df["account"] = pd.to_numeric(df["account"], errors='coerce')

    df["trans_type"] = df['trans_type'].astype('category').cat.codes
    df["operation"] = df['operation'].astype('category').cat.codes
    df["k_symbol"] = df['k_symbol'].astype('category').cat.codes
    df["bank"] = df['bank'].astype('category').cat.codes

    corrMatrix = df.corr()

    plt.figure(figsize=(20,15))
    ax=plt.subplot(111)
    sns.heatmap(corrMatrix,ax=ax,annot=True)
    plt.savefig(get_correlation_folder('trans')/'correlation_matrix.jpg')
    plt.clf()

def transactions_loan_date():
    df_trans = db.df_query('SELECT * FROM trans_train')
    df_loan = db.df_query('SELECT * FROM loan_train')
    df_account = db.df_query('SELECT * FROM account')
    
    merged_df = pd.merge(df_loan, df_account, how="inner",on="account_id")
    merged_df = pd.merge(merged_df, df_trans, how="inner",on="account_id")

    for index, row in merged_df.iterrows():
        merged_df["granted_date"][index] = datetime(int("19"+str(merged_df["granted_date"][index])[0:2]),int(str(merged_df["granted_date"][index])[2:4]),int(str(merged_df["granted_date"][index])[4:6]))
        merged_df["trans_date"][index] = datetime(int("19"+str(merged_df["trans_date"][index])[0:2]),int(str(merged_df["trans_date"][index])[2:4]),int(str(merged_df["trans_date"][index])[4:6]))

    x = merged_df["granted_date"]
    y = merged_df["trans_date"]

    plt.xlabel('Loan Granted Date')
    plt.ylabel('Transaction Date')

    plt.scatter(x, y)
    plt.savefig(get_correlation_folder('trans')/'loan_date.jpg')
    plt.clf()

def last_balance_loan():
    df = db.df_query('SELECT * FROM trans_train')
    df_loan = db.df_query('SELECT account_id,amount FROM loan_train')
    
    # transação mais recente feita
    df_trans = db.df_query('SELECT account_id, MAX(trans_date) AS trans_date FROM trans_train GROUP BY account_id')
   
    # visualização melhor das datas
    #for index, row in df_trans.iterrows():
    #    df_trans["trans_date"][index] = datetime(int("19"+str(df_trans["trans_date"][index])[0:2]),int(str(df_trans["trans_date"][index])[2:4]),int(str(df_trans["trans_date"][index])[4:6]))

    df_trans2 = pd.merge(df_trans, df, how="inner",on=['account_id','trans_date'])
    df_merged =pd.merge(df_trans2, df_loan, how="inner",on=['account_id'])

    print(df_merged)

    x = df_merged["balance"]
    y = df_merged["amount_y"]

    plt.xlabel('Account balance')
    plt.ylabel('Loan quantity')

    plt.scatter(x, y)
    plt.savefig(get_correlation_folder('trans')/'loan_balance.jpg')
    plt.clf()

def trans_operations():
    df = db.df_query('SELECT * FROM trans_train')

    # Rename columns and fill NaN
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    df["operation"].fillna("interest credited", inplace=True)
    df.loc[df["operation"]=="credit in cash", "operation"] = "CashC"
    df.loc[df["operation"]=="collection from anot", "operation"] = "Coll"
    df.loc[df["operation"]=="interest credited", "operation"] = "Interest"
    df.loc[df["operation"]=="withdrawal in cash", "operation"] = "CashW"
    df.loc[df["operation"]=="remittance to anothe", "operation"] = "Rem"
    df.loc[df["operation"]=="credit card withdraw", "operation"] = "CardW"

    # Get number of each operation for each account

    operation_df = df.groupby(['account_id', 'operation']).agg({'trans_id': ['count']}).reset_index()
    operation_df.columns = ['account_id', 'operation', 'operation_count']
    
    # credit in cash = CashC
    cashC_operation = operation_df[operation_df['operation'] == 'CashC']
    cashC_operation.columns = ['account_id', 'operation', 'num_cash_credit']
    cashC_operation = cashC_operation.drop(['operation'], axis=1)

    # collection from another bank = Coll
    coll_operation = operation_df[operation_df['operation'] == 'Coll']
    coll_operation.columns = ['account_id', 'operation', 'num_coll']
    coll_operation = coll_operation.drop(['operation'], axis=1)

    # interest credited = Interest,
    interest_operation = operation_df[operation_df['operation'] == 'Interest']
    interest_operation.columns = ['account_id', 'operation', 'num_interest']
    interest_operation = interest_operation.drop(['operation'], axis=1)

    # withdrawal in cash = CashW
    cashW_operation = operation_df[operation_df['operation'] == 'CashW']
    cashW_operation.columns = ['account_id', 'operation', 'num_cash_withdrawal']
    cashW_operation = cashW_operation.drop(['operation'], axis=1)

    # remittance to another bank = Rem
    rem_operation = operation_df[operation_df['operation'] == 'Rem']
    rem_operation.columns = ['account_id', 'operation','num_rem']
    rem_operation = rem_operation.drop(['operation'], axis=1)

    # credit card withdrawal = CardW
    cardW_operation = operation_df[operation_df['operation'] == 'CardW']
    cardW_operation.columns = ['account_id', 'operation', 'num_card_withdrawal']
    cardW_operation = cardW_operation.drop(['operation'], axis=1)
    
    operation_df = cashC_operation.merge(coll_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(interest_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(cashW_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(rem_operation, on='account_id',how='outer')
    operation_df = operation_df.merge(cardW_operation, on='account_id',how='outer')
    operation_df.fillna(0, inplace=True)

    operation_num = ['num_cash_credit','num_rem','num_card_withdrawal', 'num_cash_withdrawal', 'num_interest', 'num_coll']
    operation_df['total_operations'] = operation_df[operation_num].sum(axis=1)

    # Calculate Ratio for each operation
    operation_df['cash_credit_ratio'] = operation_df['num_cash_credit']/operation_df['total_operations']
    operation_df['rem_ratio'] = operation_df['num_rem']/operation_df['total_operations']
    operation_df['card_withdrawal_ratio'] = operation_df['num_card_withdrawal']/operation_df['total_operations']
    operation_df['cash_withdrawal_ratio'] = operation_df['num_cash_withdrawal']/operation_df['total_operations']
    operation_df['interest_ratio'] = operation_df['num_interest']/operation_df['total_operations']
    operation_df['coll_ratio'] = operation_df['num_coll']/operation_df['total_operations']

    # Join trans with loan
    loan_df = db.df_query('SELECT * FROM loan_train')  
    df = pd.merge(loan_df, operation_df, how="left",on="account_id")

    df_good = df.loc[df['loan_status'] == 1]
    df_bad = df.loc[df['loan_status'] == -1]

    operation_ratios = ['cash_credit_ratio','rem_ratio','card_withdrawal_ratio', 'cash_withdrawal_ratio', 'interest_ratio', 'coll_ratio']

    for operation in operation_ratios:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.histplot(ax=ax1, label='status 1', color='green', alpha=0.6, data=df_good, x=operation)
        sns.histplot(ax=ax2, label='status -1', color='red', alpha=0.6, data=df_bad, x=operation)

        ax1.set_xlim([0,1])
        ax2.set_xlim([0,1])
        ax1.set_ylim([0,100])
        ax2.set_ylim([0,100])

        # ax1.set_title('Average transaction balance of the accounts')
        # ax2.set_title('Average transaction balance of the accounts')
        ax1.legend()
        ax2.legend()

        ax1.xaxis.set_major_formatter(PercentFormatter(1))
        ax2.xaxis.set_major_formatter(PercentFormatter(1))

        name = operation+'_status.jpg'
        plt.savefig(get_correlation_folder('trans')/name)
        plt.clf()
    

if __name__ == '__main__':
    create_plots_folders('trans')
    trans_test_du()
    trans_train_du()

    num_trans_status()
    avg_amount_status()
    avg_balance_status()

    transactions_correlation_matrix()
    transactions_loan_date()
    last_balance_loan()

    trans_operations()
