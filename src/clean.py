import pandas as pd
import numpy as np
from pandas.io.formats.format import DataFrameFormatter
from sklearn import preprocessing
from datetime import datetime, date
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go

#pd.set_option('display.max_columns', None)

#############
# Correlation
#############

def get_df_correlation(df, size=(11, 9)):
    '''Get the correlation between the dataframe features'''
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    plt.subplots(figsize=size)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                     square=True, linewidths=.1, cbar_kws={"shrink": .5})
    
    y_lim = ax.get_ylim()
    ax.set_ylim(np.ceil(y_lim[0]), np.floor(y_lim[1]))

    plt.show()

#######
# Utils
#######
def format_date(df, col, format='%y%m%d'):
    df[col] = pd.to_datetime(df[col], format=format)
    return df

def encode_category(df, col):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[col].unique())
    df[col] = encoder.transform(df[col])
    return df


def split_birth(birth_number):
    year = 1900 + (birth_number // 10000)
    month = (birth_number % 10000) // 100
    day = birth_number % 100

    # Male
    if month < 50:
       gender = 1
    # Female
    else:
        gender = 0
        month = month - 50
    
    birth_date = year*10000 + month*100 +day
    birth_date = pd.to_datetime(birth_date, format='%Y%m%d')

    return gender, birth_date

def calculate_age(birth_date):
    today = datetime.now()
    age = (today - birth_date).days // 365
    return age

def calculate_age_loan(birth_date, loan_date):
    frame = { 'birth': birth_date, 'granted': loan_date }
    dates = pd.DataFrame(frame)

    dates['birth'] = pd.to_datetime(dates['birth'], format='%Y-%m-%d')
    dates['granted'] = pd.to_datetime(dates['granted'], format='%Y-%m-%d')
    dates['difference'] = (dates['granted'] - dates['birth']).dt.days // 365

    return dates['difference']

def economic_features():
    # Average Balance, Num Transactions
    df1 = db.df_query(f'SELECT account_id, AVG(balance) AS average_balance, MIN(balance) AS min_balance, COUNT(trans_id) AS num_trans \
        FROM account JOIN trans_train USING(account_id) JOIN disposition USING(account_id) JOIN client USING(client_id) \
        GROUP BY account_id')

    # Amount Features
    df2 = db.df_query('SELECT amount, account_id, trans_type FROM trans_train')
    df2.loc[df2["trans_type"]=="withdrawal", "amount"] *= -1
    amount = df2.groupby(['account_id']).agg({'amount':['mean','min', 'max']}).reset_index()
    amount.columns = ['account_id', 'avg_amount', 'min_amount', 'max_amount']

    df_economic = pd.merge(df1, amount, on='account_id', how='left')

    # Ratio of credits
    df4 = db.df_query('SELECT trans_type, account_id FROM trans_train')
    type_counts = df4.groupby(['account_id', 'trans_type']).size().reset_index(name='counts')

    credit_counts = type_counts[type_counts['trans_type'] == 'credit']
    credit_counts.columns = ['account_id', 'trans_type', 'num_credits']
    credit_counts = credit_counts.drop(columns=["trans_type"])

    withdrawal_counts = type_counts[type_counts['trans_type'] == 'withdrawal']
    withdrawal_counts.columns = ['account_id', 'trans_type', 'num_withdrawals']
    withdrawal_counts = withdrawal_counts.drop(columns=["trans_type"])

    trans_type_count_df = pd.merge(credit_counts, withdrawal_counts, on="account_id", how="outer")
    trans_type_count_df.fillna(0, inplace=True)
    trans_type_count_df['credit_ratio'] = trans_type_count_df['num_credits'] / (trans_type_count_df['num_credits'] + trans_type_count_df['num_withdrawals'])
    trans_type_count_df.drop(columns=['num_credits', 'num_withdrawals'], inplace=True)

    df_economic = pd.merge(df_economic, trans_type_count_df, on='account_id', how='left')


    df3 = db.df_query('SELECT operation, account_id, trans_id FROM trans_train')

    # Operation Nan and rename
    df3["operation"].fillna("interest credited", inplace=True)
    df3.loc[df3["operation"]=="credit in cash", "operation"] = "CashC"
    df3.loc[df3["operation"]=="collection from anot", "operation"] = "Coll"
    df3.loc[df3["operation"]=="interest credited", "operation"] = "Interest"
    df3.loc[df3["operation"]=="withdrawal in cash", "operation"] = "CashW"
    df3.loc[df3["operation"]=="remittance to anothe", "operation"] = "Rem"
    df3.loc[df3["operation"]=="credit card withdraw", "operation"] = "CardW"

    operation = df3.groupby(['account_id', 'operation']).agg({'trans_id': ['count']}).reset_index()
    operation.columns = ['account_id', 'operation','operation_count']
    
    # credit in cash = CashC
    cashC_operation = operation[operation['operation'] == 'CashC']
    cashC_operation.columns = ['account_id', 'operation', 'num_cash_credit']
    cashC_operation = cashC_operation.drop(['operation'], axis=1)

    # collection from another bank = Coll
    coll_operation = operation[operation['operation'] == 'Coll']
    coll_operation.columns = ['account_id', 'operation',  'num_coll']
    coll_operation = coll_operation.drop(['operation'], axis=1)

    # interest credited = Interest,
    interest_operation = operation[operation['operation'] == 'Interest']
    interest_operation.columns = ['account_id', 'operation',  'num_interest']
    interest_operation = interest_operation.drop(['operation'], axis=1)

    # withdrawal in cash = CashW
    cashW_operation = operation[operation['operation'] == 'CashW']
    cashW_operation.columns = ['account_id', 'operation', 'num_cash_withdrawal']
    cashW_operation = cashW_operation.drop(['operation'], axis=1)

    # remittance to another bank = Rem
    rem_operation = operation[operation['operation'] == 'Rem']
    rem_operation.columns = ['account_id', 'operation', 'num_rem']
    rem_operation = rem_operation.drop(['operation'], axis=1)

    # credit card withdrawal = CardW
    cardW_operation = operation[operation['operation'] == 'CardW']
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

    operation_df.drop(columns=operation_num, inplace=True)
    operation_df.drop(columns=['total_operations'], inplace=True)

    df_economic = pd.merge(df_economic, operation_df, on="account_id", how="outer")
    
    return df_economic

def transform_status(df):
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    df["loan_status"].replace({1: 0}, inplace=True)
    df["loan_status"].replace({-1: 1}, inplace=True)


#######
# Clean
#######

def clean_loans(db, test=False):
    table = 'loan_test' if test is True else 'loan_train'
    df = db.df_query('SELECT loan_id, granted_date, amount, payments, loan_status, account_id FROM ' + table)

    # Format loan date
    df = format_date(df, 'granted_date')
   
    return df

def clean_accounts(db):
    df = db.df_query("SELECT * FROM account")
  
    # Format creation date
    df = format_date(df, 'creation_date')
    
    # Encode frequency 
    df = encode_category(df, 'frequency')

    df.rename(columns={'district_id': 'account_district_id'}, inplace=True)

    return df

def clean_disp(db):
    # Keep only the owner of the account, because only the owner can ask for a loan
    df = db.df_query("SELECT * FROM disposition WHERE disp_type = 'OWNER'")

    has_disponent = db.df_query("SELECT account_id, COUNT(disp_id) AS n_disponents FROM disposition WHERE disp_type = 'DISPONENT' GROUP BY account_id")
    df = pd.merge(df, has_disponent, on="account_id", how="left")
    df.fillna(0, inplace=True)

    df['has_disponent'] = df['n_disponents'] > 0
    df.drop(columns=['disp_type', 'n_disponents'], inplace=True)
    df = encode_category(df, 'has_disponent')

    return df

def clean_clients(db):
    df = db.df_query('SELECT * FROM client')

    # Gender and Date of birth
    df['gender'], df['birth_date'] = zip(*df['birth_number'].map(split_birth))
    
    df.rename(columns={'district_id': 'client_district_id'}, inplace=True)
    return df

def clean_districts(db):
    df = db.df_query('SELECT * FROM district')
   
    # REPLACE NANs
    df["nr_commited_crimes_95"] = pd.to_numeric(df["nr_commited_crimes_95"], errors='coerce')
    df["unemployment_rate_95"] = pd.to_numeric(df["unemployment_rate_95"], errors='coerce')
    
    # Replace nr of crimes and unemployment missing values by median
    # TODO: maybe try to extract the equation of the relations between 96 and 95 and use it instead
    median_nr_crimes_95 = df["nr_commited_crimes_95"].median(skipna=True)
    median_unemployment_rate_95 = df["unemployment_rate_95"].median(skipna=True)
    # mean_nr_crimes_95 = df["nr_commited_crimes_95"].mean(skipna=True)
    # mean_unemployment_rate_95 = df["unemployment_rate_95"].mean(skipna=True)

    # df["nr_commited_crimes_95"].fillna(median_nr_crimes_95, inplace=True)
    # df["unemployment_rate_95"].fillna(median_unemployment_rate_95, inplace=True)
    
    # Fill the missing values with the mean
    # df["nr_commited_crimes_95"].fillna(mean_nr_crimes_95, inplace=True)
    # df["unemployment_rate_95"].fillna(mean_unemployment_rate_95, inplace=True)

    # Replace nr of crimes and unemployment missing values by 96 values
    df["nr_commited_crimes_95"].fillna(df['nr_commited_crimes_96'], inplace=True)
    df["unemployment_rate_95"].fillna(df['unemployment_rate_96'], inplace=True)

    # FEATURE EXTRACTION

    # Entrepeneurs Ratio
    df['ratio_entrepeneurs'] = df['nr_enterpreneurs_1000_inhabitants'] / 1000

    # Average Crimes
    df["avg_crimes"] = df[['nr_commited_crimes_95', 'nr_commited_crimes_96']].mean(axis=1) / df["nr_inhabitants"]
    
    # From percentage to ratio
    df['ratio_urban_inhabitants'] = df['ratio_urban_inhabitants']/100

    # Average Unemployment
    df["avg_unemployment"] = df[['unemployment_rate_95', 'unemployment_rate_96']].mean(axis=1)

    # Criminality Growth
    df['criminality_growth'] = (df["nr_commited_crimes_96"] - df['nr_commited_crimes_95']) / df['nr_inhabitants']

    # Unemployment Growth
    df['unemployment_growth'] = df['unemployment_rate_96'] - df['unemployment_rate_95']

    # Drop
    df.drop(columns=['nr_commited_crimes_95','nr_commited_crimes_96','unemployment_rate_95','unemployment_rate_96', 'nr_enterpreneurs_1000_inhabitants', 'district_name'], inplace=True)

    # Encode Region
    # df = encode_category(df, 'region')
    
    return df

def clean_transactions(db, test=False):
    table = 'trans_test' if test is True else 'trans_train'
    df = db.df_query('SELECT * FROM ' + table)

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # Drop bank column - more than 50% NaN values
    df.drop(columns=['bank'], inplace=True)

    # Operation Nan and rename
    df["operation"].fillna("interest credited", inplace=True)
    df.loc[df["operation"]=="credit in cash", "operation"] = "CashC"
    df.loc[df["operation"]=="collection from anot", "operation"] = "Coll"
    df.loc[df["operation"]=="interest credited", "operation"] = "Interest"
    df.loc[df["operation"]=="withdrawal in cash", "operation"] = "CashW"
    df.loc[df["operation"]=="remittance to anothe", "operation"] = "Rem"
    df.loc[df["operation"]=="credit card withdraw", "operation"] = "CardW"

    # K_symbol Nan and rename
    df["k_symbol"].fillna("None", inplace=True)
    df.loc[df["k_symbol"]=="insurrance payment", "k_symbol"] = "Insurance"
    df.loc[df["k_symbol"]=="interest credited", "k_symbol"] = "Interest"
    df.loc[df["k_symbol"]=="household", "k_symbol"] = "Household"
    df.loc[df["k_symbol"]=="payment for statemen", "k_symbol"] = "Statement"
    df.loc[df["k_symbol"]=="sanction interest if", "k_symbol"] = "Sanction"
    df.loc[df["k_symbol"]=="old-age pension", "k_symbol"] = "Pension"

    # TYPE & AMOUNT
    # Rename withdrawal in cash - wrong label
    df.loc[df['trans_type'] == 'withdrawal in cash','trans_type'] = 'withdrawal'

    # Make withdrawal amount negative
    df.loc[df["trans_type"]=="withdrawal", "amount"] *= -1

    # Format date
    df = format_date(df, 'trans_date')

    df_copy = df.copy()
    # FEATURE EXTRACTION

    # Average Amount by type
    avg_amount_type = df_copy.groupby(['account_id', 'trans_type']).agg({'amount':['mean']}).reset_index()
    avg_amount_type.columns = ['account_id', 'trans_type', 'avg_amount']

    avg_amount_credit = avg_amount_type[avg_amount_type['trans_type'] == 'credit']
    avg_amount_credit.columns = ['account_id', 'trans_type', 'avg_amount_credit']
    avg_amount_credit = avg_amount_credit.drop(columns=["trans_type"])

    avg_amount_withdrawal = avg_amount_type[avg_amount_type['trans_type'] == 'withdrawal']
    avg_amount_withdrawal.columns = ['account_id', 'trans_type', 'avg_amount_withdrawal']
    avg_amount_withdrawal = avg_amount_withdrawal.drop(columns=["trans_type"])

    avg_amount_df = pd.merge(avg_amount_credit, avg_amount_withdrawal, on="account_id", how="outer")
    avg_amount_df.fillna(0, inplace=True)

    avg_amount_total = df_copy.groupby(['account_id']).agg({'amount':['mean','min', 'max']}).reset_index()
    avg_amount_total.columns = ['account_id', 'avg_amount_total', 'min_amount', 'max_amount']
    new_df = pd.merge(avg_amount_df, avg_amount_total, on="account_id", how="outer")
    new_df.fillna(0, inplace=True)


    # Number of withdrawals and credits
    type_counts = df_copy.groupby(['account_id', 'trans_type']).size().reset_index(name='counts')

    credit_counts = type_counts[type_counts['trans_type'] == 'credit']
    credit_counts.columns = ['account_id', 'trans_type', 'num_credits']
    credit_counts = credit_counts.drop(columns=["trans_type"])

    withdrawal_counts = type_counts[type_counts['trans_type'] == 'withdrawal']
    withdrawal_counts.columns = ['account_id', 'trans_type', 'num_withdrawals']
    withdrawal_counts = withdrawal_counts.drop(columns=["trans_type"])

    trans_type_count_df = pd.merge(credit_counts, withdrawal_counts, on="account_id", how="outer")
    trans_type_count_df.fillna(0, inplace=True)
    trans_type_count_df['credit_ratio'] = trans_type_count_df['num_credits'] / (trans_type_count_df['num_credits'] + trans_type_count_df['num_withdrawals'])
    #trans_type_count_df['withdrawal_ratio'] = trans_type_count_df['num_withdrawals'] / (trans_type_count_df['num_credits'] + trans_type_count_df['num_withdrawals'])

    trans_type_count_df.drop(columns=['num_credits', 'num_withdrawals'], inplace=True)
    new_df = pd.merge(new_df, trans_type_count_df, on="account_id", how="outer")

    # Average, Min, Max Balance & Num Transactions
    balance_count_df = db.df_query('SELECT account_id, AVG(balance) AS avg_balance, COUNT(trans_id) AS num_trans, '\
            ' MIN(balance) AS min_balance, STDDEV(balance) AS std_balance '\
            'FROM ' + table + ' GROUP BY account_id')
    balance_count_df['negative_balance'] = balance_count_df['min_balance'] < 0
    #balance_count_df.drop(columns=['min_balance'], inplace=True)
    balance_count_df = encode_category(balance_count_df, 'negative_balance')

    # Last Transaction
    last_balance_df = db.df_query('SELECT account_id, AVG(balance) AS last_balance FROM '\
            + table + ' JOIN (SELECT account_id, max(trans_date) AS last_date FROM ' + table + ' GROUP BY account_id) AS last_date '\
            'USING(account_id) WHERE last_date = trans_date GROUP BY account_id')
    last_balance_df['last_balance_negative'] = last_balance_df['last_balance'] < 0
    last_balance_df = encode_category(last_balance_df, 'last_balance_negative')

    new_df = pd.merge(new_df, balance_count_df, on="account_id", how="outer")
    new_df = pd.merge(new_df, last_balance_df, on="account_id", how="outer")

    ###########
    # Operation
    ###########
    operation_amount = df_copy.groupby(['account_id', 'operation']).agg({'amount': ['mean', 'count']}).reset_index()
    operation_amount.columns = ['account_id', 'operation', 'operation_amount_mean', 'operation_amount_count']
    
    # credit in cash = CashC
    cashC_operation = operation_amount[operation_amount['operation'] == 'CashC']
    cashC_operation.columns = ['account_id', 'operation', 'mean_cash_credit', 'num_cash_credit']
    cashC_operation = cashC_operation.drop(['operation'], axis=1)

    # collection from another bank = Coll
    coll_operation = operation_amount[operation_amount['operation'] == 'Coll']
    coll_operation.columns = ['account_id', 'operation', 'mean_coll', 'num_coll']
    coll_operation = coll_operation.drop(['operation'], axis=1)

    # interest credited = Interest,
    interest_operation = operation_amount[operation_amount['operation'] == 'Interest']
    interest_operation.columns = ['account_id', 'operation', 'mean_interest', 'num_interest']
    interest_operation = interest_operation.drop(['operation'], axis=1)

    # withdrawal in cash = CashW
    cashW_operation = operation_amount[operation_amount['operation'] == 'CashW']
    cashW_operation.columns = ['account_id', 'operation', 'mean_cash_withdrawal', 'num_cash_withdrawal']
    cashW_operation = cashW_operation.drop(['operation'], axis=1)

    # remittance to another bank = Rem
    rem_operation = operation_amount[operation_amount['operation'] == 'Rem']
    rem_operation.columns = ['account_id', 'operation', 'mean_rem', 'num_rem']
    rem_operation = rem_operation.drop(['operation'], axis=1)

    # credit card withdrawal = CardW
    cardW_operation = operation_amount[operation_amount['operation'] == 'CardW']
    cardW_operation.columns = ['account_id', 'operation', 'mean_card_withdrawal', 'num_card_withdrawal']
    cardW_operation = cardW_operation.drop(['operation'], axis=1)
    
    operation_amount_df = cashC_operation.merge(coll_operation, on='account_id',how='outer')
    operation_amount_df = operation_amount_df.merge(interest_operation, on='account_id',how='outer')
    operation_amount_df = operation_amount_df.merge(cashW_operation, on='account_id',how='outer')
    operation_amount_df = operation_amount_df.merge(rem_operation, on='account_id',how='outer')
    operation_amount_df = operation_amount_df.merge(cardW_operation, on='account_id',how='outer')
    operation_amount_df.fillna(0, inplace=True)
    
    # new_df = pd.merge(new_df, operation_amount_df, on="account_id", how="outer")

    # K-Symbol
    symbol_amount = df_copy.groupby(['account_id', 'k_symbol']).agg({'amount': ['mean', 'count']}).reset_index()
    symbol_amount.columns = ['account_id', 'k_symbol', 'symbol_amount_mean', 'symbol_amount_count']

    no_symbol = symbol_amount[symbol_amount['k_symbol'] == 'None']
    no_symbol.columns = ['account_id', 'k_symbol', 'mean_no_symbol', 'num_no_symbol']
    no_symbol = no_symbol.drop(['k_symbol'], axis=1)

    household_symbol = symbol_amount[symbol_amount['k_symbol'] == 'Household']
    household_symbol.columns = ['account_id', 'k_symbol', 'mean_household', 'num_household']
    household_symbol = household_symbol.drop(['k_symbol'], axis=1)
    
    statement_symbol = symbol_amount[symbol_amount['k_symbol'] == 'Statement']
    statement_symbol.columns = ['account_id', 'k_symbol', 'mean_statement', 'num_statement']
    statement_symbol = statement_symbol.drop(['k_symbol'], axis=1)

    insurance_symbol = symbol_amount[symbol_amount['k_symbol'] == 'Insurance']
    insurance_symbol.columns = ['account_id', 'k_symbol', 'mean_insurance', 'num_insurance']
    insurance_symbol = insurance_symbol.drop(['k_symbol'], axis=1)

    sanction_symbol = symbol_amount[symbol_amount['k_symbol'] == 'Sanction']
    sanction_symbol.columns = ['account_id', 'k_symbol', 'mean_sanction', 'num_sanction']
    sanction_symbol = sanction_symbol.drop(['k_symbol'], axis=1)

    pension_symbol = symbol_amount[symbol_amount['k_symbol'] == 'Pension']
    pension_symbol.columns = ['account_id', 'k_symbol', 'mean_pension', 'num_pension']
    pension_symbol = pension_symbol.drop(['k_symbol'], axis=1)

    symbol_amount_df = no_symbol.merge(household_symbol, on='account_id',how='outer')
    symbol_amount_df = symbol_amount_df.merge(statement_symbol, on='account_id',how='outer')
    symbol_amount_df = symbol_amount_df.merge(insurance_symbol, on='account_id',how='outer')
    symbol_amount_df = symbol_amount_df.merge(sanction_symbol, on='account_id',how='outer')
    symbol_amount_df = symbol_amount_df.merge(pension_symbol, on='account_id',how='outer')
    symbol_amount_df.fillna(0, inplace=True)

    # new_df = pd.merge(new_df, symbol_amount_df, on="account_id", how="outer")

    return new_df


def clean_cards(db, test=False):
    loan_table = 'loan_test' if test is True else 'loan_train'
    card_table = 'card_test' if test is True else 'card_train'
    
    df_card = db.df_query('SELECT account_id, COUNT(card_id) AS num_cards FROM ' + loan_table + \
        ' JOIN disposition USING(account_id) LEFT JOIN ' + card_table + ' USING(disp_id) GROUP BY account_id')
    
    return df_card


#########
# Merge
#########

def merge_datasets(db, test=False):
    loan = clean_loans(db, test)
    account = clean_accounts(db)
    disp = clean_disp(db)
    client = clean_clients(db)
    district = clean_districts(db)
    transaction = clean_transactions(db, test)
    cards = clean_cards(db, test)
    
    df = pd.merge(loan, account, on='account_id', how="left")
    df = pd.merge(df, disp,  on='account_id', how="left")
    df = pd.merge(df, client,  on='client_id', how="left")
    # TODO - fazer 2 gráficos para escolher os dados do district relativos à account ou ao cliente
    df = pd.merge(df, district, left_on='client_district_id', right_on='district_id')
    df = pd.merge(df, transaction, how="left", on="account_id")
    df = pd.merge(df, cards, how="left", on="account_id")

    df.drop(columns=['birth_number'], inplace=True)
    
    return df

def extract_features(df):

    # Age when the loan was requested
    df['age_when_loan'] = calculate_age_loan(df['birth_date'], df['granted_date'])
    df.drop(columns=['birth_date'], inplace=True)

    # Days between loan and account creation
    df['days_between'] = (df['granted_date'] - df['creation_date']).dt.days
    df.drop(columns=['creation_date', 'granted_date'], inplace=True)

    # Boolean value telling if the account was created on the owner district
    df['same_district'] = df['account_district_id'] == df['client_district_id']

    # Has card
    df['has_card'] = df['num_cards'] > 0
    df.drop(columns=['num_cards'], inplace=True)

    return df


#############
# Submissions
#############

# Submission 1 - only loan
def first_submission(output_name):

    loan_train = db.df_query('SELECT granted_date, amount, payments, loan_status FROM loan_train')
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    #transform_status(loan_train)
    loan_train.to_csv('clean_data/' + output_name + '-train.csv', index=False)

    loan_test = db.df_query('SELECT loan_id, granted_date, amount, payments FROM loan_test')
    loan_test.to_csv('clean_data/' + output_name + '-test.csv', index=False)


# For other submissions
def clean(output_name):

    ############
    # TRAIN DATA
    ############
    df_train = merge_datasets(db)
    df_train = extract_features(df_train)

    df_train.drop(columns=["account_id", "disp_id", "client_id", "district_id",
    "account_district_id", "client_district_id"], inplace=True)

    # Reorder columns
    cols = list(df_train.columns)
    a, b = cols.index('same_district'), cols.index('loan_status')
    cols[b], cols[a] = cols[a], cols[b]
    df_train = df_train[cols]

    df_train = df_train.set_index('loan_id')

    transform_status(df_train)
    print(len(df_train.columns))
    df_train.to_csv('clean_data/' + output_name + '-train.csv', index=False)

    ############
    # TEST DATA
    ############
    df_test = merge_datasets(db, True)
    df_test = extract_features(df_test)

    df_test.drop(columns=["account_id", "loan_status", "disp_id", "client_id", 
    "district_id", "account_district_id", "client_district_id"], inplace=True)

    df_test = df_test.set_index('loan_id')

    print(df_test.columns)

    df_test.to_csv('clean_data/' + output_name + '-test.csv', index=True)


#############
# CLUSTERING
#############
def clustering_agglomerative():
    clients = clean_clients(db)
    district = clean_districts(db)
    loan = clean_loans(db)

    district['region'] = district.apply(lambda x: get_cardinal_point(x['region']), axis=1)
    
    disp = db.df_query('SELECT * FROM disposition')
    
    df = pd.merge(loan, disp, on='account_id', how="left")
    df = pd.merge(df, clients, on='client_id', how="left")
    df = pd.merge(df, district, left_on="client_district_id", right_on="district_id", how="left")

    df_economic = economic_features()
    df = pd.merge(df, df_economic, on='account_id', how="left")
    df.fillna(0, inplace=True)

    # Maybe use the age of the client when the loan was issued
    df['age'] = df['birth_date'].apply(lambda x: calculate_age(x))

    df = df[['average_balance', 'age', 'num_trans']]

    # Create Dendrogram
    #dendrogram = sch.dendrogram(sch.linkage(df, method='ward'))
    # plt.savefig('dendogram.jpg')
    # plt.clf()

    # Create Clusters
    hc = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward')
    data = df.values

    # save clusters for chart
    labels = hc.fit_predict(data)

    # plt.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')

    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[labels == 0,0],data[labels == 0,1],data[labels == 0,2], s = 40 , color = 'orange', label = "cluster 1", alpha=0.6)
    ax.scatter(data[labels == 1,0],data[labels == 1,1],data[labels == 1,2], s = 40 , color = 'green', label = "cluster 2", alpha=0.6)
    ax.scatter(data[labels == 2,0],data[labels == 2,1],data[labels == 2,2], s = 40 , color = 'red', label = "cluster 3")
    ax.set_xlabel('Rem Ratio-->')
    ax.set_ylabel('Average Salary->')
    ax.set_zlabel('Min Balance-->')
    ax.legend()
    plt.show()

def get_cardinal_point(region):
    cardinal_points = ['south','west','north','east', 'central']
    for cp in cardinal_points:
        if cp in region:
            return cp
    return 'central'

def clustering_kmeans():

    clients = clean_clients(db)
    district = clean_districts(db)
    loan = clean_loans(db)

    district['region'] = district.apply(lambda x: get_cardinal_point(x['region']), axis=1)
    
    disp = db.df_query('SELECT * FROM disposition')
    
    df = pd.merge(loan, disp, on='account_id', how="left")
    df = pd.merge(df, clients, on='client_id', how="left")
    df = pd.merge(df, district, left_on="client_district_id", right_on="district_id", how="left")

    df_economic = economic_features()
    df = pd.merge(df, df_economic, on='account_id', how="left")
    df.fillna(0, inplace=True)

    # Maybe use the age of the client when the loan was issued
    df['age'] = df['birth_date'].apply(lambda x: calculate_age(x))

    # Create Dendrogram
    # dendrogram = sch.dendrogram(sch.linkage(df, method='ward'))
    # plt.savefig('dendogram.jpg')
    # plt.clf()

    # rem_ration - coll ration OK
    # rem_ration - age 
    # average_balance - coll_ratio
    # average_balance - cash_withdrawal_ratio
    # max_amount - cash_withdrawal_ratio OK with average salary and age
    # max_amount - credit_withdrawal_ratio
    # max_amount - card_withdrawal_ratio
    # max_amount - credit_raio
    # avg_balance - credit_ratio - nr_ent UNA MIERDA
    # num_trans - credit_ratio - ratio_ent

    # plt.scatter(df['num_trans'],df['credit_ratio'])
    # plt.show()
    # plt.clf()

    
    df = encode_category(df, 'disp_type')
    df = df[['amount', 'payments', 'loan_status', 'num_trans', 'avg_amount', 'disp_type', 'district_id', 'average_salary', 'unemployment_growth', 'ratio_entrepeneurs', 'credit_ratio', 'age']]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(df)
    
    kmeans = KMeans(3)
    identified_clusters = kmeans.fit_predict(x)

    print(f'Inertia: {kmeans.inertia_}')

    df.insert(loc=0, column='cluster', value=identified_clusters.tolist())

    # pd.concat([k1, k2, k3]).groupby('cluster').mean()

    # k1,k2,k3 = [x[np.where(kmeans.labels_==i)] for i in range(3)] # range(3) because 3 clusters

    k1,k2,k3 = [df.loc[df['cluster']==i] for i in range(3)] # range(3) because 3 clusters

    print(type(k1))

    print(len(k1))
    print(len(k2))
    print(len(k3))

    print(k1.describe())
    print(k2.describe())
    print(k3.describe())

    # # PCA
    # reduced_data = PCA(n_components=2).fit_transform(df)
    # results = pd.DataFrame(reduced_data,columns=['pca1', 'pca2'])

    # fig = plt.figure(figsize=(6,6))
    # ax = Axes3D(fig, auto_add_to_figure=False)
    # fig.add_axes(ax)

    plt.scatter(x=df["age"], y=df["amount"],s=40, marker='o', alpha=1, c=df['cluster'])
    plt.show()
    plt.clf()

    nr_clusters = []
    inertias = []
    scores = []
    range_values = np.arange(2,11)
    for k in range_values:
        kmeans = KMeans(k)
        kmeans.fit(x)
        nr_clusters.append(k)
        inertias.append(kmeans.inertia_)
        score = metrics.silhouette_score(x, kmeans.labels_, metric='euclidean', sample_size=len(x))
        print('Silhouette score =', score)
        scores.append(score)

    plt.plot(nr_clusters, inertias)
    plt.title('Evolution of Inertia with number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    
    plt.figure()
    plt.bar(range_values, scores, width=0.6, color='k', align='center')
    plt.title('Silhouette score vs number of clusters')
    

    
    # 3D

    # fig = plt.figure(figsize = (15,15))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[identified_clusters == 0,0],x[identified_clusters == 0,1],x[identified_clusters == 0,2], s = 40 , color = 'orange', label = "cluster 1", alpha=0.6)
    # ax.scatter(x[identified_clusters == 1,0],x[identified_clusters == 1,1],x[identified_clusters == 1,2], s = 40 , color = 'green', label = "cluster 2", alpha=0.6)
    # ax.scatter(x[identified_clusters == 2,0],x[identified_clusters == 2,1],x[identified_clusters == 2,2], s = 40 , color = 'red', label = "cluster 3", alpha=0.6)
    # ax.set_xlabel('max_amount -->')
    # ax.set_ylabel('cash withdrawal -->')
    # ax.set_zlabel('age -->')
    # ax.legend()
    # plt.show()


def clustering_dbscan():

    clients = clean_clients(db)
    district = clean_districts(db)
    loan = clean_loans(db)

    district['region'] = district.apply(lambda x: get_cardinal_point(x['region']), axis=1)
    
    disp = db.df_query('SELECT * FROM disposition')
    
    df = pd.merge(loan, disp, on='account_id', how="left")
    df = pd.merge(df, clients, on='client_id', how="left")
    df = pd.merge(df, district, left_on="client_district_id", right_on="district_id", how="left")

    df_economic = economic_features()
    df = pd.merge(df, df_economic, on='account_id', how="left")
    df.fillna(0, inplace=True)

    # Maybe use the age of the client when the loan was issued
    df['age'] = df['birth_date'].apply(lambda x: calculate_age(x))
    
    print(df.columns)
    print("----------------------------------------")

    df = df[['amount', 'payments', 'num_trans', 'avg_amount', 'average_salary', 'unemployment_growth', 'ratio_entrepeneurs', 'credit_ratio', 'age']]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(df)

    # pca = PCA(n_components=7)
    # pca.fit(x)
    # variance = pca.explained_variance_ratio_ 
    # var=np.cumsum(np.round(variance, 3)*100)
    # plt.figure(figsize=(12,6))
    # plt.ylabel('% Variance Explained')
    # plt.xlabel('# of Features')
    # plt.title('PCA Analysis')
    # plt.ylim(0,100.5)
    # plt.plot(var)
    # plt.show()

    pca = PCA(n_components=3)
    pca.fit(x)
    pca_scale = pca.transform(x)
    pca_df = pd.DataFrame(pca_scale, columns=['pc1', 'pc2', 'pc3'])
    print(pca.explained_variance_ratio_)
    
    dbscan = DBSCAN(eps=1.0, min_samples=4).fit(pca_df)
    labels = dbscan.labels_

    Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
    # model.labels_ is nothing but the predicted clusters i.e y_clusters
    labels = dbscan.labels_
    trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
    layout = go.Layout(scene = Scene, height = 1000,width = 1000)
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    fig.update_layout(title="'DBSCAN Clusters (6) Derived from PCA'", font=dict(size=12,))
    fig.show()

if __name__ == "__main__":
    # clean(sys.argv[1])
    clustering_kmeans()
    #clustering_agglomerative()
    #clustering_dbscan()