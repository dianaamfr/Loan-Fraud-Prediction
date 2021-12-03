import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
from datetime import date
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

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

def calculate_age(birth_date, loan_date):
    frame = { 'birth': birth_date, 'granted': loan_date }
    dates = pd.DataFrame(frame)

    dates['birth'] = pd.to_datetime(dates['birth'], format='%Y-%m-%d')
    dates['granted'] = pd.to_datetime(dates['granted'], format='%Y-%m-%d')
    dates['difference'] = (dates['granted'] - dates['birth']).dt.days // 365

    return dates['difference'] 


def transform_status(df):
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    df["loan_status"].replace({1: 0}, inplace=True)
    df["loan_status"].replace({-1: 1}, inplace=True)


#######
# Clean
#######

def clean_loans(db, test=False):
    table = 'loan_test' if test is True else 'loan_train'
    df = db.df_query('SELECT * FROM ' + table)

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
    df.drop(columns=['disp_type'], inplace=True)

    return df

def clean_clients(db):
    df = db.df_query('SELECT * FROM client')

    # Gender and Date of birth
    df['gender'], df['birth_date'] = zip(*df['birth_number'].map(split_birth))
    df.drop(columns=['birth_number'], inplace=True)

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
    df = encode_category(df, 'region')
    
    return df

def clean_transactions(db, test=False):
    table = 'trans_test' if test is True else 'trans_train'
    df = db.df_query('SELECT * FROM ' + table)

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # Drop columns with more than x% of NaN values
    df.drop(columns=['bank', 'k_symbol'], inplace=True)
    # TODO - extract features from k_symbol instead of dropping it

    # TODO - extract features from operation instead of dropping it
    df["operation"].fillna("interest credited", inplace=True)
    df.drop(columns=['operation'], inplace=True)


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
    trans_type_count_df['withdrawal_ratio'] = trans_type_count_df['num_withdrawals'] / (trans_type_count_df['num_credits'] + trans_type_count_df['num_withdrawals'])

    trans_type_count_df.drop(columns=['num_credits', 'num_withdrawals'], inplace=True)
    new_df = pd.merge(new_df, trans_type_count_df, on="account_id", how="outer")

    # Average, Min, Max Balance & Num Transactions
    balance_count_df = db.df_query('SELECT account_id, AVG(balance) AS avg_balance, COUNT(trans_id) AS num_trans, '\
            ' MAX(balance) AS max_balance, MIN(balance) AS min_balance '\
            'FROM ' + table + ' GROUP BY account_id')
    balance_count_df['negative_balance'] = balance_count_df['min_balance'] < 0
    balance_count_df = encode_category(balance_count_df, 'negative_balance')

    # Last Transaction
    last_balance_df = db.df_query('SELECT account_id, AVG(balance) AS last_balance FROM '\
            + table + ' JOIN (SELECT account_id, max(trans_date) AS last_date FROM ' + table + ' GROUP BY account_id) AS last_date '\
            'USING(account_id) WHERE last_date = trans_date GROUP BY account_id')
    last_balance_df['last_balance_negative'] = last_balance_df['last_balance'] < 0
    last_balance_df = encode_category(last_balance_df, 'last_balance_negative')

    new_df = pd.merge(new_df, balance_count_df, on="account_id", how="outer")
    new_df = pd.merge(new_df, last_balance_df, on="account_id", how="outer")

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

    return df

def extract_features(df):

    # Age when the loan was requested
    df['age_when_loan'] = calculate_age(df['birth_date'], df['granted_date'])
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

    loan_train = db.df_query('SELECT granted_date, amount, duration, payments, loan_status FROM loan_train')
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    #transform_status(loan_train)
    loan_train.to_csv('clean_data/' + output_name + '-train.csv', index=False)

    loan_test = db.df_query('SELECT loan_id, granted_date, amount, duration, payments FROM loan_test')
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

    df_train.to_csv('clean_data/' + output_name + '-train.csv', index=False)

    ############
    # TEST DATA
    ############
    df_test = merge_datasets(db, True)
    df_test = extract_features(df_test)

    df_test.drop(columns=["account_id", "loan_status", "disp_id", "client_id", 
    "district_id", "account_district_id", "client_district_id"], inplace=True)

    df_test = df_test.set_index('loan_id')

    df_test.to_csv('clean_data/' + output_name + '-test.csv', index=True)


if __name__ == "__main__":
    clean(sys.argv[1])
