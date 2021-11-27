import pandas as pd
import category_encoders as ce
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import datetime
from datetime import date
import sys

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

#pd.set_option('display.max_columns', None)

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

# TODO - use this to get age of client on loan
def calculate_age(birth_date, loan_date):
    birth_date = datetime.datetime.strptime(birth_date, "%Y%m%d")
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


# TODO - delete and use encode category isntead for each individual column
def transform_numeric_categorical(df):
    num_cols = ['amount', 'duration', 'payments', 'avg_amount','avg_balance','num_trans','days_between', 'age',
        'avg_crimes', 'avg_unemployment', 'nr_inhabitants', 'average_salary', 'nr_enterpreneurs_1000_inhabitants']
    cat_cols = ['frequency', 'gender', 'same_district'] # TODO: card type

    # Normalize
    scaler = MinMaxScaler()
    scaler.fit(df[num_cols])
    df[num_cols] = pd.DataFrame(scaler.transform(df[num_cols]), index=df[num_cols].index, columns=df[num_cols].columns)

    # Encode Categorical Variables
    encoder= ce.OrdinalEncoder(cols=[cat_cols],return_df=True,mapping=[
        {'col':'gender', 'mapping':{'M':0,'F':1}},
        {'col':'same_district', 'mapping':{True:1,False:0}},
        {'col':'frequency', 'mapping':{'monthly issuance':0,'weekly issuance':0.5, 'issuance after transaction':1}}])
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    return df

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
    
    # TODO - check if the median is the best value to use
    median_nr_crimes_95 = df["nr_commited_crimes_95"].median(skipna=True)
    median_unemployment_rate_95 = df["unemployment_rate_95"].median(skipna=True)

    df["nr_commited_crimes_95"].fillna(median_nr_crimes_95, inplace=True)
    df["unemployment_rate_95"].fillna(median_unemployment_rate_95, inplace=True)

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

    # TODO - maybe define which features to keep with algorithm and not randomly
    """
    region VARCHAR(20) NOT NULL,
    nr_inhabitants INT NOT NULL,
    nr_municip_inhabitants_499 INT NOT NULL,
    nr_municip_inhabitants_500_1999 INT NOT NULL,
    nr_municip_inhabitants_2000_9999 INT NOT NULL,
    nr_municip_inhabitants_10000 INT NOT NULL,
    nr_cities INT NOT NULL,
    """

    return df

def clean_transactions(db, test=False):
    table = 'trans_test' if test is True else 'trans_train'

    df = db.df_query('SELECT account_id, AVG(amount) AS avg_amount, AVG(balance) AS avg_balance, COUNT(trans_id) AS num_trans '\
        'FROM ' + table + ' GROUP BY account_id')

    # IN DEVELOPMENT
    # df = db.df_query('SELECT * FROM ' + table)

    # df = df.replace(r'^\s*$', np.NaN, regex=True)
    # # print(df.isna().mean())
    # # Drop columns with more than x% of NaN values
    # # TODO - check if k_symbol should be dropped
    # df.drop(columns=['bank'], inplace=True)

    # # Format date
    # df = format_date(df, 'trans_date')

    # # Rename withdrawal in cash - wrong label
    # df.loc[df['type'] == 'withdrawal in cash','type'] = 'withdrawal'

    # # TODO - check if the fact that a transaction is a credit or withdrawal should be considered (negative conversion)
    # # df.loc[df["type"]=="withdrawal", "amount"] *=-1


    # # FEATURE EXTRACTION
    # # Get average amount, balance and number of transactions

    return df

def clean_cards(db, test=False):
    table = 'card_test' if test is True else 'card_train'
    df = db.df_query("SELECT * FROM " + table)

    # Format issuance date
    df['issued'] = df['issued'].apply(lambda date: int('19' + str(date)))

    return df


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
    card = clean_cards(db, test)
    
    df = pd.merge(loan, account, on='account_id', how="left")
    df = pd.merge(df, disp,  on='account_id', how="left")
    df = pd.merge(df, client,  on='client_id', how="left")
    # TODO - fazer 2 gráficos para mostrar pq que escolhemos os dados do district relativos à account e não ao client ou vice versa; Experimentar escolher os do cliente;
    df = pd.merge(df, district, left_on='account_district_id', right_on='district_id') # associate account's district
    df = pd.merge(df, transaction, how="left", on="account_id")
    # TODO - merge CARD.

    return df

def extract_features(df):
    # Days between loan and account creation
    df['days_between'] = df['granted_date'] - df['creation_date']
    df.drop(columns=['creation_date', 'granted_date'], inplace=True)

    # Boolean value telling if the account was created on the owner district
    df['same_district'] = df['account_district_id'] == df['client_district_id']

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

    df_train.drop(columns=["account_id", "loan_id", "disp_id", "client_id", 
    "account_district_id", "client_district_id"], inplace=True)
    
    # transform_status(df)
    df = transform_numeric_categorical(df_train)

    df.to_csv('clean_data/' + output_name + '-train.csv', index=False)


    ############
    # TEST DATA
    ############
    df_test = merge_datasets(db, True)
    df_test = extract_features(df_test)

    df_test.drop(columns=["account_id", "loan_status", "disp_id", "client_id", 
    "district_id", "account_district_id", "client_district_id"], inplace=True)

    df_test = transform_numeric_categorical(df_test)

    df_test.to_csv('clean_data/' + output_name + '-test.csv', index=False)


if __name__ == "__main__":
    #clean(sys.argv[1])
    clean_transactions(db)
