import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import datetime
from datetime import date
import sys

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

def clean_loans(db, test=False):
    table = 'loan_test' if test is True else 'loan_train'
    df = db.df_query('SELECT * FROM ' + table)

    # Format loan date
    df['granted_date'] = df['granted_date'].apply(lambda date: int('19' + str(date)))

    return df

def clean_districts(db):
    df = db.df_query('SELECT * FROM district')
    
    # Drop NaN values
    df["nr_commited_crimes_95"] = pd.to_numeric(df["nr_commited_crimes_95"], errors='coerce')
    df["unemployment_rate_95"] = pd.to_numeric(df["unemployment_rate_95"], errors='coerce')
    df.dropna(inplace=True) # Remove NaN values

    # Average Crimes
    df["avg_crimes"] = (df["nr_commited_crimes_95"] + df["nr_commited_crimes_96"]) / 2.0

    # Average Unemployment
    df["avg_unemployment"] = (df["unemployment_rate_95"] + df["unemployment_rate_96"]) / 2.0

    # TODO - decide if we should drop the non aggregated columns (unemployment and crimes 95,96)
    # df.drop(columns=['nr_commited_crimes_95'], inplace=True)
    # df.drop(columns=['nr_commited_crimes_96'], inplace=True)
    # df.drop(columns=['unemployment_rate_95'], inplace=True)
    # df.drop(columns=['unemployment_rate_96'], inplace=True)

    # TODO - decide if the region is relevant, given that the id already specifies different regions

    return df

def calculate_age(birth_date):
    birth_date = datetime.datetime.strptime(birth_date, "%Y%m%d")
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def split_birth(birth_number):
    birth_str = str(birth_number)
    
    # Male
    if int(birth_str[2:4]) < 50:
        return 'M', calculate_age('19' + birth_str)

    # Female
    year = '19' + birth_str[0:2]
    month = str(int(birth_str[2:4]) - 50).zfill(2)
    day = birth_str[4:]
    return 'F', calculate_age(year + month + day)

def clean_clients(db):
    df = db.df_query('SELECT * FROM client')

    # Gender and Date of birth
    df['gender'], df['age'] = zip(*df['birth_number'].map(split_birth))

    # TODO - decide if we should drop the birth_number now that we have age and birth_date
    df.drop(columns=['birth_number'], inplace=True)

    return df

def clean_transactions(db, test=False):
    table = 'trans_test' if test is True else 'trans_train'
    # TODO -check if this is needed:
    # Convert 'type'='withdrawal with cash' to 'withdrawal'
    #df.loc[df['type'] == 'withdrawal in cash','type'] = 'withdrawal'
    # TODO - check if the fact that a transaction is a credit or withdrawal should be considered

    # Get average amount, balance and number of transactions
    df = db.df_query('SELECT account_id, AVG(amount) AS avg_amount, AVG(balance) AS avg_balance, COUNT(trans_id) AS num_trans '\
        'FROM ' + table + ' GROUP BY account_id')

    return df

def clean_card(db, test=False):
    table = 'card_test' if test is True else 'card_train'
    df = db.df_query("SELECT * FROM " + table)

    # Format issuance date
    df['issued'] = df['issued'].apply(lambda date: int('19' + str(date)))

    return df

def clean_disp(db):
    # Keep only the owner of the account
    df = db.df_query("SELECT * FROM disposition WHERE disp_type = 'OWNER'")
    df.drop(columns=['disp_type'], inplace=True)

    return df

def clean_account(db):
    df = db.df_query("SELECT * FROM account")

    # Format creation date
    df['creation_date'] = df['creation_date'].apply(lambda date: int('19' + str(date)))
    
    return df


def transform_status(df):
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    df["loan_status"].replace({1: 0}, inplace=True)
    df["loan_status"].replace({-1: 1}, inplace=True)

def transform_numeric_categorical(df):
    num_cols = ['amount', 'duration', 'age', 'payments','avg_amount','avg_balance','num_trans','days_between']
    cat_cols = ['frequency', 'gender']

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df = pd.get_dummies(df, prefix=cat_cols, columns = cat_cols, drop_first=True)

    return df

# Submission 1 - only loan
def clean1(output_name):
    db = database.Database('bank_database')

    loan_train = db.df_query('SELECT granted_date, amount, duration, payments, loan_status FROM loan_train')
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    #transform_status(loan_train)
    loan_train.to_csv('clean_data/' + output_name + '-train.csv', index=False)

    loan_test = db.df_query('SELECT loan_id, granted_date, amount, duration, payments FROM loan_test')
    loan_test.to_csv('clean_data/' + output_name + '-test.csv', index=False)

# For submission 2
def clean2(output_name):
    db = database.Database('bank_database')

    ############
    # TRAIN DATA
    ############
    from functools import reduce
    loan_train = clean_loans(db)
    district = clean_districts(db)
    client = clean_clients(db)
    transaction_train = clean_transactions(db)
    card_train = clean_card(db)
    disp = clean_disp(db)
    account = clean_account(db)

    df = pd.merge(loan_train, account, on='account_id', how="left")
    df = pd.merge(df, disp,  on='account_id', how="left")
    df = pd.merge(df, client,  on='client_id', how="left")
    df = pd.merge(df, transaction_train, how="left", on="account_id")

    # Days between loan and account creation
    df['days_between'] = df['granted_date'] - df['creation_date']

    # TODO - decide if we should drop the dates
    df.drop(columns=['creation_date', 'granted_date'], inplace=True)
    
    # TODO - train data
    # - merge CARD. 
    # - Understand type of relation between type and dispostion
    # - decide if we want to use both the district info of the account and of the owner 
    # - decide if we want to use the same_district attribute(indicates if the owner and account have the same district)
    # - merge DISTRICT

    # Drop Irrelevant IDs
    # TODO - drop district attributes and card if needed 
    df.drop(columns=["account_id", "loan_id", "disp_id", "client_id", "district_id_x", "district_id_y"], inplace=True)

    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    transform_status(df)

    # Scale numeric variables and encode categorical variables
    df = transform_numeric_categorical(df)

    df.to_csv('clean_data/' + output_name + '-train.csv', index=False)

    ############
    # TEST DATA
    ############

    # TODO - clean test data and save to output file
    loan_test = clean_loans(db, True)
    transaction_test = clean_transactions(db, True)
    card_test = clean_card(db, True)    

    df_test = pd.merge(loan_test, account, how="left",on="account_id")
    df_test = pd.merge(df_test, disp, how="left",on="account_id")
    df_test = pd.merge(df_test, client, how="left",on="client_id")
    df_test = pd.merge(df_test, transaction_test, how="left",on="account_id")

    # Days between loan and account creation
    df_test['days_between'] = df_test['granted_date'] - df_test['creation_date']

    # TODO - decide if we should drop the dates
    df_test.drop(columns=['creation_date', 'granted_date'], inplace=True)

    # TODO - test data
    # - merge CARD. 
    # - Understand type of relation between type and dispostion
    # - decide if we want to use both the district info of the account and of the owner 
    # - decide if we want to use the same_district attribute(indicates if the owner and account have the same district)
    # - merge DISTRICT

    # Drop Irrelevant IDs
    # TODO - drop district attributes and card if needed 

    df_test.drop(columns=["account_id", "disp_id", "client_id", "district_id_x", "district_id_y", "loan_status"], inplace=True)

    # Scale numeric variables and encode categorical variables
    df_test = transform_numeric_categorical(df_test)

    df_test.to_csv('clean_data/' + output_name + '-test.csv', index=False)

if __name__ == "__main__":
    #clean1(sys.argv[1])
    clean2(sys.argv[1])
