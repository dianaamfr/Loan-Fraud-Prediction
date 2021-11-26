import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import datetime
from datetime import date
import sys

sys.path.insert(1, '.')
from database import database
db = database.Database('bank_database')

#pd.set_option('display.max_columns', None)

#######
# Clean
#######

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
    df["avg_crimes"] = df[['nr_commited_crimes_95', 'nr_commited_crimes_96']].mean(axis=1)
    df.drop(columns=['nr_commited_crimes_95'], inplace=True)
    df.drop(columns=['nr_commited_crimes_96'], inplace=True)

    # Average Unemployment
    df["avg_unemployment"] = df[['unemployment_rate_95', 'unemployment_rate_96']].mean(axis=1)
    df.drop(columns=['unemployment_rate_95'], inplace=True)
    df.drop(columns=['unemployment_rate_96'], inplace=True)

    df = df[['district_id', 'avg_crimes', 'avg_unemployment','nr_inhabitants', 
    'average_salary', 'nr_enterpreneurs_1000_inhabitants']]

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

    df.rename(columns={'district_id': 'client_district_id'}, inplace=True)

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
    
    df.rename(columns={'district_id': 'account_district_id'}, inplace=True)

    return df


############
# Transform
############

def transform_status(df):
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    df["loan_status"].replace({1: 0}, inplace=True)
    df["loan_status"].replace({-1: 1}, inplace=True)

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
   

    # df.frequency = df.frequency.astype('category')
    # df.frequency = df.frequency.astype('category') # TODO: ver vídeo do YT da conversão das categorical
    # df.frequency = df.frequency.astype('category')

    # TODO - try to change to OneHotEncoder because this add two columns instead of one
    #df = pd.get_dummies(df, prefix=cat_cols, columns = cat_cols, drop_first=True)

    return df


#############
# Submissions
#############

# Submission 1 - only loan
def clean1(output_name):

    loan_train = db.df_query('SELECT granted_date, amount, duration, payments, loan_status FROM loan_train')
    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    #transform_status(loan_train)
    loan_train.to_csv('clean_data/' + output_name + '-train.csv', index=False)

    loan_test = db.df_query('SELECT loan_id, granted_date, amount, duration, payments FROM loan_test')
    loan_test.to_csv('clean_data/' + output_name + '-test.csv', index=False)

# For submission 2
def clean2(output_name):

    ############
    # TRAIN DATA
    ############
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
    # TODO - fazer 2 gráficos para mostrar pq que escolhemos os dados do district relativos à account e não ao client
    df = pd.merge(df, district, left_on='account_district_id', right_on='district_id') # associate account's district
    df = pd.merge(df, transaction_train, how="left", on="account_id")
    # TODO - merge CARD.

    # Days between loan and account creation
    df['days_between'] = df['granted_date'] - df['creation_date']
    df.drop(columns=['creation_date', 'granted_date'], inplace=True)

    # Boolean value telling if the account was created on the owner district
    df['same_district'] = df['account_district_id'] == df['client_district_id']

    # Drop Irrelevant IDs
    # TODO - drop card_id
    df.drop(columns=["account_id", "loan_id", "disp_id", "client_id", 
    "district_id", "account_district_id", "client_district_id"], inplace=True)

    # Transform Status - 1 => 0 (loan granted) and -1 => 1 (loan not granted - aim of the analysis)
    # transform_status(df)

    # Scale numeric variables and encode categorical variables
    df = transform_numeric_categorical(df)

    df.to_csv('clean_data/' + output_name + '-train.csv', index=False)


    ############
    # TEST DATA
    ############

    loan_test = clean_loans(db, True)
    transaction_test = clean_transactions(db, True)
    card_test = clean_card(db, True)    

    df_test = pd.merge(loan_test, account, how="left",on="account_id")
    df_test = pd.merge(df_test, disp, how="left",on="account_id")
    df_test = pd.merge(df_test, client, how="left",on="client_id")
    df_test = pd.merge(df_test, district, left_on='account_district_id', right_on='district_id') # associate account's district
    df_test = pd.merge(df_test, transaction_test, how="left",on="account_id")
    # TODO - merge CARD. 

    # Days between loan and account creation
    df_test['days_between'] = df_test['granted_date'] - df_test['creation_date']
    df_test.drop(columns=['creation_date', 'granted_date'], inplace=True)

    # Boolean value telling if the account was created on the owner district
    df_test['same_district'] = df_test['account_district_id'] == df_test['client_district_id'] 

    # Drop Irrelevant IDs
    # TODO - drop card_id
    df_test.drop(columns=["account_id", "disp_id", "client_id", 
    "district_id", "account_district_id", "client_district_id", "loan_status"], inplace=True)

    # Scale numeric variables and encode categorical variables
    df_test = transform_numeric_categorical(df_test)

    df_test.to_csv('clean_data/' + output_name + '-test.csv', index=False)

if __name__ == "__main__":
    #clean1(sys.argv[1])
    clean2(sys.argv[1])
