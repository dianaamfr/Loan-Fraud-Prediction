from database import database
import pandas as pd

def clean_loans(db):
    df = db.df_query('SELECT * FROM loan_train')

    # Format loan date
    df['granted_date'] = df['granted_date'].apply(lambda date: int('19' + str(date)))

    # TODO - maybe change here the status because of the AUC computation

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
    
def split_birth(birth_number):
    birth_str = str(birth_number)
    
    # Male
    if int(birth_str[2:4]) < 50:
        return 'M', int('19' + birth_str)

    # Female
    year = '19' + birth_str[0:2]
    month = str(int(birth_str[2:4]) - 50).zfill(2)
    day = birth_str[4:]
    return 'F', int(year + month + day)

def clean_clients(db):
    df = db.df_query('SELECT * FROM client')

    # Gender and Date of birth
    df['gender'], df['birth_date'] = zip(*df['birth_number'].map(split_birth))

    # TODO - decide if we should drop the birth_number now that we have age and birth_date
    df.drop(columns=['birth_number'], inplace=True)

    return df

def clean_transactions(db):
    # TODO -check if this is needed:
    # Convert 'type'='withdrawal with cash' to 'withdrawal'
    #df.loc[df['type'] == 'withdrawal in cash','type'] = 'withdrawal'
    # TODO - check if the fact that a transaction is a credit or withdrawal should be considered

    # Get average amount, balance and number of transactions
    df = db.df_query('SELECT account_id, AVG(amount) AS avg_amount, AVG(balance) AS avg_balance, COUNT(trans_id) AS num_trans '\
        'FROM trans_train '\
        'GROUP BY account_id')

    return df

def clean_card(db):
    df = db.df_query("SELECT * FROM card_train")

    # Format issuance date
    df['issued'] = df['issued'].apply(lambda date: int('19' + str(date)))

    return df

def clean_disp(db):
    # Keep only the owner of the account
    df = db.df_query("SELECT * FROM disposition WHERE disp_type = 'OWNER'")

    return df

def clean_account(db):
    df = db.df_query("SELECT * FROM account")

    # Format creation date
    df['creation_date'] = df['creation_date'].apply(lambda date: int('19' + str(date)))
    
    return df


# For submission 2
def clean():
    db = database.Database('bank_database')
    loan = clean_loans(db)
    # print("LOANS:")
    # print(loans.head())

    district = clean_districts(db)
    # print("DISTRICTS:")
    # print(districts.head())

    client = clean_clients(db)
    # print("CLIENTS:")
    # print(clients.head())

    transaction = clean_transactions(db)
    # print("TRANSACTIONS:")
    # print(transactions.head())

    card = clean_card(db)
    # print("CARDS:")
    # print(transactions.head())

    disp = clean_disp(db)
    # print("DISPOSITION:")
    # print(disp.head())

    account = clean_account(db)
    # print("ACCOUNT:")
    # print(account.head())

    df = pd.merge(loan, account, how="inner",on="account_id")
    df = pd.merge(df, disp, how="inner",on="account_id")
    df = pd.merge(df, client, how="inner",on="client_id")
    df = pd.merge(df, transaction, how="inner",on="account_id")

    # TODO - add difference between creation and loan dates
    
    # TODO - merge card. Understand type of relation between type and dispostion

    # TODO - decide if we want to use both the district of the account and of the owner 
    # TODO - decide if we want to use the same_district attributes 
    # (which indicates if the owner and the account have the same district)
    # df = pd.merge(df, district, how="inner",on="client_id")

if __name__ == "__main__":
    clean()
