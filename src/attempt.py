import database

db = database.Database('bank_database')
df = db.df_query('SELECT * FROM account;')

print(df)