import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)

trans_train_csv = pd.read_csv('../ficheiros_competicao/trans_train.csv', delimiter=";", low_memory=False)

print('Transaction:\n ', trans_train_csv, '\n')
print('Types for NaN operations: ', trans_train_csv[trans_train_csv['operation'].isnull()]['type'].unique(), '\n')
print('Unique Types: ', trans_train_csv['type'].unique(), '\n')
print('Unique Operations: ', trans_train_csv['operation'].unique(), '\n')
print('Unique K-Symbols: ', trans_train_csv['k_symbol'].unique(), '\n')
print('K-Symbols, Types and Operations:\n ', trans_train_csv[['operation', 'k_symbol', 'type']].groupby(['operation', 'k_symbol', 'type'], dropna=False).size().reset_index().rename(columns={0:'count'}))

print('\n\n')
print('Widthdrawal exploration:\n')
print(trans_train_csv.loc[(trans_train_csv['type'] == 'withdrawal') | (trans_train_csv['type'] == 'withdrawal in cash'), 'amount'].min())

