import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)

account_csv = pd.read_csv('../ficheiros_competicao/account.csv', delimiter=";", low_memory=False)
card_test_csv = pd.read_csv('../ficheiros_competicao/card_test.csv', delimiter=";", low_memory=False)
card_train_csv = pd.read_csv('../ficheiros_competicao/card_train.csv', delimiter=";", low_memory=False)
client_csv = pd.read_csv('../ficheiros_competicao/client.csv', delimiter=";", low_memory=False)
disp_csv = pd.read_csv('../ficheiros_competicao/disp.csv', delimiter=";", low_memory=False)
district_csv = pd.read_csv('../ficheiros_competicao/district.csv', delimiter=";" , low_memory=False)
loan_test_csv = pd.read_csv('../ficheiros_competicao/loan_test.csv', delimiter=";", low_memory=False)
loan_train_csv = pd.read_csv('../ficheiros_competicao/loan_train.csv', delimiter=";", low_memory=False)
trans_test_csv = pd.read_csv('../ficheiros_competicao/trans_test.csv', delimiter=";", low_memory=False)
trans_train_csv = pd.read_csv('../ficheiros_competicao/trans_train.csv', delimiter=";", low_memory=False)


'''
Example - join csv by a specific column
# Multiple clients can operate the same account;
# and the same client may have more than one account,
# which means there will be columns with the same account_id or client_id if we merge this two "tables"
'''
# account_disp = account_csv.merge(disp_csv, on='account_id', how='inner')
#   print(account_disp.info())
