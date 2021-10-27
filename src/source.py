import pandas as pd

account_csv = pd.read_csv('../ficheiros_competicao/account.csv', delimiter=";")
card_test_csv = pd.read_csv('../ficheiros_competicao/card_test.csv', delimiter=";")
card_train_csv = pd.read_csv('../ficheiros_competicao/card_train.csv', delimiter=";")
client_csv = pd.read_csv('../ficheiros_competicao/client.csv', delimiter=";")
disp_csv = pd.read_csv('../ficheiros_competicao/disp.csv', delimiter=";")
district_csv = pd.read_csv('../ficheiros_competicao/district.csv', delimiter=";")
loan_test_csv = pd.read_csv('../ficheiros_competicao/loan_test.csv', delimiter=";")
loan_train_csv = pd.read_csv('../ficheiros_competicao/loan_train.csv', delimiter=";")
trans_test_csv = pd.read_csv('../ficheiros_competicao/trans_test.csv', delimiter=";")
trans_train_csv = pd.read_csv('../ficheiros_competicao/trans_train.csv', delimiter=";")

#print(account_csv.info())

'''
Example - join csv by a specific column
# Multiple clients can operate the same account;
# and the same client may have more than one account,
# which means there will be columns with the same account_id or client_id if we merge this two "tables"
'''
# account_disp = account_csv.merge(disp_csv, on='account_id', how='inner')
#   print(account_disp.info())
