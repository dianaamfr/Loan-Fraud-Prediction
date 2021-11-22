import pandas as pd
import re

def split_birth(birth_number):
    birth_str = str(birth_number)
    if int(birth_str[2:4]) < 50:
        return 'M', birth_number
    return 'F', int(birth_str[0:2] + str(int(birth_str[2:4]) - 50) + birth_str[4:5])
    

# Em desenvolvilmento
def prepare_clients(clients_csv):
    clients_df = pd.read_csv(clients_csv, delimiter=";", low_memory=False)

    # Get 'gender' and 'birth_date' from birth_number
    clients_df['gender'], clients_df['birth_date'] = zip(*clients_df['birth_number'].map(split_birth))

    return clients_df

def prepare_transactions(trans_csv):
    trans_df = pd.read_csv(trans_csv, delimiter=";", low_memory=False)

    # Convert 'type'='withdrawal with cash' to 'withdrawal'
    trans_df.loc[trans_df['type'] == 'withdrawal in cash','type'] = 'withdrawal'

    return trans_df

clients = prepare_clients('../ficheiros_competicao/client.csv')
print(prepare_clients('../ficheiros_competicao/client.csv'))
print(clients.nunique())
#print(prepare_transactions('../ficheiros_competicao/trans_train.csv'))
