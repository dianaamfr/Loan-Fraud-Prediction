import pandas as pd
import re

# Em desenvolvilmento
def prepare_clients(clients_csv):
    clients_df = pd.read_csv(clients_csv, delimiter=";", low_memory=False)

    # Get 'gender' and 'birth_date' from birth_number
    # clients_df['gender', 'birth_date'] = clients_df.apply([('M', clients_df['birth_number']) if re.match(clients_df['birth_number'], '\d{6}') else ('F', ''.join(clients_df['birth_number'].split('+50')))]
    return clients_df

def prepare_transactions(trans_csv):
    trans_df = pd.read_csv(trans_csv, delimiter=";", low_memory=False)

    # Convert 'type'='withdrawal with cash' to 'withdrawal'
    trans_df.loc[trans_df['type'] == 'withdrawal in cash','type'] = 'withdrawal'

    # Transform withdrawal into negative values
    trans_df.loc[trans_df['type'] == 'withdrawal', 'amount'] *= -1

    # Drop irrelevant parter information
    trans_df = trans_df.drop(['bank', 'account'], axis=1)

    return trans_df

print(prepare_transactions('../ficheiros_competicao/trans_train.csv'))