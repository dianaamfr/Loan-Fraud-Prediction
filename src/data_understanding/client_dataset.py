import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

sys.path.insert(1, '.')
from clean import split_birth
from database import database
db = database.Database('bank_database')

def client_du():
    df =  db.df_query('SELECT * FROM client')
    stats(df)
    client_distribution(df)
    client_decades_distribution(df)

def client_distribution(df):
    # Client Dataset
    sns.histplot(df['birth_number'])
    plt.savefig(get_distribution_folder('client')/'birth_number.jpg')
    plt.clf()
   
def client_decades_distribution(df):
    # Gender and Date of birth
    df['gender'], df['birth_date'] = zip(*df['birth_number'].map(split_birth))
    df['year'] = df['birth_date'].dt.year
    df.drop(columns=['birth_number', 'birth_date'], inplace=True)

    def get_decade(year):
        year_int = int(year)
        string = str(year_int//10) + "0-" + str(year_int//10) + "9"
        return string

    df['decade'] = df['year'].apply(lambda x: get_decade(x))

    df = df.sort_values('decade')

    # Birth Decades Bar Chart
    fig = plt.figure(figsize=(16, 7))
    plt.title("Birth decades of clients", fontsize=15)
    ax = sns.countplot(x="decade", data=df)
    ax.set_xlabel('birth decade', fontsize=15)
    ax.set_ylabel('number of clients', fontsize=15)
    plt.tight_layout()
    plt.savefig(get_distribution_folder('client')/'birth_decades.jpg')
    plt.close(fig)

if __name__ == '__main__':
    create_plots_folders('client')
    client_du()