from pathlib import Path

def stats(df):
    print(df.head())
    print()
    print("Statistical Description:")
    print(df.describe())
    print()
    print("Data Types:")
    print(df.info())
    print()
    print("NaN Values:")
    print(df.isna().sum())
    print()
    print("Unique Values:")
    print(df.nunique())

def get_files_folder():
    return Path("../ficheiros_competicao/")

def create_plots_folders(table):
    get_distribution_folder('card').mkdir(parents=True, exist_ok=True)
    get_correlation_folder('card').mkdir(parents=True, exist_ok=True)

def get_correlation_folder(table):
    return Path('data_understanding/plots/correlation/')/table

def get_distribution_folder(table):
    return Path('data_understanding/plots/distribution/')/table