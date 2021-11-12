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
    print("Null Values:")
    print(df.isnull().sum())
    print()
    print("Unique Values:")
    print(df.nunique())

def get_files_folder():
    return Path("../../ficheiros_competicao/")
    