import joblib
import database
import sys
import pandas as pd
from pathlib import Path

def test(args):
    db = database.Database('bank_database')
    feature_cols = ['granted_date', 'amount', 'duration', 'payments']

    models_folder = Path("models/")
    filename = models_folder/(args[0] + '.sav')
    model = joblib.load(filename)

    x_query_test = 'SELECT {0} FROM loan_test;'.format(','.join(feature_cols))
    x_test = db.df_query(x_query_test)
    prediction = model.predict_proba(x_test)[::,1]

    print('TEST PREDICTION:')
    print(prediction)

    # Create the pandas DataFrame
    df_result = pd.DataFrame()

    loan_id_query = 'SELECT loan_id FROM loan_test;'
    loan_ids = db.df_query(loan_id_query)

    df_result['Id'] = loan_ids
    df_result['Predicted'] = prediction

    df_result.to_csv('results/results.csv', sep=',', index=False)

if __name__ == "__main__":
    test(sys.argv[1:])