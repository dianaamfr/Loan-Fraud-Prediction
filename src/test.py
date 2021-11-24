import joblib
from database import database
import sys
import pandas as pd
from pathlib import Path

def test(classifier, data_file , output_name):
    df = pd.read_csv('clean_data/' + data_file + '-test.csv', delimiter=",", low_memory=False)

    models_folder = Path("models/")
    filename = models_folder/(classifier + '-' + output_name + '.sav')
    model = joblib.load(filename)

    x_test = df.drop(columns=['loan_id'])
    print(x_test.head())
    prediction = model.predict_proba(x_test)[::,1]

    print('TEST PREDICTION:')
    print(prediction)

    # Create the pandas DataFrame
    df_result = pd.DataFrame()

    loan_ids = df['loan_id']

    df_result['Id'] = loan_ids
    df_result['Predicted'] = prediction

    df_result.to_csv('results/' + classifier + '-' + output_name + '.csv', sep=',', index=False)

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], sys.argv[3])