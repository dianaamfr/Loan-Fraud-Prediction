import joblib
import sys
import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
from utils import *

def test(classifier_name, submission_name):
    df = pd.read_csv('clean_data/' + submission_name + '-test.csv', delimiter=",", low_memory=False)

    # Select best features
    best_features = get_best_features()
    df = df[['loan_id'] + best_features]
    x_test = df.set_index('loan_id')

    # Scaling
    scaler = MinMaxScaler()
    x_test = normalize_if_not_tree_based(x_test, classifier_name, scaler)
    print(x_test.head())

    # Get trained model
    models_folder = Path("models/")
    filename = models_folder/(classifier_name + '-' + submission_name + '.sav')
    model = joblib.load(filename)

    # Predict
    prediction = model.predict_proba(x_test)[::,1]

    # Store results
    df_result = pd.DataFrame()
    loan_ids = df['loan_id']
    df_result['Id'] = loan_ids
    df_result['Predicted'] = prediction

    df_result.to_csv('results/' + classifier_name + '-' +  submission_name + '.csv', sep=',', index=False)


def get_best_features():
    best_features = []
    try:
        output_file = open("./models/features.pkl", 'rb')
        best_features = pickle.load(output_file)
        output_file.close()
    except Exception:            
        print("Error loading best features list")
        sys.exit(1)
    print(best_features)
    return best_features

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])