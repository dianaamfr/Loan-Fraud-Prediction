import joblib
import sys
import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler

def test(classifier_name, submission_name):
    df = pd.read_csv('clean_data/' + submission_name + '-test.csv', delimiter=",", low_memory=False)

    best_attributes = []
    try:
        output_file = open("./models/attributes.pkl", 'rb')
        best_attributes = pickle.load(output_file)
        output_file.close()
    except Exception:            
        print("Error loading best features list")
        sys.exit(1)
    print(best_attributes)

    df = df[['loan_id'] + best_attributes]
    x_test = df.set_index('loan_id')
    x_test = normalize_if_not_tree_based(x_test, classifier_name)
    print(x_test.head())

    models_folder = Path("models/")
    filename = models_folder/(classifier_name + '-' + submission_name + '.sav')
    model = joblib.load(filename)

    prediction = model.predict_proba(x_test)[::,1]

    # Create the pandas DataFrame
    df_result = pd.DataFrame()

    loan_ids = df['loan_id']

    df_result['Id'] = loan_ids
    df_result['Predicted'] = prediction

    df_result.to_csv('results/' + classifier_name + '-' +  submission_name + '.csv', sep=',', index=False)



###########
# Normalize
###########

def normalize_if_not_tree_based(df, classifier_name):
    if (classifier_name != 'decision_tree' and classifier_name != 'random_forest'):
        return normalize(df)
    return df

def normalize(df):
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(df)
    df = pd.DataFrame(transformed, index=df.index, columns=df.columns)
    return df


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])