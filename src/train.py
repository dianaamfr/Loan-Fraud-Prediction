import database
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from pathlib import Path

def train(args):
    db = database.Database('bank_database')
    feature_cols = ['granted_date', 'amount', 'duration', 'payments'] # TODO: Futuramente adicionar novas colunas

    x_query = 'SELECT {0} FROM loan_train;'.format(','.join(feature_cols))
    y_query = 'SELECT loan_status FROM loan_train;'

    x = db.df_query(x_query)
    y = db.df_query(y_query)

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    classifier = get_classifier(args[0])

    cross_validation = KFold()

    parameter_grid = get_parameter_grid(args[0])

    grid_search = GridSearchCV(
        classifier, 
        param_grid = parameter_grid, 
        cv = cross_validation, 
        scoring='roc_auc'
    )

    grid_search.fit(x_train, y_train.values.ravel())

    y_pred = grid_search.predict_proba(x_test)
    prob_loan = y_pred[::, 1]
    auc = roc_auc_score(y_test, prob_loan)

    print('Train Prediction:')
    print(y_pred)
    print(f"AUC: {auc}")

    models_folder = Path("models/")
    filename = models_folder/(args[0] + '.sav')
    joblib.dump(grid_search, filename)

def get_classifier(classifier):
    if classifier == 'logistic_regression':
        return LogisticRegression()

def get_parameter_grid(classifier):
    if classifier == 'logistic_regression':
        return {"C":np.logspace(-3,3,7), "penalty":["l2"]}

if __name__ == "__main__":
    train(sys.argv[1:])