import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

def train(classifier_name, submission_name):
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)

    x = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=10)

    grid_search = grid_search_model(classifier_name, X_train, y_train)

    print("Performance on the training set")
    y_train_pred = grid_search.predict(X_test)
    y_train_proba = grid_search.predict_proba(X_train)
    auc_train = roc_auc_score(y_train, y_train_proba[:, 1])
    print(f"Train ROC AUC: {auc_train}")
    #print(y_train_pred)

    print("\nPerformance on the test set")
    y_test_pred = grid_search.predict(X_test)
    y_test_proba = grid_search.predict_proba(X_test)
    auc_test = roc_auc_score(y_test, y_test_proba[:, 1])
    print(f"Test ROC AUC: {auc_test}")
    #print(y_test_pred)

    models_folder = Path("models/")
    filename = models_folder/(classifier_name + '-' + submission_name + '.sav')
    joblib.dump(grid_search, filename)

def grid_search_model(classifier_name, X_train, y_train):
    classifier = get_classifier(classifier_name)
    parameter_grid = get_grid_params(classifier_name)

    #cross_validation = KFold()
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    grid_search = GridSearchCV(
        classifier, 
        param_grid = parameter_grid, 
        cv = cv, 
        scoring='roc_auc'
    )

    grid_search.fit(X_train, y_train) #.values.ravel() na logistic_regression ??

    print('Best Parameters: ', grid_search.best_params_)

    return grid_search

def get_classifier(classifier):
    if classifier == 'logistic_regression':
        return LogisticRegression(max_iter=300)
    elif classifier == 'random_forest':
        return RandomForestClassifier(random_state=10)

def get_grid_params(classifier):
    if classifier == 'logistic_regression':
        return {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
    elif classifier == 'random_forest':
        return {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]}


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])

    