import sys
import joblib
import pandas as pd
from sklearn import metrics
from seaborn.axisgrid import Grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path

DEBUG = False
RS = 42

def grid_search(classifier_name, submission_name):

    # Split dataset into training set and test set
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)
    df = normalize_if_not_tree_based(df, classifier_name)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    params = get_grid_params(classifier_name)
    classifier = get_classifier(classifier_name)
    grid_search_var = GridSearchCV(
        estimator=classifier,
        param_grid = params,
        scoring='roc_auc',
        cv=KFold(5, random_state=RS, shuffle=True),
        n_jobs = -1)

    grid_results = grid_search_var.fit(X, y)

    print('Best Parameters: ', grid_results.best_params_)
    print('Best Score: ', grid_results.best_score_)
    

def filter_feature_selection(X, y):
    models_folder = Path("models/")

    bestfeatures = SelectKBest(score_func=f_classif, k=10) # f_classif, f_regression
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))  #print 10 best features

    print(featureScores.nlargest(10,'Score')['Specs'].values.tolist())

    best_attributes = featureScores.nlargest(10,'Score')['Specs'].values.tolist()

    pickle.dump(best_attributes, open(models_folder/'attributes.pkl', "wb"))

    X = X[best_attributes]

    return X, y

def train(classifier_name, submission_name):
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)
    df = normalize_if_not_tree_based(df, classifier_name)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    print(X.columns)

    X, y = filter_feature_selection(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)

    classifier = get_classifier_best(classifier_name)
    classifier.fit(X_train, y_train)

    print("Performance on the training set")
    y_train_pred = classifier.predict(X_test)
    y_train_proba = classifier.predict_proba(X_train)
    auc_train = roc_auc_score(y_train, y_train_proba[:, 1])
    print(f"Train ROC AUC: {auc_train}")
    #print(y_train_pred)

    print("\nPerformance on the test set")
    y_test_pred = classifier.predict(X_test)
    y_test_proba = classifier.predict_proba(X_test)
    auc_test = roc_auc_score(y_test, y_test_proba[:, 1])
    print(f"Test ROC AUC: {auc_test}")
    #print(y_test_pred)

    models_folder = Path("models/")
    filename = models_folder/(classifier_name + '-' + submission_name + '.sav')
    joblib.dump(classifier, filename)


def get_classifier(classifier):
    if classifier == 'logistic_regression':
        return LogisticRegression(random_state=RS, max_iter=300)
    elif classifier == 'random_forest':
        return RandomForestClassifier(random_state=RS)
    elif classifier == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=RS)

def get_grid_params(classifier):
    if classifier == 'logistic_regression':
        return {'penalty': ['l2', 'none'],
                'C': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                'class_weight': ["balanced", None]}

    elif classifier == 'random_forest':
        max_depth = [int(x) for x in range(2, 16, 4)]
        max_depth.append(None)

        return {'n_estimators': [int(x) for x in range(2, 14, 2)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': max_depth,
                'criterion': ['gini', 'entropy'],
                'min_samples_split':  [2, 4, 6, 8],
                'min_samples_leaf':  [1, 2, 4, 6],
                'class_weight': ["balanced", "balanced_subsample", None]}
    
    elif classifier == 'gradient_boosting':
        return {'n_estimators': [int(x) for x in range(2, 14, 2)],
            'learning_rate': [0.1, 0.3, 0.5, 0.7],
            'loss': ['deviance', 'exponential'],
            'criterion': ['friedman_mse', 'squared_error'],
            'min_samples_split':  [4, 6, 8],
            'min_samples_leaf':  [2, 4, 6]}


def get_classifier_best(classifier):
    if classifier == 'logistic_regression':
        return LogisticRegression(C = 0.01, class_weight= 'balanced', penalty= 'none', solver= 'saga')
    elif classifier == 'random_forest':
        return RandomForestClassifier(class_weight= 'balanced_subsample', criterion= 'entropy', max_depth= 10, max_features= 'auto', min_samples_leaf= 6, min_samples_split= 2, n_estimators= 6)
    elif classifier == 'gradient_boosting':
        return GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.7, loss= 'exponential', min_samples_leaf= 2, min_samples_split= 8, n_estimators= 8)


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
    if(DEBUG):
        grid_search(sys.argv[1], sys.argv[2])
    else:
        train(sys.argv[1], sys.argv[2])


    