import sys

from matplotlib.pyplot import cla
import joblib
import pandas as pd
from sklearn import metrics
from seaborn.axisgrid import Grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
import pickle
from imblearn.over_sampling import SMOTE

from pathlib import Path

DEBUG = False
RS = 42

def grid_search(classifier_name, submission_name):

    # Split dataset into training set and test set
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X, y = select_features(X, y)
    # X , y = select_features_RFECV(X,y,classifier_name)
    
    params = get_grid_params(classifier_name)

    classifier = get_classifier(classifier_name)

    grid_search_var = GridSearchCV(
        estimator=classifier,
        param_grid = params,
        scoring='roc_auc',
        cv=StratifiedKFold(5, random_state=RS, shuffle=True),
        n_jobs = -1)

    grid_results = grid_search_var.fit(X, y)


    print('Best Parameters: ', grid_results.best_params_)
    print('Best Score: ', grid_results.best_score_)
    

def select_features(X, y):
    models_folder = Path("models/")

    bestfeatures = SelectKBest(score_func=f_classif, k=12) # f_classif, f_regression

    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(12, 'Score'))  #print 10 best features

    print(featureScores.nlargest(12,'Score')['Specs'].values.tolist())

    best_attributes = featureScores.nlargest(12,'Score')['Specs'].values.tolist()

    pickle.dump(best_attributes, open(models_folder/'attributes.pkl', "wb"))

    X = X[best_attributes]

    return X, y


def select_features_RFECV(X,y,classifier_name):
    models_folder = Path("models/")
    classifier=get_classifier(classifier_name)
    
    bestfeatures = RFECV(classifier,scoring='roc_auc') 
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.ranking_ ) 
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

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Using SMOTE:
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    # SelectKBest Feature Selector
    X, y = select_features(X, y)

    # RFECV Feature Selector 
    #X,y = select_features_RFECV(X,y,classifier_name)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)

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
    if classifier == 'decision_tree':
        return DecisionTreeClassifier(random_state=RS)
    elif classifier == 'logistic_regression':
        return LogisticRegression(random_state=RS, max_iter=300)
    elif classifier == 'random_forest':
        return RandomForestClassifier(random_state=RS)
    elif classifier == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=RS)
    elif classifier == 'svm':
        return SVC(random_state=RS)


def get_grid_params(classifier):
    if classifier == 'decision_tree':
        return {'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [3,5,7],
                'min_samples_split': [1,2,3],
                'min_samples_leaf': [1,2,3],
                'min_weight_fraction_leaf': [0.0],
                'max_features': [None, 'auto', 'sqrt', 'log2', 12],
                'max_leaf_nodes': [None],
                'min_impurity_decrease': [0.0],
                'class_weight': [None],
                'ccp_alpha': [0.0]}
    elif classifier == 'logistic_regression':
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

    elif classifier == 'svm':
        return {'C': [1, 10], 
          'gamma': [0.001, 0.01, 1,'scale','auto'],
          'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
          'degree':[1,2,3,4,5,6,7],
          'coef0':[0.0, 0.1, 0.3, 0.5, 0.7],
          'max_iter':[1,2,3,5,7,10],
          'decision_function_shape':['ovo','ovr'],
          'class_weight':[None, 'balanced', dict]}

    #elif classifier == 'SelectKBest':
    #    return {'score_func': [ mutual_info_classif, chi2, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect, f_classif]}


def get_classifier_best(classifier):
    if classifier == 'decision_tree':
        # return DecisionTreeClassifier(criterion='entropy', splitter='random')
        return DecisionTreeClassifier()
    elif classifier == 'logistic_regression':
        # SelectKBest:
        return LogisticRegression(random_state=RS, C = 0.01, class_weight= 'balanced', penalty= 'none', solver= 'saga')
        # RFECV:
        # return LogisticRegression(C = 0.01, class_weight= None, penalty= 'l2', solver= 'newton-cg')
    elif classifier == 'random_forest':
        return RandomForestClassifier(random_state=RS, class_weight= 'balanced_subsample', criterion= 'gini', max_depth= 20, max_features= 'auto', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100)
        # Juliane:
        # return RandomForestClassifier(n_estimators=100, max_depth=30,class_weight= 'balanced') 
    elif classifier == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=RS, criterion='friedman_mse', learning_rate=0.7, loss= 'exponential', min_samples_leaf= 6, min_samples_split= 4, n_estimators= 12)
    elif classifier == 'svm':
        return SVC(random_state=RS, C= 1, class_weight= 'balanced', coef0= 0.0, decision_function_shape= 'ovo', degree= 5, gamma= 'scale', kernel= 'poly', max_iter= 3, probability=True)

def auc_scorer(y_true, y_pred):
    '''Scorer of Area Under Curve value'''
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

if __name__ == "__main__":
    if(DEBUG):
        grid_search(sys.argv[1], sys.argv[2])
    else:
        train(sys.argv[1], sys.argv[2])


    