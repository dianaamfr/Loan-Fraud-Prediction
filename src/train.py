import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
from sklearn import metrics
import seaborn as sb
from seaborn.axisgrid import Grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from pathlib import Path
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn import preprocessing

DEBUG = False
RS = 42

################################################
# Cross-Validation comparison of the classifiers
################################################

def no_cross_validation(X, y, classifier, kf):
    status_values = list(y.unique())
    status_values = [-1 if x == 1 else 1 for x in status_values]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)

    classifier.fit(X_train, y_train)

    print("Performance on the training set")
    y_train_pred = classifier.predict(X_train)
    y_train_proba = classifier.predict_proba(X_train)
    auc_train = roc_auc_score(y_train, y_train_proba[:, 1])
    print(f"Train ROC AUC: {auc_train}")
    #print(y_train_pred)

    print("\nPerformance on the test set")
    y_test_pred = classifier.predict(X_test)
    y_test_proba = classifier.predict_proba(X_test)
    auc_test = roc_auc_score(y_test, y_test_proba[:, 1])
    print(f"Test ROC AUC: {auc_test}")

    conf_matrix = confusion_matrix(y_test, y_test_pred)

    print('Confusion matrix:')
    print(conf_matrix)

    confusion_matrix_fig = plt.figure(figsize = (15,8))
    sb.heatmap(conf_matrix, xticklabels=status_values, yticklabels=status_values, annot=True)
    plt.title('Confusion matrix')

    confusion_matrix_fig.tight_layout()
    plt.savefig('models/confusion_matrix.jpg')
    plt.clf()


def cross_validation(X, y, classifier, kf, num_splits):
    status_values = list(y.unique())
    status_values = [-1 if x == 1 else 1 for x in status_values]
    
    auc_train_scores = []
    auc_test_scores = []
 
    confusion_matrix_fig = plt.figure(figsize = (15,8))

    plot_index = num_splits + 1

    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        classifier.fit(X_train,y_train)

        y_pred = classifier.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f'Confusion matrix of split {plot_index-num_splits}:')
        print(conf_matrix)

        confusion_matrix_fig.add_subplot(3, num_splits, plot_index)
        sb.heatmap(conf_matrix, xticklabels=status_values, yticklabels=status_values, annot=True)
        plt.title(f'Split {plot_index-num_splits}')

        plot_index += 1

        pred_proba_test = classifier.predict_proba(X_test)[:, 1]
        pred_proba_train = classifier.predict_proba(X_train)[:, 1]
        
        auc_test = roc_auc_score(y_test, pred_proba_test)
        auc_train = roc_auc_score(y_train, pred_proba_train)

        auc_test_scores.append(auc_test)
        auc_train_scores.append(auc_train)

    confusion_matrix_fig.tight_layout()
    plt.savefig('models/confusion_matrix.jpg')
    plt.clf()
  
    print('AUC Train scores of each fold - {}'.format(auc_train_scores))
    print('AUC Test scores of each fold - {}'.format(auc_test_scores))

def compare_classifiers(kfold, X, y, X_normalized, y_normalized):
    decision_tree_classifier = get_classifier_best('decision_tree')
    logistic_regression_classifier = get_classifier_best('logistic_regression')
    random_forest_classifier = get_classifier_best('random_forest')
    gradient_boosting_classifier = get_classifier_best('gradient_boosting')
    svm_classifier = get_classifier_best('svm')
    knn_classifier = get_classifier_best('knn')
    neural_network_classifier = get_classifier_best('neural_network')
    xgboost_classifier = get_classifier_best('xgboost')
    bagging_classifier = get_classifier_best('bagging')

    decision_tree_scores = cross_val_score(decision_tree_classifier, X, y, cv=kfold)
    logistic_regression_scores = cross_val_score(logistic_regression_classifier, X_normalized, y_normalized, cv=kfold)
    random_forest_scores = cross_val_score(random_forest_classifier, X, y, cv=kfold)
    gradient_boosting_scores = cross_val_score(gradient_boosting_classifier, X, y, cv=kfold)
    svm_scores = cross_val_score(svm_classifier, X_normalized, y_normalized, cv=kfold)
    knn_scores = cross_val_score(knn_classifier, X_normalized, y_normalized, cv=kfold)
    neural_network_scores = cross_val_score(neural_network_classifier, X_normalized, y_normalized, cv=kfold)
    xgboost_scores = cross_val_score(xgboost_classifier, X, y, cv=kfold)
    bagging_scores = cross_val_score(bagging_classifier, X, y, cv=kfold)

    print("DECISION TREE: %0.2f accuracy with a standard deviation of %0.2f" % (decision_tree_scores.mean(), decision_tree_scores.std()))
    print("LOGISTIC REGRESSION: %0.2f accuracy with a standard deviation of %0.2f" % (logistic_regression_scores.mean(), logistic_regression_scores.std()))
    print("RANDOM FOREST: %0.2f accuracy with a standard deviation of %0.2f" % (random_forest_scores.mean(), random_forest_scores.std()))
    print("GRADIENT BOOSTING: %0.2f accuracy with a standard deviation of %0.2f" % (gradient_boosting_scores.mean(), gradient_boosting_scores.std()))
    print("SVM: %0.2f accuracy with a standard deviation of %0.2f" % (svm_scores.mean(), svm_scores.std()))
    print("KNN: %0.2f accuracy with a standard deviation of %0.2f" % (knn_scores.mean(), knn_scores.std()))
    print("NEURAL NETWORK: %0.2f accuracy with a standard deviation of %0.2f" % (neural_network_scores.mean(), neural_network_scores.std()))
    print("XGBOOST: %0.2f accuracy with a standard deviation of %0.2f" % (xgboost_scores.mean(), xgboost_scores.std()))
    print("BAGGING: %0.2f accuracy with a standard deviation of %0.2f" % (bagging_scores.mean(), bagging_scores.std()))


def train(classifier_name, submission_name):
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)

    scaler = MinMaxScaler()
    df = normalize_if_not_tree_based(df, classifier_name, scaler)
    normalized_df = normalize(df, scaler)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_normalized = normalized_df.drop(columns=['loan_status'])
    y_normalized = normalized_df['loan_status']

    oversample = SMOTE(random_state=RS)
    X, y = oversample.fit_resample(X, y)
    X_normalized, y_normalized = oversample.fit_resample(X_normalized, y_normalized)

    X, y = filter_feature_selection(X, y)
    X_normalized, y_normalized = filter_feature_selection(X_normalized, y_normalized)

    classifier = get_classifier_best(classifier_name)
    
    num_splits = 5
    kf = KFold(num_splits, random_state=RS, shuffle=True)

    #compare_classifiers(kf, X, y, X_normalized, y_normalized)

    cross_validation(X, y, classifier, kf, num_splits)

    #no_cross_validation(X, y, classifier, kf)

    models_folder = Path("models/")
    filename = models_folder/(classifier_name + '-' + submission_name + '.sav')
    joblib.dump(classifier, filename)

# smote for grid search https://stackoverflow.com/questions/50245684/using-smote-with-gridsearchcv-in-scikit-learn
def grid_search(classifier_name, submission_name):

    # Split dataset into training set and test set
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)

    scaler = MinMaxScaler()
    df = normalize_if_not_tree_based(df, classifier_name, scaler)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X, y = filter_feature_selection(X, y)

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


################################
# Filter Based Feature Selection
################################  

def filter_feature_selection(X, y, k=15):
    models_folder = Path("models/")

    bestfeatures = SelectKBest(score_func=f_classif, k=k) # f_classif, f_regression
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(k, 'Score'))  #print 10 best features

    print(featureScores.nlargest(k,'Score')['Specs'].values.tolist())

    best_attributes = featureScores.nlargest(k,'Score')['Specs'].values.tolist()

    pickle.dump(best_attributes, open(models_folder/'attributes.pkl', "wb"))

    X = X[best_attributes]

    return X, y

# TODO - check if this is filter based
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


#################################
# Wrapper Based Feature Selection
#################################
def sequential_selection(classifier, forward, X, y, k_features=12):
    models_folder = Path("models/")
    sfs = SFS(classifier,
          k_features=k_features,
          forward=forward,
          floating=False,
          scoring = 'roc_auc',
          cv = 0)

    sfs.fit(X, y)
    print(list(sfs.k_feature_names_))
    best_attributes = list(sfs.k_feature_names_)

    pickle.dump(best_attributes, open(models_folder/'attributes.pkl', "wb"))

    X = X[best_attributes]

    return X, y


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
    elif classifier == 'knn':
        return KNeighborsClassifier(random_state=RS)
    elif classifier == 'neural_network':
        return MLPClassifier(random_state=RS) # TODO: random_state
    elif classifier == 'xgboost':
        return XGBClassifier()

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

    elif classifier == 'knn':
        return {'n_neighbors': [3, 5, 7, 9],
          'weights': ['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
          'leaf_size':[20, 30, 40],
          'p':[1, 2, 3],
          'metric':['minkowski'],
          'metric_params':[None],
          'n_jobs':[None, 5]}

    elif classifier == 'neural_network':
        return {'hidden_layer_sizes': [(3,5,8,13,21,34)],
          'activation': ['identity', 'logistic', 'tanh', 'relu'],
          'solver':['lbfgs', 'sgd', 'adam'],
          'learning_rate':['constant', 'invscaling', 'adaptive'],
          'nesterovs_momentum':[True, False],
          'early_stopping':[False, True]}
        
    elif classifier == 'xgboost':
        return {'booster': ['gbtree', 'gblinear', 'dart']}

    
def get_classifier_best(classifier):
    if classifier == 'decision_tree':
        # return DecisionTreeClassifier(criterion='entropy', splitter='random')
        return DecisionTreeClassifier()
    elif classifier == 'logistic_regression':
        return LogisticRegression(C = 0.1, class_weight= 'balanced', penalty= 'l2', solver= 'saga', max_iter=200)
    elif classifier == 'random_forest':
        return RandomForestClassifier(class_weight= 'balanced_subsample', criterion= 'gini', max_depth= 30)
    elif classifier == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=RS, criterion='friedman_mse', learning_rate=0.7, loss= 'exponential', min_samples_leaf= 6, min_samples_split= 4, n_estimators= 12)
    elif classifier == 'svm':
        return SVC(random_state=RS, C= 1, class_weight= 'balanced', coef0= 0.0, decision_function_shape= 'ovo', degree= 5, gamma= 'scale', kernel= 'poly', max_iter= 1000, probability=True)
    elif classifier == 'knn':
        return KNeighborsClassifier(n_neighbors=5, weights='distance', leaf_size=20, p=1)
    elif classifier == 'neural_network':
        return MLPClassifier(activation='tanh', hidden_layer_sizes= (3, 5, 8, 13, 21, 34), solver='lbfgs', max_iter=300)
    elif classifier == 'xgboost':
        return XGBClassifier(colsample_bytree=0.8, gamma= 1.5, max_depth= 30, min_child_weight= 1, subsample= 1.0,)
    elif classifier == 'bagging':
        return BaggingClassifier(bootstrap_features= True, max_features= 1.0, max_samples= 1.0, n_estimators= 20)

###########
# Normalize
###########

def not_tree_based(classifier_name):
    return classifier_name not in ['decision_tree','random_forest','xgboost','gradient_boosting']

def normalize_if_not_tree_based(df, classifier_name, scaler):
    if (not_tree_based(classifier_name)):
        return normalize(df,scaler)
    return df

def normalize(df, scaler):
    transformed = scaler.fit_transform(df)
    df = pd.DataFrame(transformed, index=df.index, columns=df.columns)
    return df


if __name__ == "__main__":
    if(DEBUG):
        grid_search(sys.argv[1], sys.argv[2])
    else:
        train(sys.argv[1], sys.argv[2])
    