import sys
import graphviz
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from statistics import mean
from utils import *
from feature_selection import *
from xgboost import plot_tree
from sklearn import tree

DEBUG = False
CROSS_VALIDATION = True

#######
# Train
#######

def train(classifier_name, submission_name):
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)

    # Scaling
    scaler = MinMaxScaler()
    df = normalize_if_not_tree_based(df, classifier_name, scaler)

    # Define goal feature
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Apply SMOTE on imbalanced dataset
    oversample = SMOTE(random_state=RS)
    X, y = oversample.fit_resample(X, y)

    # Get Classifier
    classifier = get_classifier_best(classifier_name)

    # Cross Validation
    num_splits = 3
    kf = KFold(num_splits, random_state=RS, shuffle=True)

    # Feature Selection
    # X = recursive_cv_feature_selection(X, y, classifier, kf)
    # X = recursive_feature_selection(X, y, classifier)
    X = filter_feature_selection(X, y)
    # X = extra_tree_feature_selection(X,y)

    # Fit Classifier and Predict
    if CROSS_VALIDATION:
        cross_validation(X, y, classifier, kf, num_splits)
    else:
        no_cross_validation(X, y, classifier, kf)

    # Save model
    models_folder = Path("models/")
    filename = models_folder/(classifier_name + '-' + submission_name + '.sav')
    joblib.dump(classifier, filename)


##################
# Cross-Validation
##################

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

    classifier_name = type(classifier).__name__

    if classifier_name == 'XGBClassifier':
        plt.figure(figsize=(12,12))
        plot_tree(classifier, fontsize=6)
        plt.savefig('models/tree.jpg', dpi=400)
        plt.clf()
    elif classifier_name == 'DecisionTreeClassifier': 
        fn=X.columns.tolist()
        cn=['default', 'paid']
        
        dot_data = tree.export_graphviz(classifier, 
            out_file=None, 
            feature_names=fn,  
            class_names=cn,
            filled=True)

        graph = graphviz.Source(dot_data, format="png") 
        graph.render("models/decision_tree")
        
    print('AUC Train scores of each fold - {}'.format(auc_train_scores))
    print('AUC Test scores of each fold - {}'.format(auc_test_scores))

    print('Average Train AUC = ',round(mean(auc_train_scores),3))
    print('Average Test AUC = ',round(mean(auc_test_scores),6))

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

##############
# Grid Search
#############
def grid_search(classifier_name, submission_name):

    # Split dataset into training set and test set
    df = pd.read_csv('clean_data/' + submission_name + '-train.csv', delimiter=",", low_memory=False)

    scaler = MinMaxScaler()
    df = normalize_if_not_tree_based(df, classifier_name, scaler)
    #normalized_df = normalize(df, scaler)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    # X_normalized = normalized_df.drop(columns=['loan_status'])
    # y_normalized = normalized_df['loan_status']

    # Apply SMOTE on imbalanced dataset
    oversample = SMOTE(random_state=RS)
    X, y = oversample.fit_resample(X, y)
    # X_normalized, y_normalized = oversample.fit_resample(X_normalized, y_normalized)

    # Get classifier
    classifier = get_classifier(classifier_name)

    # Cross-Validation
    num_splits = 3
    kf = KFold(num_splits, random_state=RS, shuffle=True)

    # Feature Selection
    # X = recursive_cv_feature_selection(X, y, classifier, kf)
    # X = recursive_feature_selection(X, y, classifier)
    X = filter_feature_selection(X, y)
    # X_normalized = filter_feature_selection(X_normalized, y_normalized)
    # X = extra_tree_feature_selection(X,y)

    params = get_grid_params(classifier_name)
    grid_search_var = GridSearchCV(
        estimator=classifier,
        param_grid = params,
        scoring='roc_auc',
        cv=kf,
        n_jobs = -1)

    grid_results = grid_search_var.fit(X, y)

    #compare_classifiers(kf, X, y, X_normalized, y_normalized)

    print('Best Parameters: ', grid_results.best_params_)
    print('Best Score: ', grid_results.best_score_)


if __name__ == "__main__":
    if(DEBUG):
        grid_search(sys.argv[1], sys.argv[2])
    else:
        train(sys.argv[1], sys.argv[2])
    