from pathlib import Path
import pandas as pd
import pickle
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectKBest, RFECV, RFE
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from utils import *

K_FEATURES = 16
MODELS_FOLDER = Path("models/")

##############
# Filter Based
##############

def filter_feature_selection(X, y):
    """
    Univariate feature selectiona that removes all features except the k with highest score.
    For classification: chi2, f_classif, mutual_info_classif
    """

    best_features = SelectKBest(score_func=f_classif, k=K_FEATURES) # f_classif, f_regression
    fit = best_features.fit(X,y)
    
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns,df_scores],axis=1)
    feature_scores.columns = ['Specs','Score']
    feature_scores = feature_scores.nlargest(K_FEATURES,'Score')

    selected_features = feature_scores['Specs'].values.tolist()

    print(feature_scores) 
    print(selected_features)

    pickle.dump(selected_features, open(MODELS_FOLDER/'features.pkl', "wb"))

    return X[selected_features]


###############
# Wrapper Based
###############

def recursive_cv_feature_selection(X, y, classifier, cv):
    """
    Recursive Feature Elimination with cross validation (RFECV) / Backward selection
    Select features by recursively considering smaller and smaller sets of features.
    Trains the model on the original number of features, giving an importance to each 
    feature and kicking the least importantout. 
    """
    rfecv = RFECV(classifier,scoring='roc_auc', cv=cv) 
    rfecv.fit(X,y)

    df_scores = pd.DataFrame(rfecv.support_) 
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns,df_scores],axis=1)
    feature_scores.columns = ['Feature','Selected']

    selected_features = feature_scores[feature_scores['Selected'] == True]['Feature'].values.tolist()

    print(selected_features)

    pickle.dump(selected_features, open(MODELS_FOLDER/'features.pkl', "wb"))

    return X[selected_features]

def recursive_feature_selection(X, y, classifier):
    """
    Recursive Feature Elimination (RFE) / Backward selection
    Select features by recursively considering smaller and smaller sets of features.
    Trains the model on the original number of features, giving an importance to each 
    feature and kicking the least importantout. 
    """
    rfecv = RFE(classifier, n_features_to_select=K_FEATURES) 
    rfecv.fit(X,y)

    df_scores = pd.DataFrame(rfecv.support_) 
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns,df_scores],axis=1)
    feature_scores.columns = ['Feature','Selected']

    selected_features = feature_scores[feature_scores['Selected'] == True]['Feature'].values.tolist()

    print(selected_features)

    pickle.dump(selected_features, open(MODELS_FOLDER/'features.pkl', "wb"))

    return X[selected_features]

def sequential_feature_selection(classifier, forward, X, y, k_features=12):
    """
    Sequential Feature Selection can be either forward or backward:
    Forward-SFS is a greedy procedure that iteratively finds the best new 
    feature to add to the set of selected features.
    Backward-SFS follows the same idea but works in the opposite direction, 
    starting with all the features and greedily removing features from the set.
    """
    sfs = SFS(classifier,
          k_features=k_features,
          forward=forward,
          floating=False,
          scoring = 'roc_auc',
          cv = 0)

    sfs.fit(X, y)

    selected_features = list(sfs.k_feature_names_)
    print(selected_features)

    pickle.dump(selected_features, open(MODELS_FOLDER/'attributes.pkl', "wb"))

    return X[selected_features]


def extra_tree_feature_selection(X,y):
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, y)

    df_scores = pd.DataFrame(model.feature_importances_)
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns,df_scores],axis=1)
    feature_scores.columns = ['Specs','Score']
    feature_scores = feature_scores.nlargest(K_FEATURES,'Score')

    selected_features = feature_scores['Specs'].values.tolist()

    print(feature_scores) 
    print(selected_features)

    pickle.dump(selected_features, open(MODELS_FOLDER/'features.pkl', "wb"))

    return X[selected_features]