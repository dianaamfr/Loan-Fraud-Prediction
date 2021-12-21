import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

RS = 42
BASE_ESTIMATOR = 'random_forest'
TREE_BASED_CLASSIFIERS = ['decision_tree','random_forest','xgboost','gradient_boosting']

###############
# Normalization
###############

def normalize_if_not_tree_based(df, classifier_name, scaler):
    if not_tree_based(classifier_name):
        return normalize(df,scaler)
    return df

def not_tree_based(classifier_name):
    if classifier_name == 'bagging':
        return BASE_ESTIMATOR not in TREE_BASED_CLASSIFIERS
    else:
        return classifier_name not in TREE_BASED_CLASSIFIERS

def normalize(df, scaler):
    df_copy = df.copy()
    transformed = scaler.fit_transform(df_copy)
    df_normalized = pd.DataFrame(transformed, index=df_copy.index, columns=df_copy.columns)
    return df_normalized

#############
# Classifiers
#############

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
        return MLPClassifier(random_state=RS)
    elif classifier == 'xgboost':
        return XGBClassifier(use_label_encoder=False, eval_metric='auc')
    elif classifier == 'bagging':
        return BaggingClassifier(get_classifier(BASE_ESTIMATOR))

def get_grid_params(classifier):
    if classifier == 'decision_tree':
        return {'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [3,5,7,10],
                'min_samples_split': [1,2,3],
                'min_samples_leaf': [1,2,3],
                'min_weight_fraction_leaf': [0.0],
                'max_features': [None, 'auto', 'sqrt', 'log2', 12, 15],
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
        return {'min_child_weight':range(1,6,2),
                'gamma': [0,0.5,1,1.5],
                'max_depth': [5, 14, 30],
                'reg_alpha':[1e-5, 1e-2, 0.1, 1]}

    
def get_classifier_best(classifier):
    if classifier == 'decision_tree':
        return DecisionTreeClassifier(criterion='entropy', max_features='auto', max_depth=10, min_samples_leaf=3)
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
        return XGBClassifier(colsample_bytree=0.8, gamma= 1.5, 
                    max_depth= 30, min_child_weight= 1, subsample= 1.0,
                    use_label_encoder=False, eval_metric='auc')
    elif classifier == 'bagging':
        return BaggingClassifier(get_classifier_best(BASE_ESTIMATOR), random_state=RS, n_jobs=-1)

