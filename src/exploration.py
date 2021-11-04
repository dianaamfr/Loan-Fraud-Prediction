import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, auc, roc_curve

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)

loan_train_csv = pd.read_csv('../ficheiros_competicao/loan_train.csv', delimiter=";", low_memory=False)
loan_test_csv = pd.read_csv('../ficheiros_competicao/loan_test.csv', delimiter=";", low_memory=False)

loan_train_csv.columns.drop('account_id')

feature_cols = ['loan_id', 'date', 'amount', 'duration', 'payments']
X = loan_train_csv[feature_cols] # Features
y = loan_train_csv.status # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

## Grid Search for Decision Tree

decision_tree_classifier = DecisionTreeClassifier()

parameter_grid = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best'], 
                  'max_depth': [4, 6, 8, 10],
                  'max_features': ['auto'],
                  'min_samples_leaf': [1, 2, 4, 6],
                  'class_weight': ['balanced', None] }

cross_validation = StratifiedKFold()

# Not sure if using this custom scorer is equivalent to using scoring='roc_auc'
'''
def auc_scorer(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)[:, 1]
'''

# roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
grid_search = GridSearchCV(
    decision_tree_classifier, 
    param_grid = parameter_grid, 
    cv = cross_validation, 
    scoring='roc_auc')
grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

y_pred_proba = grid_search.predict_proba(X_test)
# Keep probabilities for the positive outcome only
prob_loan = y_pred_proba[:, 1]
# Evaluate Prediction
print()
auc = roc_auc_score(y_test,prob_loan)
print(f"AUC: {auc}")
print()

#print(prob_loan)
#print()

# Predict

test_data = loan_test_csv[feature_cols]
prediction = grid_search.predict_proba(test_data)[:,1]
#print(prediction)




## Grid Search for Logistic Regression - TODO?

# loan_train_csv = pd.read_csv('../ficheiros_competicao/loan_train.csv', delimiter=";", low_memory=False)
# data = pd.read_csv(loan_train_csv)

# #define the predictor variables and the response variable
# X = data[feature_cols]
# y = data['status']

# #split the dataset into training (70%) and testing (30%) sets
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

# #instantiate the model
# log_regression = LogisticRegression()

# #fit the model using the training data
# log_regression.fit(X_train,y_train)

# #use model to predict probability that given y value is 1
# y_pred_proba = log_regression.predict_proba(X_test)[::,1]

# #calculate AUC of model
# auc = metrics.roc_auc_score(y_test, y_pred_proba)

# #print AUC score
# print(auc)
