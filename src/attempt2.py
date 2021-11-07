import database
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV

db = database.Database('bank_database')
feature_cols = ['granted_date', 'amount', 'duration', 'payments']

x_query = 'SELECT {0} FROM loan_train;'.format(','.join(feature_cols))
y_query = 'SELECT loan_status FROM loan_train;'

X = db.df_query(x_query)
y = db.df_query(y_query)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logistic_regression_classifier = LogisticRegression()
cross_validation = KFold()

parameter_grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}

# parameter_grid = {'penalty': ['none',  'l2', 'elasticnet'],
#                   'dual': [True, False], 
#                   'tol': [1e-3, 1e-4, 1e-5],
#                   'fit_intercept': [True, False],
#                   'intercept_scaling': [0.5, 1, 2],
#                   'class_weight': [None, 'balanced'],
#                   'random_state': [1, 2, 3],
#                   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#                   'max_iter': [50, 100, 150],
#                   'multi_class': ['auto', 'ovr', 'multinomial'],
#                   'verbose': [0, 1, 2],
#                   'warm_start': [True, False],
#                   'n_jobs': [1, 2, 4],   
#                   'l1_ratio': [None, 1.0, 2.0]}

grid_search = GridSearchCV(
    logistic_regression_classifier, 
    param_grid = parameter_grid, 
    cv = cross_validation, 
    scoring='roc_auc')

grid_search.fit(X_train, y_train.values.ravel())

y_pred = grid_search.predict_proba(X_test)
prob_loan = y_pred[::, 1]
auc = roc_auc_score(y_test, prob_loan)

print('Train Prediction:')
print(y_pred)
print(f"AUC: {auc}")



# Test Model
x_query_test = 'SELECT {0} FROM loan_test;'.format(','.join(feature_cols))
X_test = db.df_query(x_query_test)
prediction = grid_search.predict_proba(X_test)[::,1]

print('TEST PREDICTION:')
print(prediction)
  
# Create the pandas DataFrame
df_result = pd.DataFrame()

loan_id_query = 'SELECT loan_id FROM loan_test;'
loan_ids = db.df_query(loan_id_query)

df_result['Id'] = loan_ids
df_result['Predicted'] = prediction

df_result.to_csv('results/results2.csv', sep=',', index=False)