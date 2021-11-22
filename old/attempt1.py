import database
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd

db = database.Database('bank_database')

feature_cols = ['granted_date', 'amount', 'duration', 'payments']

x_query = 'SELECT {0} FROM loan_train;'.format(','.join(feature_cols))
y_query = 'SELECT loan_status FROM loan_train;'

X = db.df_query(x_query)
y = db.df_query(y_query)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

## Grid Search for Decision Tree
decision_tree_classifier = DecisionTreeClassifier()
cross_validation = KFold()

parameter_grid = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best'], 
                  'max_depth': [4, 6, 8, 10],
                  'max_features': ['auto'],
                  'min_samples_leaf': [1, 2, 4, 6],
                  'class_weight': ['balanced', None] }

grid_search = GridSearchCV(
    decision_tree_classifier, 
    param_grid = parameter_grid, 
    cv = cross_validation, 
    scoring='roc_auc')

grid_search.fit(X_train, y_train)

# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

y_pred_proba = grid_search.predict_proba(X_test)
prob_loan = y_pred_proba[::, 1]
auc = roc_auc_score(y_test,prob_loan)

print('Train Prediction:')
print(prob_loan)
print(f"AUC: {auc}")

x_query_test = 'SELECT {0} FROM loan_test;'.format(','.join(feature_cols))
X_test = db.df_query(x_query_test)
prediction = grid_search.predict_proba(X_test)[:,1]

print('TEST PREDICTION')
#print(prediction)
  
# Create the pandas DataFrame
df_result = pd.DataFrame()

loan_id_query = 'SELECT loan_id FROM loan_test;'
loan_ids = db.df_query(loan_id_query)
  
# print dataframe.
df_result['Id'] = loan_ids
df_result['Predicted'] = prediction

df_result.to_csv('results/results1.csv', sep=',', index=False)